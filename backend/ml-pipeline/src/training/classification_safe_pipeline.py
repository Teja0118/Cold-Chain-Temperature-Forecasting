"""Leakage-safe logistics-action classification pipeline.

This module deliberately uses only operational sensor inputs.  The true
``Forecasted_4Hr_Spoilage_Risk_Pct`` column is excluded because it is an
outcome; a future production integration may supply an out-of-fold regression
prediction in its place.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from src.training.regression_pipeline import _make_pipeline as make_regression_pipeline
from src.prediction.prediction_utils import (
    is_emergency_condition,
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names"
)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names"
)

TARGET = "Logistics_Action_Recommendation"
SAFE = "SAFE_Maintain_Course"
WARNING = "WARNING_Request_HVAC_Remote_Reset"
EMERGENCY = "EMERGENCY"
INPUT_COLUMNS = [
    "Cargo_Type", "Ambient_External_Temp_C", "Internal_Set_Point_C",
    "Actual_Internal_Temp_C", "HVAC_Power_Consumption_Watts", "Seal_Integrity_Index",
]
PREDICTED_RISK = "Predicted_Spoilage_Risk_Pct"
CLASSIFIER_INPUT_COLUMNS = INPUT_COLUMNS + [PREDICTED_RISK]
NUMERIC_COLUMNS = INPUT_COLUMNS[1:] + [

    "Temperature_Deviation",

    "Positive_Temperature_Deviation",

    "Ambient_Load",

    "Seal_Loss",

    "Thermal_Leak_Load",

    "HVAC_Load_Per_Degree",

    "Cooling_Efficiency",

    "HVAC_Stress",

    "Ambient_Stress",

    "Heat_Ingress",

    PREDICTED_RISK,

]


class ClassificationFeatureBuilder(BaseEstimator, TransformerMixin):
    """Create physical features from raw values before scaling."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X[CLASSIFIER_INPUT_COLUMNS].copy()
        df["Temperature_Deviation"] = df["Actual_Internal_Temp_C"] - df["Internal_Set_Point_C"]
        df["Positive_Temperature_Deviation"] = df["Temperature_Deviation"].clip(lower=0)
        df["Ambient_Load"] = df["Ambient_External_Temp_C"] - df["Internal_Set_Point_C"]
        df["Seal_Loss"] = 1 - df["Seal_Integrity_Index"]
        df["Thermal_Leak_Load"] = df["Ambient_Load"].clip(lower=0) * df["Seal_Loss"]
        df["HVAC_Load_Per_Degree"] = (
            df["HVAC_Power_Consumption_Watts"] / (df["Ambient_Load"].abs() + 1)
        )
        # Cooling efficiency
        df["Cooling_Efficiency"] = (
            (df["Ambient_External_Temp_C"] - df["Actual_Internal_Temp_C"]) /
            (df["HVAC_Power_Consumption_Watts"] + 1)
        )

        # HVAC stress relative to temperature deviation
        df["HVAC_Stress"] = (
            df["HVAC_Power_Consumption_Watts"] /
            (df["Positive_Temperature_Deviation"] + 1)
        )

        # Environmental stress
        df["Ambient_Stress"] = (
            df["Ambient_External_Temp_C"] *
            df["Seal_Loss"]
        )

        # Heat ingress estimate
        df["Heat_Ingress"] = (
            (df["Ambient_External_Temp_C"] - df["Actual_Internal_Temp_C"]) *
            df["Seal_Loss"]
        )
        return df

###############################################################################
# Physics Aware Synthetic Data Generator
###############################################################################

class ColdChainSyntheticGenerator:
    """
    Physics-aware synthetic data generator for cold-chain logistics.

    This generator creates realistic WARNING and EMERGENCY operating
    conditions by preserving relationships between:

        Ambient Temperature
                ↓
        Thermal Load
                ↓
        HVAC Demand
                ↓
        Internal Temperature
                ↓
        Spoilage Risk

    Unlike simple Gaussian sampling, every synthetic sample belongs to
    a realistic operational scenario.
    """

    WARNING_SCENARIOS = {
        "high_ambient": 0.25,
        "hvac_degradation": 0.20,
        "seal_degradation": 0.20,
        "door_open": 0.10,
        "refrigeration_delay": 0.10,
        "combined_warning": 0.10,
        "operational_noise": 0.05,
    }

    EMERGENCY_SCENARIOS = {
        "compressor_failure": 0.20,
        "refrigerant_leak": 0.15,
        "seal_failure": 0.15,
        "door_left_open": 0.15,
        "hvac_shutdown": 0.15,
        "extreme_ambient": 0.10,
        "cascading_failure": 0.10,
    }

    def __init__(self, random_state=42):

        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)

        # Statistical profile learned from real WARNING samples
        self.warning_profile = None

        # Statistical profile learned from SAFE samples
        self.safe_profile = None

    def fit_profiles(
        self,
        safe_df: pd.DataFrame,
        warning_df: pd.DataFrame,
    ):
        """
        Learn statistical profiles from the real SAFE and WARNING data.
        """

        numeric_columns = [

            "Ambient_External_Temp_C",

            "Internal_Set_Point_C",

            "Actual_Internal_Temp_C",

            "HVAC_Power_Consumption_Watts",

            "Seal_Integrity_Index",

            "Forecasted_4Hr_Spoilage_Risk_Pct",

        ]

        self.warning_profile = {}

        self.safe_profile = {}

        for column in numeric_columns:

            self.warning_profile[column] = {

                "mean": warning_df[column].mean(),

                "std": warning_df[column].std(),

                "min": warning_df[column].min(),

                "max": warning_df[column].max(),

                "q05": warning_df[column].quantile(0.05),

                "q95": warning_df[column].quantile(0.95),

            }

            self.safe_profile[column] = {

                "mean": safe_df[column].mean(),

                "std": safe_df[column].std(),

                "min": safe_df[column].min(),

                "max": safe_df[column].max(),

                "q05": safe_df[column].quantile(0.05),

                "q95": safe_df[column].quantile(0.95),

            }

    def generate(
        self,
        seeds: pd.DataFrame,
        count: int,
        kind: str,
    ) -> pd.DataFrame:

        """
        Generate synthetic rows.

        Parameters
        ----------
        seeds : Real WARNING samples

        count : Number of rows required

        kind : warning / emergency
        """

        synthetic_rows = []

        for i in range(count):

            row = (
                seeds.sample(
                    n=1,
                    replace=True,
                    random_state=None,
                )
                .iloc[0]
                .copy()
            )

            if kind == "warning":

                row = self._generate_warning(row)

                row[TARGET] = WARNING

                row["Data_Origin"] = "synthetic_warning"

            else:

                row = self._generate_emergency(row)

                row[TARGET] = EMERGENCY

                row["Data_Origin"] = "synthetic_emergency"

            row["Container_ID"] = (
                f"SYN_{kind.upper()}_{i:06d}"
            )

            synthetic_rows.append(row)

        return pd.DataFrame(synthetic_rows)
    
        ###########################################################################
    # WARNING Generator
    ###########################################################################

    def _generate_warning(self, row):
        """
        Generate a WARNING sample that stays close to the real WARNING
        distribution while preserving correlations.
        """

        profile = self.warning_profile

        numeric_columns = [
            "Ambient_External_Temp_C",
            "Internal_Set_Point_C",
            "Actual_Internal_Temp_C",
            "HVAC_Power_Consumption_Watts",
            "Seal_Integrity_Index",
            "Forecasted_4Hr_Spoilage_Risk_Pct",
        ]

        for column in numeric_columns:

            std = profile[column]["std"]

            if pd.isna(std):
                continue

            noise = self.rng.normal(
                0,
                std * 0.12
            )

            row[column] += noise

            row[column] = np.clip(
                row[column],
                profile[column]["q05"],
                profile[column]["q95"]
            )

        return self._enforce_warning_constraints(row)

    def _enforce_warning_constraints(self, row):
        """
        Ensure WARNING samples remain physically realistic.
        """

        row["Seal_Integrity_Index"] = np.clip(
            row["Seal_Integrity_Index"],
            0.70,
            1.00,
        )

        row["HVAC_Power_Consumption_Watts"] = max(
            row["HVAC_Power_Consumption_Watts"],
            100,
        )

        row["Actual_Internal_Temp_C"] = max(
            row["Actual_Internal_Temp_C"],
            row["Internal_Set_Point_C"] - 1.0,
        )

        row["Forecasted_4Hr_Spoilage_Risk_Pct"] = np.clip(
            row["Forecasted_4Hr_Spoilage_Risk_Pct"],
            20,
            70,
        )

        return row

    ###############################################################################
    # EMERGENCY Generator
    ###############################################################################

    def _generate_emergency(self, row):
        """
        Generate realistic EMERGENCY samples based on distinct failure modes.
        """

        scenario = self.rng.choice(
            [
                "compressor_failure",
                "power_failure",
                "door_open",
                "seal_failure",
                "refrigerant_leak",
                "extreme_ambient",
                "combined_failure",
            ],
            p=[
                0.20,
                0.15,
                0.15,
                0.15,
                0.15,
                0.10,
                0.10,
            ],
        )

        return getattr(self, f"_{scenario}")(row)

    ###############################################################################
    # Compressor Failure
    ###############################################################################

    def _compressor_failure(self, row):

        row["Ambient_External_Temp_C"] += self.rng.uniform(2, 5)

        row["HVAC_Power_Consumption_Watts"] *= self.rng.uniform(1.25, 1.50)

        row["Actual_Internal_Temp_C"] += self.rng.uniform(6, 10)

        row["Forecasted_4Hr_Spoilage_Risk_Pct"] = self.rng.uniform(80, 100)

        return self._enforce_emergency_constraints(row)
    
    def _power_failure(self, row):

        row["HVAC_Power_Consumption_Watts"] *= self.rng.uniform(0.05, 0.20)

        row["Actual_Internal_Temp_C"] += self.rng.uniform(8, 12)

        row["Forecasted_4Hr_Spoilage_Risk_Pct"] = self.rng.uniform(90, 100)

        return self._enforce_emergency_constraints(row)
    
    def _door_open(self, row):

        row["Seal_Integrity_Index"] *= self.rng.uniform(0.25, 0.50)

        row["Actual_Internal_Temp_C"] += self.rng.uniform(5, 8)

        row["HVAC_Power_Consumption_Watts"] *= self.rng.uniform(1.10, 1.30)

        row["Forecasted_4Hr_Spoilage_Risk_Pct"] = self.rng.uniform(80, 95)

        return self._enforce_emergency_constraints(row)
    
    def _seal_failure(self, row):

        row["Seal_Integrity_Index"] *= self.rng.uniform(0.15, 0.40)

        row["Actual_Internal_Temp_C"] += self.rng.uniform(6, 10)

        row["HVAC_Power_Consumption_Watts"] *= self.rng.uniform(1.20, 1.45)

        row["Forecasted_4Hr_Spoilage_Risk_Pct"] = self.rng.uniform(80, 100)

        return self._enforce_emergency_constraints(row)
    
    def _refrigerant_leak(self, row):

        row["HVAC_Power_Consumption_Watts"] *= self.rng.uniform(1.20, 1.40)

        row["Actual_Internal_Temp_C"] += self.rng.uniform(5, 8)

        row["Forecasted_4Hr_Spoilage_Risk_Pct"] = self.rng.uniform(75, 95)

        return self._enforce_emergency_constraints(row)
    
    def _extreme_ambient(self, row):

        row["Ambient_External_Temp_C"] = self.rng.uniform(42, 48)

        row["HVAC_Power_Consumption_Watts"] *= self.rng.uniform(1.20, 1.40)

        row["Actual_Internal_Temp_C"] += self.rng.uniform(5, 8)

        row["Forecasted_4Hr_Spoilage_Risk_Pct"] = self.rng.uniform(75, 90)

        return self._enforce_emergency_constraints(row)
    
    def _combined_failure(self, row):

        row = self._compressor_failure(row)

        row["Seal_Integrity_Index"] *= self.rng.uniform(0.40, 0.70)

        row["Actual_Internal_Temp_C"] += self.rng.uniform(2, 4)

        row["Forecasted_4Hr_Spoilage_Risk_Pct"] = 100

        return self._enforce_emergency_constraints(row)
    
    def _enforce_emergency_constraints(self, row):
        """
        Keep emergency samples physically valid.
        """

        row["Seal_Integrity_Index"] = np.clip(
            row["Seal_Integrity_Index"],
            0.10,
            0.70,
        )

        row["Actual_Internal_Temp_C"] = max(
            row["Actual_Internal_Temp_C"],
            row["Internal_Set_Point_C"] + 5,
        )

        row["HVAC_Power_Consumption_Watts"] = max(
            row["HVAC_Power_Consumption_Watts"],
            0,
        )

        row["Forecasted_4Hr_Spoilage_Risk_Pct"] = np.clip(
            row["Forecasted_4Hr_Spoilage_Risk_Pct"],
            70,
            100,
        )

        return row

    
    ###############################################################################
    # Door Left Open
    ###############################################################################

    def _door_left_open(self, row):

        row["Seal_Integrity_Index"] *= self.rng.uniform(
            0.25,
            0.55
        )

        row["Ambient_External_Temp_C"] = self.rng.uniform(
            35,
            45
        )

        row["Actual_Internal_Temp_C"] += self.rng.uniform(
            10,
            18
        )

        row["HVAC_Power_Consumption_Watts"] *= self.rng.uniform(
            1.20,
            1.45
        )

        return row
    
    ###############################################################################
    # HVAC Shutdown
    ###############################################################################

    def _hvac_shutdown(self, row):

        row["HVAC_Power_Consumption_Watts"] *= self.rng.uniform(
            0.10,
            0.40
        )

        row["Actual_Internal_Temp_C"] += self.rng.uniform(
            10,
            16
        )

        row["Ambient_External_Temp_C"] = self.rng.uniform(
            35,
            45
        )

        return row
     
    ###############################################################################
    # Cascading Failure
    ###############################################################################

    def _cascading_failure(self, row):

        row = self._compressor_failure(row)

        row = self._seal_failure(row)

        row["Actual_Internal_Temp_C"] += self.rng.uniform(
            3,
            6
        )

        return row
    


def _make_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        ("cargo", OneHotEncoder(handle_unknown="ignore"), ["Cargo_Type"]),
    ])
    return Pipeline([
        ("features", ClassificationFeatureBuilder()),
        ("preprocessor", preprocessor),
        ("model", model),
    ])


class ClassificationPipeline:
    """Build a 30k / 7.5k / 7.5k training set and evaluate on real holdouts."""

    def __init__(self, data_path: str, version: str = "v2", random_state: int = 42):
        self.data_path = Path(data_path)
        self.version = version
        self.random_state = random_state
        self.model_dir = Path("models/classification") / version
        self.report_dir = Path("reports/classification") / version
        self.data_dir = Path("data/balanced")
        self.encoder = LabelEncoder()

    def _load_and_split(self):
        df = pd.read_csv(self.data_path)
        required = set(INPUT_COLUMNS + [TARGET, "Container_ID"])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        # The raw data has no Emergency class; this split is intentionally real-only.
        train, temporary = train_test_split(
            df, test_size=0.30, random_state=self.random_state, stratify=df[TARGET]
        )
        validation, test = train_test_split(
            temporary, test_size=0.50, random_state=self.random_state,
            stratify=temporary[TARGET]
        )
        return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)

    def _build_training_data(self, train: pd.DataFrame):

        safe = train[train[TARGET] == SAFE].copy()

        warning = train[train[TARGET] == WARNING].copy()

        if len(safe) < 30000:

            raise ValueError(
                "Need at least 30,000 SAFE samples."
            )

        if warning.empty:

            raise ValueError(
                "No WARNING samples found."
            )

        ###################################################################
        # Real SAFE
        ###################################################################

        real_safe = safe.sample(
            n=30000,
            random_state=self.random_state
        ).copy()

        real_safe["Data_Origin"] = "real"

        ###################################################################
        # Real WARNING
        ###################################################################

        real_warning = warning.copy()

        real_warning["Data_Origin"] = "real"

        ###################################################################
        # Physics-aware Generator
        ###################################################################

        generator = ColdChainSyntheticGenerator(

            random_state=self.random_state

        )

        generator.fit_profiles(

            safe_df=real_safe,

            warning_df=real_warning,

        )

        synthetic_warning = generator.generate(

            seeds=warning,

            count=7500 - len(real_warning),

            kind="warning"

        )

        synthetic_emergency = generator.generate(

            seeds=warning,

            count=7500,

            kind="emergency"

        )

        ###################################################################
        # Merge
        ###################################################################

        balanced = pd.concat(

            [

                real_safe,

                real_warning,

                synthetic_warning,

                synthetic_emergency

            ],

            ignore_index=True

        )

        balanced = balanced.sample(

            frac=1,

            random_state=self.random_state

        ).reset_index(

            drop=True

        )

        expected = {

            SAFE:30000,

            WARNING:7500,

            EMERGENCY:7500

        }

        actual = (

            balanced[TARGET]

            .value_counts()

            .to_dict()

        )

        if actual != expected:

            raise RuntimeError(

                f"Unexpected class balance : {actual}"

            )

        return balanced

    @staticmethod
    def _candidates(random_state: int):
        return {
            "LightGBM": LGBMClassifier(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.9,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
                class_weight="balanced",      # ADD
            ),

            "RandomForest": RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",      # ADD
            ),

            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",      # ADD
            ),
        }
    
    @staticmethod
    def _tune_model(model, model_name, random_state):

        param_grid = {

            "LightGBM": {

                "model__n_estimators": [300, 500, 700],

                "model__learning_rate": [0.03, 0.05, 0.08],

                "model__num_leaves": [31, 50, 80],

                "model__subsample": [0.8, 0.9, 1.0],

                "model__colsample_bytree": [0.8, 0.9, 1.0],

            },

            "RandomForest": {

                "model__n_estimators": [300, 400, 500],

                "model__max_depth": [10, 15, 20, None],

                "model__min_samples_leaf": [1, 2, 4],

                "model__min_samples_split": [2, 5, 10],

            },

            "ExtraTrees": {

                "model__n_estimators": [300, 400, 500],

                "model__max_depth": [10, 15, 20, None],

                "model__min_samples_leaf": [1, 2, 4],

                "model__min_samples_split": [2, 5, 10],

            },

        }

        search = RandomizedSearchCV(

            estimator=model,

            param_distributions=param_grid[model_name],

            n_iter=20,

            scoring="f1_macro",

            cv=3,

            random_state=random_state,

            n_jobs=-1,

            verbose=1,

        )

        return search

    def _calibrate_warning_threshold(self, model, validation):
        """
        Select a Warning probability threshold that prioritizes detecting
        WARNING cases while keeping reasonable precision.
        """

        probabilities = model.predict_proba(
            validation[CLASSIFIER_INPUT_COLUMNS]
        )

        warning_code = int(
            self.encoder.transform([WARNING])[0]
        )

        y_true = self.encoder.transform(
            validation[TARGET]
        )

        best_threshold = 0.50
        best_score = -1.0

        for threshold in np.arange(0.05, 0.61, 0.01):

            prediction = np.where(
                probabilities[:, warning_code] >= threshold,
                warning_code,
                self.encoder.transform([SAFE])[0]
            )

            warning_recall = recall_score(
                y_true,
                prediction,
                labels=[warning_code],
                average="macro",
                zero_division=0
            )

            warning_precision = precision_score(
                y_true,
                prediction,
                labels=[warning_code],
                average="macro",
                zero_division=0
            )

            score = (
                0.70 * warning_recall +
                0.30 * warning_precision
            )

            if score > best_score:

                best_score = score
                best_threshold = threshold

        print(f"\nSelected Warning Threshold : {best_threshold:.2f}")
        print(f"Threshold Score           : {best_score:.4f}")

        return best_threshold, best_score

   
    def _predict_actions(self, model, frame: pd.DataFrame, warning_threshold: float):
        probabilities = model.predict_proba(frame[CLASSIFIER_INPUT_COLUMNS])
        safe_code = int(self.encoder.transform([SAFE])[0])
        warning_code = int(self.encoder.transform([WARNING])[0])
        emergency_code = int(self.encoder.transform([EMERGENCY])[0])
        predictions = np.full(len(frame), safe_code, dtype=int)
        predictions[probabilities[:, warning_code] >= warning_threshold] = warning_code
        # Explicit safety override for an Emergency-like physical condition.
        emergency_condition = is_emergency_condition(
            frame
        )
        predictions[emergency_condition.to_numpy()] = emergency_code
        return predictions, probabilities

    def _evaluate_real_holdout(self, model, frame: pd.DataFrame, split_name: str, warning_threshold: float):
        y = self.encoder.transform(frame[TARGET])
        prediction, probability = self._predict_actions(model, frame, warning_threshold)
        warning_code = int(self.encoder.transform([WARNING])[0])
        # Emergency has no real holdout labels; report only observable classes here.
        observed = np.unique(y)
        metrics = {
            "split": split_name,
            "macro_f1_observed_classes": float(f1_score(y, prediction, labels=observed, average="macro", zero_division=0)),
            "warning_recall": float(recall_score(y, prediction, labels=[warning_code], average="macro", zero_division=0)),
            "samples": int(len(frame)), "warning_threshold": warning_threshold,
            "real_emergency_samples": int((frame[TARGET] == EMERGENCY).sum()),
        }
        predicted_labels = self.encoder.inverse_transform(prediction)
        prediction_report = pd.DataFrame({
            "Actual_Class": frame[TARGET].to_numpy(),
            "Predicted_Class": predicted_labels,
            "Predicted_Spoilage_Risk_Pct": frame[PREDICTED_RISK].round(3).to_numpy(),
            "Confidence": probability[np.arange(len(frame)), prediction],
        })
        for class_index, class_name in enumerate(self.encoder.classes_):
            prediction_report[f"Probability_{class_name}"] = probability[:, class_index]
        prediction_report.to_csv(self.report_dir / f"{split_name}_predictions.csv", index=False)
        pd.DataFrame(
            confusion_matrix(y, prediction, labels=np.arange(len(self.encoder.classes_))),
            index=self.encoder.classes_, columns=self.encoder.classes_,
        ).to_csv(self.report_dir / f"{split_name}_confusion_matrix.csv")
        pd.DataFrame(classification_report(
            y, prediction, labels=observed, target_names=self.encoder.inverse_transform(observed),
            output_dict=True, zero_division=0,
        )).transpose().to_csv(self.report_dir / f"{split_name}_classification_report.csv")
        return metrics

    @staticmethod
    def _risk_regressor(random_state: int):
        return LGBMRegressor(
            n_estimators=500, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.9, random_state=random_state,
            n_jobs=-1, verbosity=-1,
        )

    def _add_predicted_risk(self, frame: pd.DataFrame, risk_model):
        """Add risk predicted solely from operational inputs, never actual risk."""
        output = frame.copy()
        output[PREDICTED_RISK] = np.clip(
            risk_model.predict(output[INPUT_COLUMNS]), 0, 100
        )
        return output

    def run(self):
        train, validation, test = self._load_and_split()
        balanced_train = self._build_training_data(train)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        balanced_train.to_csv(self.data_dir / f"classification_train_{self.version}.csv", index=False)

        self.encoder.fit([SAFE, WARNING, EMERGENCY])
        # Fit only on the real classifier-training split.  Validation and test
        # never contribute their risk labels to this derived feature.
        risk_model = make_regression_pipeline(self._risk_regressor(self.random_state))
        risk_model.fit(train[INPUT_COLUMNS], train["Forecasted_4Hr_Spoilage_Risk_Pct"])
        balanced_train = self._add_predicted_risk(balanced_train, risk_model)
        validation = self._add_predicted_risk(validation, risk_model)
        test = self._add_predicted_risk(test, risk_model)
        X_train = balanced_train[CLASSIFIER_INPUT_COLUMNS]
        y_train = self.encoder.transform(balanced_train[TARGET])
        comparison = []
        fitted = {}
        
        for name, estimator in self._candidates(self.random_state).items():

            print(f"\nTuning {name}...")

            model = _make_pipeline(estimator)

            search = self._tune_model(
                model,
                name,
                self.random_state,
            )

            search.fit(X_train, y_train)

            model = search.best_estimator_

            print(f"\n{name}")
            print(search.best_params_)
            print(f"Best CV Score: {search.best_score_:.4f}")

            threshold, _ = self._calibrate_warning_threshold(
                model,
                validation,
            )

            result = self._evaluate_real_holdout(
                model,
                validation,
                f"validation_{name}",
                threshold,
            )

            comparison.append({"Model": name, **result})

            fitted[name] = model

        comparison_df = pd.DataFrame(comparison).sort_values(
            ["macro_f1_observed_classes", "warning_recall"], ascending=False
        ).reset_index(drop=True)
        best_name = comparison_df.iloc[0]["Model"]
        # All real training rows and synthetic rows are now used after selection.
        final_model = fitted[best_name]
        warning_threshold, _ = self._calibrate_warning_threshold(final_model, validation)
        test_metrics = self._evaluate_real_holdout(final_model, test, "test", warning_threshold)

        comparison_df.to_csv(self.report_dir / "validation_model_comparison.csv", index=False)
        pd.DataFrame([test_metrics]).to_csv(self.report_dir / "test_metrics.csv", index=False)
        distribution = balanced_train[TARGET].value_counts().rename_axis("Class").reset_index(name="Count")
        distribution["Percentage"] = (distribution["Count"] / len(balanced_train) * 100).round(3)
        distribution.to_csv(self.report_dir / "training_class_distribution.csv", index=False)
        joblib.dump(
            {"pipeline": final_model, "label_encoder": self.encoder, "input_columns": INPUT_COLUMNS,
             "risk_model": risk_model, "risk_column_excluded": "Forecasted_4Hr_Spoilage_Risk_Pct",
             "model_name": best_name,
             "warning_threshold": warning_threshold},
            self.model_dir / "logistics_action_model.joblib",
        )
        with open(self.model_dir / "metadata.json", "w", encoding="utf-8") as file:
            json.dump({
                "version": self.version, "best_model": best_name,
                "training_distribution": distribution.set_index("Class")["Count"].to_dict(),
                "synthetic_training_rows": int((balanced_train["Data_Origin"] != "real").sum()),
                "risk_target_excluded": True, "predicted_risk_feature_used": True,
                "warning_threshold": warning_threshold,
                "test_metrics": test_metrics,
                "emergency_evaluation_note": "No real Emergency holdout records are available.",
            }, file, indent=2)
        print("Classification training distribution:")
        print(distribution.to_string(index=False))
        print(f"Best classifier: {best_name}")
        print(f"Real-test macro F1: {test_metrics['macro_f1_observed_classes']:.4f}")
        print(f"Real-test Warning recall: {test_metrics['warning_recall']:.4f}")
        print("Emergency performance is not measurable until real Emergency labels are available.")
