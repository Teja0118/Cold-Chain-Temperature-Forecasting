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
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.training.regression_pipeline import _make_pipeline as make_regression_pipeline


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
    "Temperature_Deviation", "Positive_Temperature_Deviation", "Ambient_Load",
    "Seal_Loss", "Thermal_Leak_Load", "HVAC_Load_Per_Degree",
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
        return df


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

    def _synthetic_rows(self, seeds: pd.DataFrame, count: int, kind: str, rng):
        rows = seeds.sample(n=count, replace=True, random_state=self.random_state).copy().reset_index(drop=True)
        rows["Container_ID"] = [f"SYN_{kind.upper()}_{i:06d}" for i in range(count)]
        set_point = rows["Internal_Set_Point_C"]
        if kind == "warning":
            # Preserve the observed Warning distribution.  Real Warning rows
            # are not uniformly hot; forcing a high-temperature profile made
            # synthetic warnings unlike the real validation/test warnings.
            rows["Ambient_External_Temp_C"] = np.clip(
                rows["Ambient_External_Temp_C"] + rng.normal(0, 1.0, count), 20, 45
            ).round(1)
            rows["Actual_Internal_Temp_C"] = np.clip(
                rows["Actual_Internal_Temp_C"] + rng.normal(0, 0.5, count), -20, 20
            ).round(1)
            rows["HVAC_Power_Consumption_Watts"] = np.clip(
                rows["HVAC_Power_Consumption_Watts"] * rng.uniform(0.90, 1.10, count), 500, 3500
            ).round(1)
            rows["Seal_Integrity_Index"] = np.clip(
                rows["Seal_Integrity_Index"] + rng.normal(0, 0.025, count), 0.25, 1.0
            ).round(3)
            rows[TARGET] = WARNING
        else:
            rows["Ambient_External_Temp_C"] = rng.uniform(38, 45, count).round(1)
            rows["Actual_Internal_Temp_C"] = (set_point + rng.uniform(6, 14, count)).round(1)
            rows["HVAC_Power_Consumption_Watts"] = rng.uniform(1800, 2800, count).round(1)
            rows["Seal_Integrity_Index"] = rng.uniform(0.20, 0.60, count).round(3)
            rows[TARGET] = EMERGENCY
        rows["Data_Origin"] = f"synthetic_{kind}"
        return rows

    def _build_training_data(self, train: pd.DataFrame):
        rng = np.random.default_rng(self.random_state)
        safe = train[train[TARGET] == SAFE]
        warning = train[train[TARGET] == WARNING]
        if len(safe) < 30000 or warning.empty:
            raise ValueError("Need at least 30,000 Safe and one real Warning training row.")

        real_safe = safe.sample(n=30000, random_state=self.random_state).copy()
        real_safe["Data_Origin"] = "real"
        real_warning = warning.copy()
        real_warning["Data_Origin"] = "real"
        synthetic_warning = self._synthetic_rows(
            warning, 7500 - len(real_warning), "warning", rng
        )
        synthetic_emergency = self._synthetic_rows(warning, 7500, "emergency", rng)
        balanced = pd.concat(
            [real_safe, real_warning, synthetic_warning, synthetic_emergency], ignore_index=True
        ).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        expected = {SAFE: 30000, WARNING: 7500, EMERGENCY: 7500}
        actual = balanced[TARGET].value_counts().to_dict()
        if actual != expected:
            raise RuntimeError(f"Unexpected class balance: {actual}")
        return balanced

    @staticmethod
    def _candidates(random_state: int):
        return {
            "LightGBM": LGBMClassifier(
                n_estimators=500, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.9, random_state=random_state,
                n_jobs=-1, verbosity=-1,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2, random_state=random_state, n_jobs=-1,
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=400, min_samples_leaf=2, random_state=random_state, n_jobs=-1,
            ),
        }

    def _calibrate_warning_threshold(self, model, validation: pd.DataFrame):
        """Select a Warning probability threshold using real validation labels."""
        probabilities = model.predict_proba(validation[CLASSIFIER_INPUT_COLUMNS])
        warning_code = int(self.encoder.transform([WARNING])[0])
        y_true = self.encoder.transform(validation[TARGET])
        safe_code = int(self.encoder.transform([SAFE])[0])
        best_threshold, best_score = 0.50, -1.0
        for threshold in np.linspace(0.01, 0.60, 60):
            predicted = np.where(probabilities[:, warning_code] >= threshold, warning_code, safe_code)
            score = f1_score(y_true, predicted, average="macro", zero_division=0)
            if score > best_score:
                best_threshold, best_score = float(threshold), float(score)
        return best_threshold, best_score

    def _predict_actions(self, model, frame: pd.DataFrame, warning_threshold: float):
        probabilities = model.predict_proba(frame[CLASSIFIER_INPUT_COLUMNS])
        safe_code = int(self.encoder.transform([SAFE])[0])
        warning_code = int(self.encoder.transform([WARNING])[0])
        emergency_code = int(self.encoder.transform([EMERGENCY])[0])
        predictions = np.full(len(frame), safe_code, dtype=int)
        predictions[probabilities[:, warning_code] >= warning_threshold] = warning_code
        # Explicit safety override for an Emergency-like physical condition.
        deviation = frame["Actual_Internal_Temp_C"] - frame["Internal_Set_Point_C"]
        emergency_condition = (
            (deviation >= 6)
            & (frame["Seal_Integrity_Index"] < 0.60)
            & (frame["Ambient_External_Temp_C"] >= 38)
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
            model = _make_pipeline(estimator)
            model.fit(X_train, y_train)
            threshold, _ = self._calibrate_warning_threshold(model, validation)
            result = self._evaluate_real_holdout(model, validation, f"validation_{name}", threshold)
            comparison.append({"Model": name, **result})
            fitted[name] = model

        comparison_df = pd.DataFrame(comparison).sort_values(
            ["macro_f1_observed_classes", "warning_recall"], ascending=False
        ).reset_index(drop=True)
        best_name = comparison_df.iloc[0]["Model"]
        # All real training rows and synthetic rows are now used after selection.
        final_model = _make_pipeline(clone(self._candidates(self.random_state)[best_name]))
        final_model.fit(X_train, y_train)
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
