"""Leakage-safe training pipeline for spoilage-risk regression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


TARGET_COLUMN = "Forecasted_4Hr_Spoilage_Risk_Pct"
ID_COLUMN = "Container_ID"
CATEGORICAL_COLUMNS = ["Cargo_Type"]
BASE_NUMERIC_COLUMNS = [
    "Ambient_External_Temp_C",
    "Internal_Set_Point_C",
    "Actual_Internal_Temp_C",
    "HVAC_Power_Consumption_Watts",
    "Seal_Integrity_Index",
]
ENGINEERED_NUMERIC_COLUMNS = [
    "Temperature_Deviation_C",
    "Positive_Temperature_Deviation_C",
    "Cooling_Load_C",
    "HVAC_Efficiency",
    "Seal_Loss_Score",
    "HVAC_Stress_Index",
    "Thermal_Leak_Load",
    "HVAC_Load_Per_Degree",
]
MODEL_INPUT_COLUMNS = CATEGORICAL_COLUMNS + BASE_NUMERIC_COLUMNS
RISK_BINS = [-np.inf, 10, 20, 30, 40, 50, 60, 70, 80, 90, np.inf]


class ColdChainFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build physical features before scaling, preserving their real units."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X.copy()
        missing = set(MODEL_INPUT_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required input columns: {sorted(missing)}")

        frame = frame[MODEL_INPUT_COLUMNS].copy()
        frame["Temperature_Deviation_C"] = (
            frame["Actual_Internal_Temp_C"] - frame["Internal_Set_Point_C"]
        )
        frame["Positive_Temperature_Deviation_C"] = frame[
            "Temperature_Deviation_C"
        ].clip(lower=0)
        frame["Cooling_Load_C"] = (
            frame["Ambient_External_Temp_C"] - frame["Internal_Set_Point_C"]
        )
        power = frame["HVAC_Power_Consumption_Watts"].replace(0, np.nan)
        frame["HVAC_Efficiency"] = frame["Cooling_Load_C"] / power
        frame["HVAC_Efficiency"] = frame["HVAC_Efficiency"].replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
        frame["Seal_Loss_Score"] = 1 - frame["Seal_Integrity_Index"]
        frame["HVAC_Stress_Index"] = (
            frame["HVAC_Power_Consumption_Watts"] * frame["Seal_Loss_Score"]
        )
        frame["Thermal_Leak_Load"] = frame["Cooling_Load_C"].clip(lower=0) * frame[
            "Seal_Loss_Score"
        ]
        frame["HVAC_Load_Per_Degree"] = frame[
            "HVAC_Power_Consumption_Watts"
        ] / (frame["Cooling_Load_C"].abs() + 1)
        return frame


def _risk_strata(y: pd.Series) -> pd.Series:
    """Create populated quantile bins for a robust regression split.

    Fixed clinical-style bands are retained for reporting, but the raw data has
    empty 50--100% bands. Quantile bins avoid creating an impossible stratum
    while still preserving the target distribution in every split.
    """
    strata = pd.qcut(y, q=10, duplicates="drop")
    counts = strata.value_counts()
    if counts.min() < 2:
        raise ValueError("Not enough target variation to create a stratified split.")
    return strata.astype(str)


def _metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    predictions = np.clip(predictions, 0, 100)
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "R2": float(r2_score(y_true, predictions)),
    }


def _metrics_by_risk_band(y_true: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    report = pd.DataFrame({"actual": y_true.to_numpy(), "prediction": predictions})
    report["risk_band"] = pd.cut(report["actual"], bins=RISK_BINS, include_lowest=True)
    rows = []
    for band, group in report.groupby("risk_band", observed=True):
        rows.append(
            {
                "Risk_Band": str(band),
                "Samples": len(group),
                "MAE": mean_absolute_error(group["actual"], group["prediction"]),
                "RMSE": np.sqrt(mean_squared_error(group["actual"], group["prediction"])),
            }
        )
    return pd.DataFrame(rows)


def _make_preprocessor() -> ColumnTransformer:
    numeric_columns = BASE_NUMERIC_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def _make_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_builder", ColdChainFeatureBuilder()),
            ("preprocessor", _make_preprocessor()),
            ("model", model),
        ]
    )


def _candidate_models(random_state: int) -> dict[str, object]:
    return {
        "LightGBM": LGBMRegressor(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


@dataclass
class RegressionTrainingResult:
    model_name: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    interval_half_width: float


class RegressionTrainingPipeline:
    """Owns the full regression workflow from raw CSV to versioned artifact."""

    def __init__(
        self,
        data_path: str,
        version: str = "v1",
        random_state: int = 42,
        augment_domain_scenarios: bool = False,
    ):
        self.data_path = Path(data_path)
        self.version = version
        self.random_state = random_state
        self.augment_domain_scenarios = augment_domain_scenarios
        self.model_dir = Path("models/regression") / version
        self.report_dir = Path("reports/regression") / version

    def _load_raw_data(self) -> tuple[pd.DataFrame, pd.Series]:
        df = pd.read_csv(self.data_path)
        required = set(MODEL_INPUT_COLUMNS + [TARGET_COLUMN, ID_COLUMN])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
        if df[TARGET_COLUMN].isna().any():
            raise ValueError(f"{TARGET_COLUMN} contains missing values.")
        # The regression model deliberately uses only real source records.
        # Action is retained only to select real Warning seed rows for optional
        # training augmentation. The feature builder never exposes it to a model.
        return df[MODEL_INPUT_COLUMNS + ["Logistics_Action_Recommendation"]], df[TARGET_COLUMN]

    def _split(self, X: pd.DataFrame, y: pd.Series):
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=self.random_state,
            stratify=_risk_strata(y),
        )
        validation_fraction = 0.15 / 0.85
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_val,
            y_train_val,
            test_size=validation_fraction,
            random_state=self.random_state,
            stratify=_risk_strata(y_train_val),
        )
        return X_train, X_validation, X_test, y_train, y_validation, y_test

    def _augment_training_data(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Create documented, physically constrained high-risk scenarios.

        Synthetic rows are a coverage aid, not real observations. They are used
        only in training and are never mixed into validation or test data.
        """
        if not self.augment_domain_scenarios:
            return X_train, y_train, {"enabled": False, "warning_rows": 0, "emergency_rows": 0}

        rng = np.random.default_rng(self.random_state)
        warning_seeds = X_train[
            X_train["Logistics_Action_Recommendation"]
            == "WARNING_Request_HVAC_Remote_Reset"
        ]
        if warning_seeds.empty:
            raise ValueError("Domain augmentation needs at least one real Warning training record.")

        def scenario_rows(count: int, scenario: str):
            rows = warning_seeds.sample(n=count, replace=True, random_state=self.random_state).copy()
            if scenario == "warning":
                rows["Ambient_External_Temp_C"] = rng.uniform(35, 40, count).round(1)
                rows["Actual_Internal_Temp_C"] = (
                    rows["Internal_Set_Point_C"] + rng.uniform(2, 5, count)
                ).round(1)
                rows["HVAC_Power_Consumption_Watts"] = rng.uniform(1400, 1800, count).round(1)
                rows["Seal_Integrity_Index"] = rng.uniform(0.60, 0.80, count).round(3)
                target = rng.uniform(45, 75, count).round(1)
            else:
                rows["Ambient_External_Temp_C"] = rng.uniform(40, 45, count).round(1)
                rows["Actual_Internal_Temp_C"] = (
                    rows["Internal_Set_Point_C"] + rng.uniform(6, 12, count)
                ).round(1)
                rows["HVAC_Power_Consumption_Watts"] = rng.uniform(1800, 2500, count).round(1)
                rows["Seal_Integrity_Index"] = rng.uniform(0.25, 0.60, count).round(3)
                target = rng.uniform(80, 100, count).round(1)
            return rows, pd.Series(target, name=TARGET_COLUMN)

        warning_needed = max(0, 5000 - len(warning_seeds))
        synthetic_warning, warning_target = scenario_rows(warning_needed, "warning")
        synthetic_emergency, emergency_target = scenario_rows(5000, "emergency")
        X_augmented = pd.concat(
            [X_train, synthetic_warning, synthetic_emergency], ignore_index=True
        )
        y_augmented = pd.concat(
            [y_train.reset_index(drop=True), warning_target, emergency_target], ignore_index=True
        )
        return X_augmented, y_augmented, {
            "enabled": True,
            "warning_rows": warning_needed,
            "emergency_rows": 5000,
        }

    def run(self) -> RegressionTrainingResult:
        X, y = self._load_raw_data()
        X_train, X_validation, X_test, y_train, y_validation, y_test = self._split(X, y)
        X_train, y_train, augmentation = self._augment_training_data(X_train, y_train)
        print(f"Regression source records: {len(X)} (raw only)")
        print(f"Split sizes — train: {len(X_train)}, validation: {len(X_validation)}, test: {len(X_test)}")
        if augmentation["enabled"]:
            print(
                "Training-only domain scenarios added — "
                f"Warning: {augmentation['warning_rows']}, Emergency: {augmentation['emergency_rows']}"
            )

        validation_rows = []
        fitted_candidates = {}
        for name, estimator in _candidate_models(self.random_state).items():
            pipeline = _make_pipeline(estimator)
            pipeline.fit(X_train, y_train)
            predictions = np.clip(pipeline.predict(X_validation), 0, 100)
            metrics = _metrics(y_validation, predictions)
            validation_rows.append({"Model": name, **metrics})
            fitted_candidates[name] = pipeline
            print(f"{name} validation RMSE: {metrics['RMSE']:.4f}")

        validation_df = pd.DataFrame(validation_rows).sort_values("RMSE").reset_index(drop=True)
        best_name = validation_df.iloc[0]["Model"]
        validation_predictions = np.clip(
            fitted_candidates[best_name].predict(X_validation), 0, 100
        )
        interval_half_width = float(
            np.quantile(np.abs(y_validation.to_numpy() - validation_predictions), 0.90)
        )

        # Refit only after model selection; the test set remains untouched.
        best_estimator = clone(_candidate_models(self.random_state)[best_name])
        production_pipeline = _make_pipeline(best_estimator)
        production_pipeline.fit(pd.concat([X_train, X_validation]), pd.concat([y_train, y_validation]))
        test_predictions = np.clip(production_pipeline.predict(X_test), 0, 100)
        test_metrics = _metrics(y_test, test_predictions)

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        validation_df.to_csv(self.report_dir / "validation_model_comparison.csv", index=False)
        _metrics_by_risk_band(y_test, test_predictions).to_csv(
            self.report_dir / "test_metrics_by_risk_band.csv", index=False
        )
        prediction_report = pd.DataFrame(
            {
                "Actual_Risk_Pct": y_test.to_numpy(),
                "Predicted_Risk_Pct": test_predictions,
            }
        )
        prediction_report["Absolute_Error"] = (
            prediction_report["Actual_Risk_Pct"]
            - prediction_report["Predicted_Risk_Pct"]
        ).abs()
        prediction_report["Lower_90pct_Interval"] = np.clip(
            prediction_report["Predicted_Risk_Pct"] - interval_half_width, 0, 100
        )
        prediction_report["Upper_90pct_Interval"] = np.clip(
            prediction_report["Predicted_Risk_Pct"] + interval_half_width, 0, 100
        )
        prediction_report.to_csv(self.report_dir / "test_predictions.csv", index=False)
        pd.DataFrame([test_metrics]).assign(Model=best_name).to_csv(
            self.report_dir / "test_metrics.csv", index=False
        )
        joblib.dump(
            {
                "pipeline": production_pipeline,
                "target": TARGET_COLUMN,
                "input_columns": MODEL_INPUT_COLUMNS,
                "prediction_interval_90_half_width": interval_half_width,
                "model_name": best_name,
                "dataset_path": str(self.data_path),
                "dataset_rows": len(X),
            },
            self.model_dir / "spoilage_risk_model.joblib",
        )
        with open(self.model_dir / "metadata.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "version": self.version,
                    "model": best_name,
                    "validation_metrics": validation_df.iloc[0].drop("Model").to_dict(),
                    "test_metrics": test_metrics,
                    "prediction_interval_90_half_width": interval_half_width,
                    "dataset_path": str(self.data_path),
                    "dataset_rows": len(X),
                "synthetic_data_used": augmentation["enabled"],
                "training_augmentation": augmentation,
                },
                file,
                indent=2,
            )

        print(f"Best regression model: {best_name}")
        print(f"Test RMSE: {test_metrics['RMSE']:.4f}, Test MAE: {test_metrics['MAE']:.4f}, Test R2: {test_metrics['R2']:.4f}")
        print(f"90% prediction interval: prediction ± {interval_half_width:.2f} risk points")
        return RegressionTrainingResult(
            model_name=best_name,
            validation_metrics=validation_df.iloc[0].drop("Model").to_dict(),
            test_metrics=test_metrics,
            interval_half_width=interval_half_width,
        )
