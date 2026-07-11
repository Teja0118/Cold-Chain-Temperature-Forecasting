import os

from .encoder import Encoder
from .scaler import Scaler


class DataPreprocessor:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

    def drop_columns(self):

        self.df.drop(
            columns=["Container_ID"],
            inplace=True
        )

        print("Container_ID dropped.")

    def encode_features(self):

        Encoder.encode_column(
            self.df,
            "Cargo_Type",
            "data/artifacts/cargo_type_encoder.pkl"
        )

        Encoder.encode_column(
            self.df,
            "Logistics_Action_Recommendation",
            "data/artifacts/action_encoder.pkl"
        )

    def split_features_targets(self):

        X = self.df.drop(
            columns=[
                "Forecasted_4Hr_Spoilage_Risk_Pct",
                "Logistics_Action_Recommendation"
            ]
        )

        y_reg = self.df[
            "Forecasted_4Hr_Spoilage_Risk_Pct"
        ]

        y_cls = self.df[
            "Logistics_Action_Recommendation"
        ]

        return X, y_reg, y_cls

    def scale_features(self, X):

        numerical_columns = [

            "Ambient_External_Temp_C",

            "Internal_Set_Point_C",

            "Actual_Internal_Temp_C",

            "HVAC_Power_Consumption_Watts",

            "Seal_Integrity_Index"

        ]

        X = Scaler.scale_features(

            X,

            numerical_columns,

            "data/artifacts/feature_scaler.pkl"

        )

        return X

    def save_dataset(self, X, y_reg, y_cls):

        X.to_csv(
            "data/processed/X_features.csv",
            index=False
        )

        y_reg.to_csv(
            "data/processed/y_regression.csv",
            index=False
        )

        y_cls.to_csv(
            "data/processed/y_classification.csv",
            index=False
        )

        print("\nProcessed datasets saved.")

    def preprocess(self):

        os.makedirs(
            "data/artifacts",
            exist_ok=True
        )

        os.makedirs(
            "data/processed",
            exist_ok=True
        )

        self.drop_columns()

        self.encode_features()

        X, y_reg, y_cls = self.split_features_targets()

        X = self.scale_features(X)

        self.save_dataset(
            X,
            y_reg,
            y_cls
        )

        return X, y_reg, y_cls