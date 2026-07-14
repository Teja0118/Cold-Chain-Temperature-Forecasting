"""
Model Saving Utility

Saves trained models to disk
"""

import os
import joblib

class ModelSaver:

    @staticmethod
    def save_model(model, filename):
        os.makedirs("models", exist_ok=True)
        path = f"models/{filename}"
        joblib.dump(model,path)
        print(f"Model Saved: {path}")

    """
    Save the final production model along with
    the feature names used during training.
    """

    @staticmethod
    def save_production_model(

        model,

        feature_columns

    ):

        import joblib

        import os

        os.makedirs(

            "models",

            exist_ok=True

        )

        joblib.dump(

            model,

            "models/final_regression_model.pkl"

        )

        joblib.dump(

            feature_columns,

            "models/feature_columns.pkl"

        )

        print("\nProduction model saved.")

        print("models/final_regression_model.pkl")

        print("models/feature_columns.pkl")

        