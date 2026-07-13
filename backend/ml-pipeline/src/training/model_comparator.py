"""
Model Comparison Utility

Maintains the performance metrics of all regression models.
After every model training, the metrics are appended to the
comparison table.

At the end of training, the table helps identify the best model.
"""

import os
import pandas as pd


class ModelComparator:

    OUTPUT_FILE = "outputs/regression_model_comparison.csv"

    @staticmethod
    def save_metrics(model_name, metrics):
        """
        Save evaluation metrics of a model.
        """

        os.makedirs("outputs", exist_ok=True)

        row = {

            "Model": model_name,

            "MAE": metrics["MAE"],

            "MSE": metrics["MSE"],

            "RMSE": metrics["RMSE"],

            "R2": metrics["R2"]

        }

        # First model
        if not os.path.exists(ModelComparator.OUTPUT_FILE):

            df = pd.DataFrame([row])

        else:

            df = pd.read_csv(ModelComparator.OUTPUT_FILE)

            # Remove existing row if model already exists
            df = df[df["Model"] != model_name]

            df = pd.concat(
                [df, pd.DataFrame([row])],
                ignore_index=True
            )

        # Highest R² first
        df = df.sort_values(
            by="R2",
            ascending=False
        )

        df.to_csv(
            ModelComparator.OUTPUT_FILE,
            index=False
        )

        print("\nRegression Model Comparison\n")

        print(df)

    @staticmethod
    def best_model():

        df = pd.read_csv(
            ModelComparator.OUTPUT_FILE
        )

        best = df.iloc[0]

        print("\nBest Regression Model")

        print("-------------------------")

        print(f"Model : {best['Model']}")

        print(f"R²    : {best['R2']:.4f}")

        print(f"RMSE  : {best['RMSE']:.4f}")