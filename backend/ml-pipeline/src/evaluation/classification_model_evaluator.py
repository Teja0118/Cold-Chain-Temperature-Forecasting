"""
===============================================================================
Classification Model Evaluator
===============================================================================

Evaluates the final tuned classification model on the test dataset.

===============================================================================
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class ClassificationModelEvaluator:

    TARGET_COLUMN = "Logistics_Action_Recommendation"

    def __init__(
        self,
        version: str = "v1",
    ):

        self.version = version

        self.model_path = Path(
            f"models/classification/{self.version}/best_tuned_model.pkl"
        )

        self.encoder_path = Path(
            f"models/classification/{self.version}/label_encoder.pkl"
        )

        self.test_file = Path(
            "data/final/classification_test_selected.csv"
        )

        self.output_directory = Path(
            f"reports/classification/{self.version}/evaluation"
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

        self.logger = logging.getLogger(__name__)

    ####################################################################
    # Load Resources
    ####################################################################

    def load_resources(self):

        self.logger.info(
            "Loading model and test dataset..."
        )

        self.model = joblib.load(
            self.model_path
        )

        self.encoder = joblib.load(
            self.encoder_path
        )

        self.test_df = pd.read_csv(
            self.test_file
        )

        self.X_test = self.test_df.drop(
            columns=[self.TARGET_COLUMN]
        )

        self.y_test = self.encoder.transform(
            self.test_df[self.TARGET_COLUMN]
        )

    ####################################################################
    # Predict
    ####################################################################

    def predict(self):

        self.logger.info(
            "Running predictions..."
        )

        self.predictions = self.model.predict(
            self.X_test
        )

    ####################################################################
    # Evaluate
    ####################################################################

    def evaluate(self):

        self.logger.info(
            "Evaluating model..."
        )

        self.metrics = {

            "Accuracy": accuracy_score(
                self.y_test,
                self.predictions
            ),

            "Precision": precision_score(
                self.y_test,
                self.predictions,
                average="weighted",
                zero_division=0
            ),

            "Recall": recall_score(
                self.y_test,
                self.predictions,
                average="weighted",
                zero_division=0
            ),

            "F1_Score": f1_score(
                self.y_test,
                self.predictions,
                average="weighted",
                zero_division=0
            )

        }

        self.confusion = confusion_matrix(
            self.y_test,
            self.predictions
        )

        self.report = classification_report(
            self.y_test,
            self.predictions,
            target_names=self.encoder.classes_,
            output_dict=True,
            zero_division=0
        )

        ####################################################################
    # Save Metrics
    ####################################################################

    def save_metrics(self):

        self.logger.info(
            "Saving evaluation metrics..."
        )

        with open(

            self.output_directory /

            "evaluation_metrics.json",

            "w"

        ) as file:

            json.dump(

                self.metrics,

                file,

                indent=4

            )

    ####################################################################
    # Save Classification Report
    ####################################################################

    def save_classification_report(self):

        self.logger.info(
            "Saving classification report..."
        )

        report_df = pd.DataFrame(
            self.report
        ).transpose()

        report_df.to_csv(

            self.output_directory /

            "classification_report.csv"

        )

    ####################################################################
    # Save Confusion Matrix
    ####################################################################

    def save_confusion_matrix(self):

        self.logger.info(
            "Saving confusion matrix..."
        )

        confusion_df = pd.DataFrame(

            self.confusion,

            index=self.encoder.classes_,

            columns=self.encoder.classes_

        )

        confusion_df.to_csv(

            self.output_directory /

            "confusion_matrix.csv"

        )

    ####################################################################
    # Save Predictions
    ####################################################################

    def save_predictions(self):

        self.logger.info(
            "Saving predictions..."
        )

        prediction_df = self.test_df.copy()

        prediction_df["Predicted_Class"] = (

            self.encoder.inverse_transform(

                self.predictions

            )

        )

        prediction_df.to_csv(

            self.output_directory /

            "test_predictions.csv",

            index=False

        )

    ####################################################################
    # Save Confusion Matrix Heatmap
    ####################################################################

    def save_confusion_heatmap(self):

        import matplotlib.pyplot as plt

        import seaborn as sns

        plt.figure(figsize=(8, 6))

        sns.heatmap(

            self.confusion,

            annot=True,

            fmt="d",

            cmap="Blues",

            xticklabels=self.encoder.classes_,

            yticklabels=self.encoder.classes_

        )

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.savefig(

            self.output_directory /

            "confusion_matrix.png",

            dpi=300

        )

        plt.close()

    ####################################################################
    # Run
    ####################################################################

    def run(self):

        self.load_resources()

        self.predict()

        self.evaluate()

        self.save_metrics()

        self.save_classification_report()

        self.save_confusion_matrix()

        self.save_confusion_heatmap()

        self.save_predictions()

        self.logger.info(
            "Model Evaluation Completed."
        )


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    evaluator = ClassificationModelEvaluator(
        version="v1"
    )

    evaluator.run()