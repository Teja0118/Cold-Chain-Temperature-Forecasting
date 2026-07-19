"""
===============================================================================
Classification Model Trainer
===============================================================================

Trains multiple classification models and saves the best performing model.

===============================================================================
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class ClassificationModelTrainer:

    TARGET_COLUMN = "Logistics_Action_Recommendation"

    def __init__(
        self,
        version: str = "v1",
        random_state: int = 42,
    ):

        self.version = version

        self.random_state = random_state

        self.train_file = Path(
            "data/final/classification_train_selected.csv"
        )

        self.validation_file = Path(
            "data/final/classification_validation_selected.csv"
        )

        self.model_directory = Path(
            f"models/classification/{self.version}"
        )

        self.model_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

        self.logger = logging.getLogger(__name__)

        self.label_encoder = LabelEncoder()

        self.models = {}

        self.results = []

    ####################################################################
    # Load Dataset
    ####################################################################

    def load_dataset(self):

        self.logger.info(
            "Loading datasets..."
        )

        train_df = pd.read_csv(
            self.train_file
        )

        validation_df = pd.read_csv(
            self.validation_file
        )

        self.X_train = train_df.drop(
            columns=[self.TARGET_COLUMN]
        )

        self.y_train = self.label_encoder.fit_transform(
            train_df[self.TARGET_COLUMN]
        )

        self.X_validation = validation_df.drop(
            columns=[self.TARGET_COLUMN]
        )

        self.y_validation = self.label_encoder.transform(
            validation_df[self.TARGET_COLUMN]
        )

        self.logger.info(
            f"Training Shape : {self.X_train.shape}"
        )

        self.logger.info(
            f"Validation Shape : {self.X_validation.shape}"
        )

    ####################################################################
    # Initialize Models
    ####################################################################

    def initialize_models(self):

        self.logger.info(
            "Initializing models..."
        )

        self.models = {

            "LogisticRegression":
                LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state
                ),

            "DecisionTree":
                DecisionTreeClassifier(
                    random_state=self.random_state
                ),

            "RandomForest":
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=self.random_state,
                    n_jobs=-1
                ),

            "ExtraTrees":
                ExtraTreesClassifier(
                    n_estimators=300,
                    random_state=self.random_state,
                    n_jobs=-1
                )

        }

        ####################################################################
    # Train & Evaluate Models
    ####################################################################

    def train_models(self):

        self.logger.info(
            "Training classification models..."
        )

        for model_name, model in self.models.items():

            self.logger.info(
                f"Training {model_name}..."
            )

            model.fit(
                self.X_train,
                self.y_train
            )

            predictions = model.predict(
                self.X_validation
            )

            accuracy = accuracy_score(
                self.y_validation,
                predictions
            )

            precision = precision_score(
                self.y_validation,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                self.y_validation,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                self.y_validation,
                predictions,
                average="weighted",
                zero_division=0
            )

            self.results.append({

                "Model": model_name,

                "Accuracy": accuracy,

                "Precision": precision,

                "Recall": recall,

                "F1_Score": f1

            })

    ####################################################################
    # Results
    ####################################################################

    def save_results(self):

        results_df = pd.DataFrame(
            self.results
        )

        results_df.sort_values(

            by="F1_Score",

            ascending=False,

            inplace=True

        )

        results_df.to_csv(

            self.model_directory /

            "classification_results.csv",

            index=False

        )

        self.results_df = results_df

        self.logger.info(
            "Model comparison saved."
        )

    ####################################################################
    # Best Model
    ####################################################################

    def select_best_model(self):

        best_model_name = self.results_df.iloc[0]["Model"]

        self.best_model = self.models[
            best_model_name
        ]

        self.best_model_name = best_model_name

        self.logger.info(
            f"Best Model : {best_model_name}"
        )

    ####################################################################
    # Save Model
    ####################################################################

    def save_model(self):

        joblib.dump(

            self.best_model,

            self.model_directory /

            "best_classification_model.pkl"

        )

        joblib.dump(

            self.label_encoder,

            self.model_directory /

            "label_encoder.pkl"

        )

        metadata = {

            "best_model": self.best_model_name,

            "accuracy":
                float(
                    self.results_df.iloc[0]["Accuracy"]
                ),

            "precision":
                float(
                    self.results_df.iloc[0]["Precision"]
                ),

            "recall":
                float(
                    self.results_df.iloc[0]["Recall"]
                ),

            "f1_score":
                float(
                    self.results_df.iloc[0]["F1_Score"]
                )

        }

        with open(

            self.model_directory /

            "model_metadata.json",

            "w"

        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

        self.logger.info(
            "Best model saved."
        )

    ####################################################################
    # Run
    ####################################################################

    def run(self):

        self.load_dataset()

        self.initialize_models()

        self.train_models()

        self.save_results()

        self.select_best_model()

        self.save_model()

        self.logger.info(
            "Classification Model Training Completed."
        )


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    trainer = ClassificationModelTrainer(
        version="v1",
        random_state=42
    )

    trainer.run()