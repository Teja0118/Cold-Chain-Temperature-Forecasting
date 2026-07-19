"""
===============================================================================
Classification Hyperparameter Tuner
===============================================================================

Performs hyperparameter tuning for classification models and saves the
best tuned model.

===============================================================================
"""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from sklearn.model_selection import (
    RandomizedSearchCV,
)

from sklearn.preprocessing import LabelEncoder


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

class ClassificationHyperparameterTuner:

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

        self.output_directory = Path(
            f"models/classification/{self.version}"
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

        self.encoder = LabelEncoder()

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

        self.y_train = self.encoder.fit_transform(
            train_df[self.TARGET_COLUMN]
        )

        self.X_validation = validation_df.drop(
            columns=[self.TARGET_COLUMN]
        )

        self.y_validation = self.encoder.transform(
            validation_df[self.TARGET_COLUMN]
        )

    ####################################################################
    # Initialize Models
    ####################################################################

    def initialize_models(self):

        self.models = {

            "RandomForest": {

                "model": RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=-1
                ),

                "params": {

                    "n_estimators":
                        [200, 300, 500],

                    "max_depth":
                        [10, 20, 30, None],

                    "min_samples_split":
                        [2, 5, 10],

                    "min_samples_leaf":
                        [1, 2, 4],

                    "max_features":
                        ["sqrt", "log2"]

                }

            },

            "ExtraTrees": {

                "model": ExtraTreesClassifier(
                    random_state=self.random_state,
                    n_jobs=-1
                ),

                "params": {

                    "n_estimators":
                        [200, 300, 500],

                    "max_depth":
                        [10, 20, 30, None],

                    "min_samples_split":
                        [2, 5, 10],

                    "min_samples_leaf":
                        [1, 2, 4],

                    "max_features":
                        ["sqrt", "log2"]

                }

            }

        }

        ####################################################################
    # Hyperparameter Tuning
    ####################################################################

    def tune_models(self):

        self.logger.info(
            "Starting hyperparameter tuning..."
        )

        self.results = []

        self.best_estimators = {}

        for model_name, config in self.models.items():

            self.logger.info(
                f"Tuning {model_name}..."
            )

            search = RandomizedSearchCV(

                estimator=config["model"],

                param_distributions=config["params"],

                n_iter=20,

                cv=5,

                scoring="f1_weighted",

                random_state=self.random_state,

                n_jobs=-1,

                verbose=1

            )

            search.fit(

                self.X_train,

                self.y_train

            )

            best_model = search.best_estimator_

            predictions = best_model.predict(
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

                "F1_Score": f1,

                "Best_Params": search.best_params_

            })

            self.best_estimators[
                model_name
            ] = best_model

    ####################################################################
    # Select Best Model
    ####################################################################

    def select_best_model(self):

        self.results_df = pd.DataFrame(
            self.results
        )

        self.results_df.sort_values(

            by="F1_Score",

            ascending=False,

            inplace=True

        )

        self.best_model_name = (
            self.results_df.iloc[0]["Model"]
        )

        self.best_model = (
            self.best_estimators[
                self.best_model_name
            ]
        )

        self.logger.info(
            f"Best Tuned Model : {self.best_model_name}"
        )

    ####################################################################
    # Save Results
    ####################################################################

    def save_results(self):

        self.results_df.to_csv(

            self.output_directory /

            "hyperparameter_results.csv",

            index=False

        )

        joblib.dump(

            self.best_model,

            self.output_directory /

            "best_tuned_model.pkl"

        )

        joblib.dump(

            self.encoder,

            self.output_directory /

            "label_encoder.pkl"

        )

        metadata = {

            "best_model":
                self.best_model_name,

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
                ),

            "best_parameters":
                self.results_df.iloc[0][
                    "Best_Params"
                ]

        }

        with open(

            self.output_directory /

            "best_model_metadata.json",

            "w"

        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
                default=str
            )

        self.logger.info(
            "Tuned model saved."
        )

    ####################################################################
    # Run
    ####################################################################

    def run(self):

        self.load_dataset()

        self.initialize_models()

        self.tune_models()

        self.select_best_model()

        self.save_results()

        self.logger.info(
            "Hyperparameter Tuning Completed."
        )


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    tuner = ClassificationHyperparameterTuner(
        version="v1",
        random_state=42
    )

    tuner.run()

    