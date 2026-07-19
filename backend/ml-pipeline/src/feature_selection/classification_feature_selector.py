"""
===============================================================================
Feature Selector Module
===============================================================================

Applies the previously learned feature selection to validation and
test datasets.

===============================================================================
"""

import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ClassificationFeatureSelector:

    TARGET_COLUMN = "Logistics_Action_Recommendation"

    def __init__(
        self,
        input_file: str,
        output_file: str,
        version: str = "v1",
    ):

        self.version = version

        self.input_file = Path(input_file)

        self.output_file = Path(output_file)

        self.feature_file = Path(
            "data/final/selected_features.json"
        )

        self.output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

        self.logger = logging.getLogger(__name__)

        self.df = None

        self.selected_features = []

    ####################################################################
    # Load Dataset
    ####################################################################

    def load_dataset(self):

        self.logger.info(
            f"Loading dataset : {self.input_file.name}"
        )

        self.df = pd.read_csv(
            self.input_file
        )

        self.logger.info(
            f"Dataset Shape : {self.df.shape}"
        )

    ####################################################################
    # Encode Categorical Features
    ####################################################################

    def encode_features(self):

        if "Container_ID" in self.df.columns:

            self.df.drop(
                columns=["Container_ID"],
                inplace=True
            )

        if "Cargo_Type" in self.df.columns:

            encoder = LabelEncoder()

            self.df["Cargo_Type"] = encoder.fit_transform(
                self.df["Cargo_Type"]
            )

    ####################################################################
    # Load Selected Features
    ####################################################################

    def load_selected_features(self):

        self.logger.info(
            "Loading selected feature list..."
        )

        with open(
            self.feature_file,
            "r"
        ) as file:

            report = json.load(file)

        self.selected_features = report[
            "selected_features"
        ]

        self.logger.info(
            f"Loaded {len(self.selected_features)} selected features."
        )

    ####################################################################
    # Apply Feature Selection
    ####################################################################

    def apply_selection(self):

        self.logger.info(
            "Applying selected features..."
        )

        missing = [

            feature

            for feature in self.selected_features

            if feature not in self.df.columns

        ]

        if missing:

            raise ValueError(
                f"Missing features : {missing}"
            )

        self.df = self.df[

            self.selected_features +

            [self.TARGET_COLUMN]

        ]

    ####################################################################
    # Save Dataset
    ####################################################################

    def save_dataset(self):

        self.df.to_csv(

            self.output_file,

            index=False

        )

        self.logger.info(
            f"Saved : {self.output_file}"
        )

    ####################################################################
    # Run Pipeline
    ####################################################################

    def run(self):

        self.load_dataset()

        self.encode_features()

        self.load_selected_features()

        self.apply_selection()

        self.save_dataset()

        self.logger.info(
            "Feature Selection Applied Successfully."
        )


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    selector = ClassificationFeatureSelector(

        input_file="data/engineered/classification_validation_engineered.csv",

        output_file="data/final/classification_validation_selected.csv",

        version="v1"

    )

    selector.run()