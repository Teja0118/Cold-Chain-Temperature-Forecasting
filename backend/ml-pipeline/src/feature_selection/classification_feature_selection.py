"""
===============================================================================
Feature Selection Module
===============================================================================

Selects the most informative features for classification using multiple
feature selection techniques.

===============================================================================
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor


class ClassificationFeatureSelection:

    TARGET_COLUMN = "Logistics_Action_Recommendation"

    CORRELATION_THRESHOLD = 0.90

    def __init__(
        self,
        version: str = "v1",
        random_state: int = 42,
    ):

        self.version = version

        self.random_state = random_state

        self.input_file = Path(
            "data/balanced/classification_balanced_dataset.csv"
        )

        self.output_directory = Path(
            "data/final"
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

        self.df = None

        self.features = None

        self.target = None

        self.selected_features = []

    ####################################################################
    # Load Dataset
    ####################################################################

    def load_dataset(self):

        self.logger.info(
            "Loading engineered dataset..."
        )

        self.df = pd.read_csv(
            self.input_file
        )

        self.features = self.df.drop(
            columns=[
                self.TARGET_COLUMN,
                "Container_ID"
            ]
        )

        encoder = LabelEncoder()

        self.features["Cargo_Type"] = encoder.fit_transform(
            self.features["Cargo_Type"]
        )

        self.target = self.df[
            self.TARGET_COLUMN
        ]

        self.logger.info(
            f"Dataset Shape : {self.df.shape}"
        )

    ####################################################################
    # Encode Target
    ####################################################################

    def encode_target(self):

        encoder = LabelEncoder()

        self.target = encoder.fit_transform(
            self.target
        )

    ####################################################################
    # Mutual Information
    ####################################################################

    def mutual_information(self):

        self.logger.info(
            "Computing Mutual Information..."
        )

        scores = mutual_info_classif(
            self.features,
            self.target,
            random_state=self.random_state
        )

        mi = pd.DataFrame({

            "Feature": self.features.columns,

            "Mutual_Information": scores

        })

        mi.sort_values(
            by="Mutual_Information",
            ascending=False,
            inplace=True
        )

        mi.to_csv(

            self.output_directory /

            "mutual_information.csv",

            index=False

        )

        self.mi_scores = mi

    ####################################################################
    # Random Forest Importance
    ####################################################################

    def random_forest_importance(self):

        self.logger.info(
            "Computing Random Forest Importance..."
        )

        model = RandomForestClassifier(

            n_estimators=300,

            random_state=self.random_state,

            n_jobs=-1

        )

        model.fit(

            self.features,

            self.target

        )

        importance = pd.DataFrame({

            "Feature": self.features.columns,

            "Importance": model.feature_importances_

        })

        importance.sort_values(

            by="Importance",

            ascending=False,

            inplace=True

        )

        importance.to_csv(

            self.output_directory /

            "random_forest_importance.csv",

            index=False

        )

        self.rf_scores = importance

        ####################################################################
    # Correlation Filter
    ####################################################################

    def correlation_filter(self):

        self.logger.info(
            "Applying correlation filter..."
        )

        correlation = self.features.corr().abs()

        upper_triangle = correlation.where(
            np.triu(
                np.ones(correlation.shape),
                k=1
            ).astype(bool)
        )

        dropped_features = [

            column

            for column in upper_triangle.columns

            if any(
                upper_triangle[column] >
                self.CORRELATION_THRESHOLD
            )

        ]

        self.filtered_features = self.features.drop(
            columns=dropped_features,
            errors="ignore"
        )

        pd.DataFrame({

            "Dropped_Features": dropped_features

        }).to_csv(

            self.output_directory /
            "correlation_filtered_features.csv",

            index=False

        )

        self.logger.info(
            f"Removed {len(dropped_features)} highly correlated features."
        )

    ####################################################################
    # VIF Analysis
    ####################################################################

    def vif_analysis(self):

        self.logger.info(
            "Computing VIF..."
        )

        vif = pd.DataFrame()

        vif["Feature"] = self.filtered_features.columns

        vif["VIF"] = [

            variance_inflation_factor(
                self.filtered_features.values,
                i
            )

            for i in range(
                self.filtered_features.shape[1]
            )

        ]

        vif.sort_values(
            by="VIF",
            ascending=False,
            inplace=True
        )

        vif.to_csv(

            self.output_directory /

            "vif_scores.csv",

            index=False

        )

    ####################################################################
    # Final Feature Selection
    ####################################################################

    def select_features(self):

        self.logger.info(
            "Selecting final features..."
        )

        mi_features = set(

            self.mi_scores.head(15)["Feature"]

        )

        rf_features = set(

            self.rf_scores.head(15)["Feature"]

        )

        available_features = set(
            self.filtered_features.columns
        )

        self.selected_features = sorted(

            list(

                (mi_features & rf_features)

                &

                available_features

            )

        )

        if not self.selected_features:

            raise ValueError(
                "No features selected."
            )

        self.logger.info(
            f"Selected {len(self.selected_features)} features."
        )

    ####################################################################
    # Save Selected Dataset
    ####################################################################

    def save_dataset(self):

        selected_df = self.df[

            self.selected_features +

            [self.TARGET_COLUMN]

        ]

        output_file = (

            self.output_directory /

            "classification_train_selected.csv"

        )

        selected_df.to_csv(

            output_file,

            index=False

        )

        self.logger.info(
            f"Saved : {output_file}"
        )

    ####################################################################
    # Save Report
    ####################################################################

    def save_report(self):

        report = {

            "selected_feature_count":
                len(self.selected_features),

            "selected_features":
                self.selected_features

        }

        with open(

            self.output_directory /

            "selected_features.json",

            "w"

        ) as file:

            json.dump(
                report,
                file,
                indent=4
            )

    ####################################################################
    # Run Pipeline
    ####################################################################

    def run(self):

        self.load_dataset()

        self.encode_target()

        self.mutual_information()

        self.random_forest_importance()

        self.correlation_filter()

        self.vif_analysis()

        self.select_features()

        self.save_dataset()

        self.save_report()

        self.logger.info(
            "Feature Selection Completed."
        )


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    pipeline = ClassificationFeatureSelection(
        version="v1",
        random_state=42
    )

    pipeline.run()
    