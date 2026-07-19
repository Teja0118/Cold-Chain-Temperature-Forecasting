"""
===============================================================================
Classification Data Preparation Module
===============================================================================

Purpose
-------
This module prepares the raw dataset for the classification pipeline.

Responsibilities
----------------
1. Load raw dataset
2. Validate dataset integrity
3. Generate dataset summary
4. Perform stratified Train/Validation/Test split
5. Save processed datasets
6. Generate metadata report

Project: Cold Chain Temperature Forecasting
===============================================================================
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class ClassificationDataPreparation:
    """
    Handles data preparation for the classification pipeline.
    """


    TARGET_COLUMN = "Logistics_Action_Recommendation"

    REQUIRED_COLUMNS = [
        "Container_ID",
        "Cargo_Type",
        "Ambient_External_Temp_C",
        "Internal_Set_Point_C",
        "Actual_Internal_Temp_C",
        "HVAC_Power_Consumption_Watts",
        "Seal_Integrity_Index",
        "Forecasted_4Hr_Spoilage_Risk_Pct",
        "Logistics_Action_Recommendation",
    ]

    def __init__(
        self,
        data_path: str,
        version: str = "v1",
        random_state: int = 42,
    ):

        self.data_path = Path(data_path)

        self.version = version

        self.random_state = random_state

        self.output_directory = Path("data/processed")

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.logger = self._configure_logger()

        self.dataframe = None

        self.train_df = None
        self.validation_df = None
        self.test_df = None

    ###########################################################################
    # Logger
    ###########################################################################

    def _configure_logger(self):

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

        return logging.getLogger(__name__)

    ###########################################################################
    # Load Dataset
    ###########################################################################

    def load_dataset(self):

        self.logger.info("=" * 70)
        self.logger.info("Loading Classification Dataset...")
        self.logger.info("=" * 70)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.input_file}"
            )

        self.dataframe = pd.read_csv(self.data_path)

        self.logger.info(
            f"Dataset Loaded Successfully : {self.data_path.name}"
        )

        self.logger.info(
            f"Dataset Shape : {self.dataframe.shape}"
        )

    ###########################################################################
    # Validate Dataset
    ###########################################################################

    def validate_dataset(self):

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Validating Dataset...")
        self.logger.info("=" * 70)

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in self.dataframe.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Missing Required Columns:\n{missing_columns}"
            )

        self.logger.info("Required columns verified.")

        duplicate_rows = self.dataframe.duplicated().sum()

        self.logger.info(
            f"Duplicate Rows : {duplicate_rows}"
        )

        missing_values = self.dataframe.isnull().sum()

        total_missing = missing_values.sum()

        self.logger.info(
            f"Total Missing Values : {total_missing}"
        )

        if total_missing > 0:

            self.logger.warning("\nMissing Values Per Column:")

            for column, value in missing_values.items():

                if value > 0:

                    self.logger.warning(
                        f"{column:<40} {value}"
                    )

        self.logger.info("Dataset validation completed.")

    ###########################################################################
    # Dataset Summary
    ###########################################################################

    def dataset_summary(self):

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Dataset Summary")
        self.logger.info("=" * 70)

        self.logger.info(
            f"Rows    : {len(self.dataframe):,}"
        )

        self.logger.info(
            f"Columns : {len(self.dataframe.columns)}"
        )

        self.logger.info("")

        self.logger.info("Data Types")

        for column in self.dataframe.columns:

            self.logger.info(
                f"{column:<40} {self.dataframe[column].dtype}"
            )

        self.logger.info("")

        self.logger.info("Target Distribution")

        target_distribution = (
            self.dataframe[self.TARGET_COLUMN]
            .value_counts()
            .sort_index()
        )

        for cls, count in target_distribution.items():

            percentage = (
                count / len(self.dataframe)
            ) * 100

            self.logger.info(
                f"{cls:<40}"
                f"{count:>8,}"
                f" ({percentage:.2f}%)"
            )

        ###########################################################################
    # Train / Validation / Test Split
    ###########################################################################

    def perform_train_validation_test_split(self):

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Performing Stratified Train / Validation / Test Split")
        self.logger.info("=" * 70)

        train_df, temp_df = train_test_split(
            self.dataframe,
            test_size=0.30,
            random_state=self.random_state,
            stratify=self.dataframe[self.TARGET_COLUMN],
        )

        validation_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=self.random_state,
            stratify=temp_df[self.TARGET_COLUMN],
        )

        self.train_df = train_df.reset_index(drop=True)
        self.validation_df = validation_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        self.logger.info("Dataset splitting completed.")

        self.logger.info(
            f"Training Samples   : {len(self.train_df):,}"
        )

        self.logger.info(
            f"Validation Samples : {len(self.validation_df):,}"
        )

        self.logger.info(
            f"Test Samples       : {len(self.test_df):,}"
        )

    ###########################################################################
    # Save Datasets
    ###########################################################################

    def save_datasets(self):

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Saving Processed Datasets")
        self.logger.info("=" * 70)

        train_path = self.output_directory / "classification_train.csv"
        validation_path = (
            self.output_directory /
            "classification_validation.csv"
        )
        test_path = self.output_directory / "classification_test.csv"

        self.train_df.to_csv(
            train_path,
            index=False
        )

        self.validation_df.to_csv(
            validation_path,
            index=False
        )

        self.test_df.to_csv(
            test_path,
            index=False
        )

        self.logger.info(f"Saved : {train_path.name}")
        self.logger.info(f"Saved : {validation_path.name}")
        self.logger.info(f"Saved : {test_path.name}")

    ###########################################################################
    # Split Report
    ###########################################################################

    def _class_distribution(self, dataframe):

        distribution = (
            dataframe[self.TARGET_COLUMN]
            .value_counts()
            .sort_index()
        )

        results = {}

        for cls, count in distribution.items():

            results[cls] = {
                "count": int(count),
                "percentage": round(
                    (count / len(dataframe)) * 100,
                    2
                )
            }

        return results

    def generate_split_report(self):

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Dataset Split Summary")
        self.logger.info("=" * 70)

        datasets = {
            "Original": self.dataframe,
            "Training": self.train_df,
            "Validation": self.validation_df,
            "Test": self.test_df,
        }

        for dataset_name, dataframe in datasets.items():

            self.logger.info("")
            self.logger.info("-" * 70)
            self.logger.info(dataset_name)
            self.logger.info("-" * 70)

            self.logger.info(
                f"Total Samples : {len(dataframe):,}"
            )

            distribution = self._class_distribution(dataframe)

            for cls, values in distribution.items():

                self.logger.info(
                    f"{cls:<40}"
                    f"{values['count']:>8,}"
                    f" ({values['percentage']:.2f}%)"
                )

    ###########################################################################
    # Metadata
    ###########################################################################

    def save_metadata(self):

        metadata = {

            "created_on": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "random_state": self.random_state,

            "original_dataset_rows": int(len(self.dataframe)),
            "training_rows": int(len(self.train_df)),
            "validation_rows": int(len(self.validation_df)),
            "test_rows": int(len(self.test_df)),

            "original_distribution":
                self._class_distribution(self.dataframe),

            "training_distribution":
                self._class_distribution(self.train_df),

            "validation_distribution":
                self._class_distribution(self.validation_df),

            "test_distribution":
                self._class_distribution(self.test_df),
        }

        metadata_path = (
            self.output_directory /
            "classification_split_metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

        self.logger.info("")
        self.logger.info(
            f"Metadata Saved : {metadata_path.name}"
        )

    ###########################################################################
    # Run Pipeline
    ###########################################################################

    def run(self):

        self.load_dataset()

        self.validate_dataset()

        self.dataset_summary()

        self.perform_train_validation_test_split()

        self.save_datasets()

        self.generate_split_report()

        self.save_metadata()

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("Classification Data Preparation Completed")
        self.logger.info("=" * 70)


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":

    INPUT_DATASET = (
        "data/raw/supply_chain_iot_cold_chain_temp_forecast_80k.csv"
    )

    OUTPUT_DIRECTORY = (
        "data/processed"
    )

    pipeline = ClassificationDataPreparation(
        data_path=INPUT_DATASET,
        version="v1",
        random_state=42,
    )

    pipeline.run()