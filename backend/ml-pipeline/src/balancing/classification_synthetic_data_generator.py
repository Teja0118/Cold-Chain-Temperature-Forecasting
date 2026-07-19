"""
===============================================================================
Synthetic Data Generator
===============================================================================

Generates realistic WARNING and EMERGENCY samples for the cold-chain
classification dataset.

===============================================================================
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


class ClassificationSyntheticDataGenerator:

    TARGET_COLUMN = "Logistics_Action_Recommendation"

    SAFE_CLASS = "SAFE_Maintain_Course"

    WARNING_CLASS = "WARNING_Request_HVAC_Remote_Reset"

    EMERGENCY_CLASS = "EMERGENCY"

    def __init__(
        self,
        version: str = "v1",
        random_state: int = 42,
    ):

        self.version = version

        self.random_state = random_state

        self.input_file = Path(
            "data/engineered/classification_train_engineered.csv"
        )

        self.output_directory = Path(
            "data/balanced"
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

        np.random.seed(self.random_state)

        self.df = None

        self.safe_df = None

        self.warning_df = None

        self.synthetic_warning = None

        self.synthetic_emergency = None

    ####################################################################
    # Load Dataset
    ####################################################################

    def load_dataset(self):

        self.logger.info(
            "Loading selected feature dataset..."
        )

        self.df = pd.read_csv(
            self.input_file
        )

        self.safe_df = self.df[
            self.df[self.TARGET_COLUMN] ==
            self.SAFE_CLASS
        ].copy()

        self.warning_df = self.df[
            self.df[self.TARGET_COLUMN] ==
            self.WARNING_CLASS
        ].copy()

        self.logger.info(
            f"SAFE Samples     : {len(self.safe_df):,}"
        )

        self.logger.info(
            f"WARNING Samples  : {len(self.warning_df):,}"
        )

    ####################################################################
    # Feature Statistics
    ####################################################################

    def calculate_statistics(self):

        self.logger.info(
            "Computing feature statistics..."
        )

        numeric = self.warning_df.select_dtypes(
            include=np.number
        )

        self.warning_mean = numeric.mean()

        self.warning_std = numeric.std()

        self.warning_min = numeric.min()

        self.warning_max = numeric.max()

    ####################################################################
    # Clip Values
    ####################################################################

    def clip_values(self, dataframe):

        numeric_columns = dataframe.select_dtypes(
            include=np.number
        ).columns

        for column in numeric_columns:

            dataframe[column] = dataframe[column].clip(
                lower=self.warning_min[column],
                upper=self.warning_max[column]
            )

        return dataframe

    ####################################################################
    # Generate WARNING Samples
    ####################################################################

    def generate_warning_samples(
        self,
        required_samples: int
    ):

        self.logger.info(
            f"Generating {required_samples:,} WARNING samples..."
        )

        samples = []

        numeric_columns = self.warning_df.select_dtypes(
            include=np.number
        ).columns

        for _ in range(required_samples):

            row = {}

            for column in self.warning_df.columns:

                if column == self.TARGET_COLUMN:

                    row[column] = self.WARNING_CLASS

                elif column in numeric_columns:

                    value = np.random.normal(
                        self.warning_mean[column],
                        self.warning_std[column] * 0.35
                    )

                    row[column] = value

                else:

                    row[column] = np.random.choice(
                        self.warning_df[column]
                    )

            samples.append(row)

        self.synthetic_warning = pd.DataFrame(
            samples
        )

        self.synthetic_warning = self.clip_values(
            self.synthetic_warning
        )

        self.logger.info(
            "WARNING generation completed."
        )

        ####################################################################
    # EMERGENCY Scenario - Compressor Failure
    ####################################################################

    def compressor_failure(self, row):

        row["Actual_Internal_Temp_C"] += np.random.uniform(8, 15)

        row["HVAC_Power_Consumption_Watts"] *= np.random.uniform(
            1.20,
            1.45
        )

        row["Seal_Integrity_Index"] *= np.random.uniform(
            0.90,
            1.00
        )

        return row

    ####################################################################
    # EMERGENCY Scenario - Refrigerant Leak
    ####################################################################

    def refrigerant_leak(self, row):

        row["Actual_Internal_Temp_C"] += np.random.uniform(6, 12)

        row["HVAC_Power_Consumption_Watts"] *= np.random.uniform(
            1.10,
            1.35
        )

        row["Seal_Integrity_Index"] *= np.random.uniform(
            0.95,
            1.00
        )

        return row

    ####################################################################
    # EMERGENCY Scenario - Door Open
    ####################################################################

    def door_open(self, row):

        row["Actual_Internal_Temp_C"] += np.random.uniform(10, 18)

        row["Seal_Integrity_Index"] *= np.random.uniform(
            0.55,
            0.80
        )

        row["HVAC_Power_Consumption_Watts"] *= np.random.uniform(
            1.10,
            1.30
        )

        return row

    ####################################################################
    # EMERGENCY Scenario - Seal Failure
    ####################################################################

    def seal_failure(self, row):

        row["Seal_Integrity_Index"] *= np.random.uniform(
            0.20,
            0.60
        )

        row["Actual_Internal_Temp_C"] += np.random.uniform(5, 10)

        row["HVAC_Power_Consumption_Watts"] *= np.random.uniform(
            1.10,
            1.25
        )

        return row

    ####################################################################
    # EMERGENCY Scenario - Combined Failure
    ####################################################################

    def combined_failure(self, row):

        row = self.compressor_failure(row)

        row = self.seal_failure(row)

        row["Actual_Internal_Temp_C"] += np.random.uniform(3, 6)

        return row

    ####################################################################
    # Generate EMERGENCY Samples
    ####################################################################

    def generate_emergency_samples(
        self,
        required_samples: int
    ):

        self.logger.info(
            f"Generating {required_samples:,} EMERGENCY samples..."
        )

        scenarios = [

            self.compressor_failure,

            self.refrigerant_leak,

            self.door_open,

            self.seal_failure,

            self.combined_failure

        ]

        emergency_samples = []

        for _ in range(required_samples):

            row = self.warning_df.sample(
                n=1,
                replace=True,
                random_state=None
            ).iloc[0].copy()

            scenario = np.random.choice(
                scenarios
            )

            row = scenario(row)

            row[self.TARGET_COLUMN] = self.EMERGENCY_CLASS

            emergency_samples.append(row)

        self.synthetic_emergency = pd.DataFrame(
            emergency_samples
        )

        self.synthetic_emergency = self.clip_values(
            self.synthetic_emergency
        )

        self.logger.info(
            "EMERGENCY generation completed."
        )
    
        ####################################################################
    # Validate Synthetic Dataset
    ####################################################################

    def validate_dataset(self):

        self.logger.info(
            "Validating synthetic samples..."
        )

        datasets = [

            ("WARNING", self.synthetic_warning),

            ("EMERGENCY", self.synthetic_emergency)

        ]

        for dataset_name, dataframe in datasets:

            if dataframe.empty:

                raise ValueError(
                    f"{dataset_name} dataset is empty."
                )

            if dataframe.isnull().sum().sum() > 0:

                raise ValueError(
                    f"{dataset_name} contains missing values."
                )

            numeric_columns = dataframe.select_dtypes(
                include=np.number
            ).columns

            for column in numeric_columns:

                if np.isinf(dataframe[column]).any():

                    raise ValueError(
                        f"Infinite values found in "
                        f"{dataset_name} -> {column}"
                    )

        self.logger.info(
            "Validation completed successfully."
        )

    ####################################################################
    # Merge Datasets
    ####################################################################

    def merge_datasets(self):

        self.logger.info(
            "Merging datasets..."
        )

        self.final_dataset = pd.concat(

            [

                self.safe_df,

                self.warning_df,

                self.synthetic_warning,

                self.synthetic_emergency

            ],

            ignore_index=True

        )

        self.final_dataset = self.final_dataset.sample(

            frac=1,

            random_state=self.random_state

        ).reset_index(drop=True)

        self.logger.info(
            f"Final Dataset Shape : {self.final_dataset.shape}"
        )

    ####################################################################
    # Dataset Summary
    ####################################################################

    def dataset_summary(self):

        self.logger.info("")

        self.logger.info("=" * 70)

        self.logger.info("Final Class Distribution")

        self.logger.info("=" * 70)

        distribution = (

            self.final_dataset[self.TARGET_COLUMN]

            .value_counts()

            .sort_index()

        )

        for cls, count in distribution.items():

            percentage = (

                count /

                len(self.final_dataset)

            ) * 100

            self.logger.info(

                f"{cls:<40}"

                f"{count:>8,}"

                f" ({percentage:.2f}%)"

            )

    ####################################################################
    # Save Dataset
    ####################################################################

    def save_dataset(self):

        output_file = (

            self.output_directory /

            "classification_balanced_dataset.csv"

        )

        self.final_dataset.to_csv(

            output_file,

            index=False

        )

        self.logger.info(
            f"Dataset Saved : {output_file}"
        )

    ####################################################################
    # Run Pipeline
    ####################################################################

    def run(self):

        self.load_dataset()

        self.calculate_statistics()

        warning_needed = (

            len(self.safe_df)

            -

            len(self.warning_df)

        )

        emergency_needed = len(self.safe_df)

        self.generate_warning_samples(
            warning_needed
        )

        self.generate_emergency_samples(
            emergency_needed
        )

        self.validate_dataset()

        self.merge_datasets()

        self.dataset_summary()

        self.save_dataset()

        self.logger.info("")

        self.logger.info("=" * 70)

        self.logger.info(
            "Synthetic Dataset Generation Completed."
        )

        self.logger.info("=" * 70)


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    pipeline = ClassificationSyntheticDataGenerator(
        version="v1",
        random_state=42
    )

    pipeline.run()