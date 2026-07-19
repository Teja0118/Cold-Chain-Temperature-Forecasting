"""
===============================================================================
Feature Transformer Module
===============================================================================

Applies the same feature engineering transformations to validation and
test datasets using the logic learned during training.

===============================================================================
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


class ClassificationFeatureTransformer:

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
    # Temperature Deviation
    ####################################################################

    def temperature_deviation(self):

        self.logger.info(
            "Creating Temperature_Deviation..."
        )

        self.df["Temperature_Deviation"] = (

            self.df["Actual_Internal_Temp_C"]

            -

            self.df["Internal_Set_Point_C"]

        )

    ####################################################################
    # Ambient Load
    ####################################################################

    def ambient_load(self):

        self.logger.info(
            "Creating Ambient_Load..."
        )

        self.df["Ambient_Load"] = (

            self.df["Ambient_External_Temp_C"]

            -

            self.df["Internal_Set_Point_C"]

        )

    ####################################################################
    # HVAC Efficiency
    ####################################################################

    def hvac_efficiency(self):

        self.logger.info(
            "Creating HVAC_Efficiency..."
        )

        self.df["HVAC_Efficiency"] = (

            self.df["Temperature_Deviation"]

            /

            (
                self.df[
                    "HVAC_Power_Consumption_Watts"
                ] + 1
            )

        )

    ####################################################################
    # Cooling Stress
    ####################################################################

    def cooling_stress(self):

        self.logger.info(
            "Creating Cooling_Stress..."
        )

        self.df["Cooling_Stress"] = (

            self.df["Ambient_Load"]

            *

            self.df[
                "HVAC_Power_Consumption_Watts"
            ]

        )

    ####################################################################
    # Seal Loss
    ####################################################################

    def seal_loss(self):

        self.logger.info(
            "Creating Seal_Loss..."
        )

        self.df["Seal_Loss"] = (

            1

            -

            self.df["Seal_Integrity_Index"]

        )

    ####################################################################
    # Thermal Stress Index
    ####################################################################

    def thermal_stress_index(self):

        self.logger.info(
            "Creating Thermal_Stress_Index..."
        )

        self.df["Thermal_Stress_Index"] = (

            self.df["Temperature_Deviation"]

            *

            self.df["Ambient_Load"]

        )

    ####################################################################
    # HVAC Load Ratio
    ####################################################################

    def hvac_load_ratio(self):

        self.logger.info(
            "Creating HVAC_Load_Ratio..."
        )

        self.df["HVAC_Load_Ratio"] = (

            self.df[
                "HVAC_Power_Consumption_Watts"
            ]

            /

            (
                self.df["Ambient_Load"].abs()

                + 1
            )

        )

    ####################################################################
    # Temperature Risk Index
    ####################################################################

    def temperature_risk_index(self):

        self.logger.info(
            "Creating Temperature_Risk_Index..."
        )

        self.df["Temperature_Risk_Index"] = (

            self.df["Temperature_Deviation"].abs()

            *

            self.df["Seal_Loss"]

        )

    ####################################################################
    # Cooling Performance Score
    ####################################################################

    def cooling_performance_score(self):

        self.logger.info(
            "Creating Cooling_Performance_Score..."
        )

        self.df["Cooling_Performance_Score"] = (

            self.df["HVAC_Power_Consumption_Watts"]

            /

            (
                self.df["Temperature_Deviation"].abs()

                + 1
            )

        )

    ####################################################################
    # Overall Risk Score
    ####################################################################

    def overall_risk_score(self):

        self.logger.info(
            "Creating Overall_Risk_Score..."
        )

        self.df["Overall_Risk_Score"] = (

            self.df["Temperature_Risk_Index"]

            +

            self.df["Cooling_Stress"]

            +

            self.df["Seal_Loss"]

        )

        ####################################################################
    # Validate Engineered Features
    ####################################################################

    def validate_engineered_features(self):

        self.logger.info(
            "Validating engineered features..."
        )

        engineered_features = [

            "Temperature_Deviation",

            "Ambient_Load",

            "HVAC_Efficiency",

            "Cooling_Stress",

            "Seal_Loss",

            "Thermal_Stress_Index",

            "HVAC_Load_Ratio",

            "Temperature_Risk_Index",

            "Cooling_Performance_Score",

            "Overall_Risk_Score"

        ]

        for feature in engineered_features:

            if self.df[feature].isnull().any():

                raise ValueError(
                    f"Missing values found in {feature}"
                )

            if np.isinf(
                self.df[feature]
            ).any():

                raise ValueError(
                    f"Infinite values found in {feature}"
                )

        self.logger.info(
            "Feature validation completed."
        )

    ####################################################################
    # Save Dataset
    ####################################################################

    def save_dataset(self):

        self.df.to_csv(

            self.output_file,

            index=False

        )

        self.logger.info(
            f"Dataset saved : {self.output_file}"
        )

    ####################################################################
    # Run Pipeline
    ####################################################################

    def run(self):

        self.load_dataset()

        self.temperature_deviation()

        self.ambient_load()

        self.hvac_efficiency()

        self.cooling_stress()

        self.seal_loss()

        self.thermal_stress_index()

        self.hvac_load_ratio()

        self.temperature_risk_index()

        self.cooling_performance_score()

        self.overall_risk_score()

        self.validate_engineered_features()

        self.save_dataset()

        self.logger.info(
            "Feature Transformation Completed."
        )


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    transformer = ClassificationFeatureTransformer(

        input_file="data/processed/classification_validation.csv",

        output_file="data/engineered/classification_validation_engineered.csv",

        version="v1"

    )

    transformer.run()