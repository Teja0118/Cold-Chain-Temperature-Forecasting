"""
Main entry point for the Cold Chain Temperature Forecasting ML pipeline.
Supports both Regression and Classification workflows.
"""

import argparse

from src.training.regression_pipeline import RegressionTrainingPipeline
from src.training.classification_safe_pipeline import ClassificationPipeline


def main():

    parser = argparse.ArgumentParser(
        description="Cold Chain ML Pipeline"
    )

    parser.add_argument(
        "--pipeline",
        choices=[
            "regression",
            "classification",
            "both"
        ],
        default="both",
        help="Pipeline to execute."
    )

    parser.add_argument(
        "--data-path",
        default="data/raw/supply_chain_iot_cold_chain_temp_forecast_80k.csv",
        help="Path to the master dataset."
    )

    parser.add_argument(
        "--version",
        default="v1",
        help="Pipeline version."
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42
    )

    parser.add_argument(
        "--augment-domain-scenarios",
        action="store_true",
        help="Generate synthetic domain scenarios for regression."
    )

    args = parser.parse_args()

    ###############################################################
    # Regression Pipeline
    ###############################################################

    if args.pipeline in ["regression", "both"]:

        RegressionTrainingPipeline(

            data_path=args.data_path,

            version=args.version,

            random_state=args.random_state,

            augment_domain_scenarios=args.augment_domain_scenarios

        ).run()

    ###############################################################
    # Classification Pipeline
    ###############################################################

    if args.pipeline in ["classification", "both"]:

        ClassificationPipeline(

            data_path=args.data_path,

            version=args.version,

            random_state=args.random_state

        ).run()


if __name__ == "__main__":

    main()


# python main.py --pipeline classification
