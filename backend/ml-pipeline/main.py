"""Regression-only entry point for cold-chain spoilage-risk modelling."""

import argparse

from src.training.regression_pipeline import RegressionTrainingPipeline


def main():
    parser = argparse.ArgumentParser(description="Train the spoilage-risk regression model.")
    parser.add_argument(
        "--data-path",
        default="data/raw/supply_chain_iot_cold_chain_temp_forecast_80k.csv",
        help="Path to the versioned master raw CSV.",
    )
    parser.add_argument("--version", default="v1", help="Model/data version label.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--augment-domain-scenarios",
        action="store_true",
        help="Add training-only synthetic Warning/Emergency risk scenarios.",
    )
    args = parser.parse_args()

    RegressionTrainingPipeline(
        data_path=args.data_path,
        version=args.version,
        random_state=args.random_state,
        augment_domain_scenarios=args.augment_domain_scenarios,
    ).run()


if __name__ == "__main__":
    main()
