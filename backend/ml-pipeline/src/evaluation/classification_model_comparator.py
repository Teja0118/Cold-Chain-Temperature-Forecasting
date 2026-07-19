"""
===============================================================================
Classification Model Comparator
===============================================================================

Compares baseline and tuned classification models and selects the best
deployment candidate.

===============================================================================
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class ClassificationModelComparator:

    def __init__(
        self,
        version: str = "v1",
    ):

        self.version = version

        self.baseline_results = Path(
            f"models/classification/{self.version}/classification_results.csv"
        )

        self.tuned_results = Path(
            f"models/classification/{self.version}/hyperparameter_results.csv"
        )

        self.output_directory = Path(
            f"reports/classification/{self.version}/comparison"
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s"
        )

        self.logger = logging.getLogger(__name__)

    ####################################################################
    # Load Results
    ####################################################################

    def load_results(self):

        self.logger.info(
            "Loading model results..."
        )

        baseline = pd.read_csv(
            self.baseline_results
        )

        baseline["Category"] = "Baseline"

        tuned = pd.read_csv(
            self.tuned_results
        )

        tuned["Category"] = "Tuned"

        self.results = pd.concat(

            [

                baseline,

                tuned

            ],

            ignore_index=True

        )

    ####################################################################
    # Rank Models
    ####################################################################

    def rank_models(self):

        self.logger.info(
            "Ranking models..."
        )

        self.results.sort_values(

            by="F1_Score",

            ascending=False,

            inplace=True

        )

        self.results.reset_index(

            drop=True,

            inplace=True

        )

        self.results["Rank"] = (

            self.results.index + 1

        )

    ####################################################################
    # Save Leaderboard
    ####################################################################

    def save_leaderboard(self):

        self.logger.info(
            "Saving leaderboard..."
        )

        self.results.to_csv(

            self.output_directory /

            "model_leaderboard.csv",

            index=False

        )

    ####################################################################
    # Save Best Model Metadata
    ####################################################################

    def save_best_model(self):

        best = self.results.iloc[0]

        metadata = {

            "Model":

                best["Model"],

            "Category":

                best["Category"],

            "Accuracy":

                float(best["Accuracy"]),

            "Precision":

                float(best["Precision"]),

            "Recall":

                float(best["Recall"]),

            "F1_Score":

                float(best["F1_Score"])

        }

        with open(

            self.output_directory /

            "best_model.json",

            "w"

        ) as file:

            json.dump(

                metadata,

                file,

                indent=4

            )

        ####################################################################
    # Performance Charts
    ####################################################################

    def generate_plots(self):

        self.logger.info(
            "Generating performance plots..."
        )

        metrics = [

            "Accuracy",

            "Precision",

            "Recall",

            "F1_Score"

        ]

        for metric in metrics:

            plt.figure(figsize=(10, 6))

            sorted_df = self.results.sort_values(
                by=metric,
                ascending=False
            )

            plt.bar(
                sorted_df["Model"],
                sorted_df[metric]
            )

            plt.title(f"{metric} Comparison")

            plt.xlabel("Models")

            plt.ylabel(metric)

            plt.xticks(rotation=45)

            plt.tight_layout()

            plt.savefig(

                self.output_directory /

                f"{metric.lower()}_comparison.png",

                dpi=300

            )

            plt.close()

    ####################################################################
    # Deployment Recommendation
    ####################################################################

    def deployment_recommendation(self):

        self.logger.info(
            "Generating deployment recommendation..."
        )

        best = self.results.iloc[0]

        recommendation = {

            "recommended_model":
                best["Model"],

            "model_category":
                best["Category"],

            "reason":
                "Highest weighted F1 Score.",

            "accuracy":
                float(best["Accuracy"]),

            "precision":
                float(best["Precision"]),

            "recall":
                float(best["Recall"]),

            "f1_score":
                float(best["F1_Score"])

        }

        with open(

            self.output_directory /

            "deployment_recommendation.json",

            "w"

        ) as file:

            json.dump(
                recommendation,
                file,
                indent=4
            )

    ####################################################################
    # Summary Report
    ####################################################################

    def summary_report(self):

        self.logger.info(
            "Generating summary report..."
        )

        summary = {

            "total_models":

                int(len(self.results)),

            "baseline_models":

                int(

                    (self.results["Category"] == "Baseline")

                    .sum()

                ),

            "tuned_models":

                int(

                    (self.results["Category"] == "Tuned")

                    .sum()

                ),

            "best_model":

                self.results.iloc[0]["Model"],

            "best_f1_score":

                float(

                    self.results.iloc[0]["F1_Score"]

                )

        }

        with open(

            self.output_directory /

            "comparison_summary.json",

            "w"

        ) as file:

            json.dump(
                summary,
                file,
                indent=4
            )

    ####################################################################
    # Run
    ####################################################################

    def run(self):

        self.load_results()

        self.rank_models()

        self.save_leaderboard()

        self.save_best_model()

        self.generate_plots()

        self.deployment_recommendation()

        self.summary_report()

        self.logger.info(
            "Model Comparison Completed."
        )


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    comparator = ClassificationModelComparator(
        version="v1"
    )

    comparator.run()