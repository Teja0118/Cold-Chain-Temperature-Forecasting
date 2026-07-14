"""
Target Variable Analysis

This module analyses the regression target variable
(Forecasted_4Hr_Spoilage_Risk_Pct).
"""

import os

import matplotlib.pyplot as plt

import pandas as pd

from scipy.stats import skew

from scipy.stats import kurtosis


class TargetAnalyzer:

    def __init__(

        self,

        target

    ):

        self.target = target

    def analyze(self):

        print("\nTarget Variable Analysis")
        print("-" * 40)

        print("\nStatistical Summary")

        print(

            self.target.describe()

        )

        print(

            "\nSkewness :",

            round(

                skew(self.target),

                4

            )

        )

        print(

            "Kurtosis :",

            round(

                kurtosis(self.target),

                4

            )

        )

        # Create output folder
        os.makedirs(

            "reports/target_analysis",

            exist_ok=True

        )

        # Histogram
        plt.figure(

            figsize=(8,5)

        )

        plt.hist(

            self.target,

            bins=20

        )

        plt.title(

            "Target Distribution"

        )

        plt.xlabel(

            "Forecasted Spoilage Risk (%)"

        )

        plt.ylabel(

            "Frequency"

        )

        plt.tight_layout()

        plt.savefig(

            "reports/target_analysis/target_distribution.png"

        )

        plt.close()

        # Distribution in bins

        bins = [

            0,

            10,

            20,

            30,

            40,

            50,

            60,

            70,

            80,

            90,

            100

        ]

        labels = [

            "0-10",

            "10-20",

            "20-30",

            "30-40",

            "40-50",

            "50-60",

            "60-70",

            "70-80",

            "80-90",

            "90-100"

        ]

        distribution = pd.cut(

            self.target,

            bins=bins,

            labels=labels,

            include_lowest=True

        ).value_counts().sort_index()

        print(

            "\nTarget Distribution"

        )

        print(

            distribution

        )

        distribution.to_csv(

            "reports/target_analysis/target_distribution.csv"

        )

        print(

            "\nReports saved to reports/target_analysis/"
        )