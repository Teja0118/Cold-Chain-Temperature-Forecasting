"""
===============================================================================
Classification Exploratory Data Analysis (EDA)
===============================================================================

Performs exploratory data analysis on the classification dataset.

Outputs
-------
- Dataset summary
- Class distribution
- Numerical statistics
- Class-wise statistics
- Correlation analysis
- Feature importance
- EDA reports
===============================================================================
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


class ClassificationEDA:

    TARGET_COLUMN = "Logistics_Action_Recommendation"

    def __init__(
        self,
        version: str = "v1",
    ):
        self.version = version

        self.input_file = Path(
            "data/processed/classification_train.csv"
        )
        
        self.output_directory = Path(
            f"reports/classification/{self.version}"
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.distribution_directory = (
            self.output_directory / "distributions"
        )

        self.correlation_directory = (
            self.output_directory / "correlation"
        )

        self.distribution_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        self.correlation_directory.mkdir(
            parents=True,
            exist_ok=True
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

        self.logger = logging.getLogger(__name__)

        self.df = None

    #######################################################################
    # Load Dataset
    #######################################################################

    def load_dataset(self):

        self.logger.info("Loading dataset...")

        self.df = pd.read_csv(self.input_file)

        self.logger.info(
            f"Dataset Shape : {self.df.shape}"
        )

    #######################################################################
    # Dataset Overview
    #######################################################################

    def dataset_overview(self):

        self.logger.info("Generating dataset overview...")

        overview = {

            "rows": len(self.df),

            "columns": len(self.df.columns),

            "duplicate_rows": int(
                self.df.duplicated().sum()
            ),

            "missing_values": self.df.isnull().sum().to_dict(),

            "data_types": {
                col: str(dtype)
                for col, dtype in self.df.dtypes.items()
            },

            "memory_usage_mb": round(
                self.df.memory_usage(deep=True).sum() /
                (1024 * 1024),
                2
            )
        }

        with open(
            self.output_directory / "dataset_overview.json",
            "w"
        ) as file:

            json.dump(
                overview,
                file,
                indent=4
            )

    #######################################################################
    # Target Analysis
    #######################################################################

    def target_analysis(self):

        self.logger.info("Generating class distribution...")

        distribution = (
            self.df[self.TARGET_COLUMN]
            .value_counts()
            .rename_axis("Class")
            .reset_index(name="Count")
        )

        distribution["Percentage"] = (
            distribution["Count"] /
            len(self.df)
        ) * 100

        distribution.to_csv(
            self.output_directory /
            "class_distribution.csv",
            index=False
        )

    #######################################################################
    # Numerical Statistics
    #######################################################################

    def numerical_statistics(self):

        self.logger.info(
            "Generating numerical statistics..."
        )

        numerical_df = self.df.select_dtypes(
            include="number"
        )

        summary = numerical_df.describe(
            percentiles=[
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
            ]
        ).T

        summary.to_csv(
            self.output_directory /
            "numerical_statistics.csv"
        )

    #######################################################################
    # Class-wise Statistics
    #######################################################################

    def classwise_statistics(self):

        self.logger.info(
            "Generating class-wise statistics..."
        )

        classwise = self.df.groupby(
            self.TARGET_COLUMN
        ).describe()

        classwise.to_csv(
            self.output_directory /
            "classwise_statistics.csv"
        )

        #######################################################################
    # Distribution Plots
    #######################################################################

    def distribution_analysis(self):

        self.logger.info(
            "Generating distribution plots..."
        )

        numerical_columns = self.df.select_dtypes(
            include="number"
        ).columns

        for column in numerical_columns:

            # Histogram + KDE
            plt.figure(figsize=(8, 5))

            sns.histplot(
                self.df[column],
                kde=True
            )

            plt.title(column)

            plt.tight_layout()

            plt.savefig(
                self.distribution_directory /
                f"{column}_histogram.png"
            )

            plt.close()

            # Boxplot
            plt.figure(figsize=(8, 2))

            sns.boxplot(
                x=self.df[column]
            )

            plt.title(column)

            plt.tight_layout()

            plt.savefig(
                self.distribution_directory /
                f"{column}_boxplot.png"
            )

            plt.close()

    #######################################################################
    # Correlation Analysis
    #######################################################################

    def correlation_analysis(self):

        self.logger.info(
            "Generating correlation analysis..."
        )

        numerical_df = self.df.select_dtypes(
            include="number"
        )

        pearson = numerical_df.corr(
            method="pearson"
        )

        spearman = numerical_df.corr(
            method="spearman"
        )

        pearson.to_csv(
            self.correlation_directory /
            "pearson.csv"
        )

        spearman.to_csv(
            self.correlation_directory /
            "spearman.csv"
        )

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            pearson,
            annot=True,
            cmap="coolwarm"
        )

        plt.tight_layout()

        plt.savefig(
            self.correlation_directory /
            "pearson_heatmap.png"
        )

        plt.close()

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            spearman,
            annot=True,
            cmap="coolwarm"
        )

        plt.tight_layout()

        plt.savefig(
            self.correlation_directory /
            "spearman_heatmap.png"
        )

        plt.close()

    #######################################################################
    # SAFE vs WARNING Comparison
    #######################################################################

    def class_comparison(self):

        self.logger.info(
            "Generating class comparison..."
        )

        numerical_columns = self.df.select_dtypes(
            include="number"
        ).columns

        grouped = self.df.groupby(
            self.TARGET_COLUMN
        )[numerical_columns].mean()

        if len(grouped.index) < 2:
            self.logger.warning(
                "Comparison skipped."
            )
            return

        safe = grouped.iloc[0]
        warning = grouped.iloc[1]

        comparison = pd.DataFrame({

            "Feature": numerical_columns,

            "SAFE_Mean": safe.values,

            "WARNING_Mean": warning.values,

            "Difference":
                warning.values - safe.values

        })

        comparison.to_csv(
            self.output_directory /
            "safe_warning_comparison.csv",
            index=False
        )

    #######################################################################
    # Outlier Analysis
    #######################################################################

    def outlier_analysis(self):

        self.logger.info(
            "Generating outlier analysis..."
        )

        results = []

        numerical_columns = self.df.select_dtypes(
            include="number"
        ).columns

        for column in numerical_columns:

            q1 = self.df[column].quantile(0.25)
            q3 = self.df[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - (1.5 * iqr)
            upper = q3 + (1.5 * iqr)

            outliers = self.df[
                (self.df[column] < lower) |
                (self.df[column] > upper)
            ]

            results.append({

                "Feature": column,

                "Q1": q1,

                "Q3": q3,

                "IQR": iqr,

                "Outlier_Count": len(outliers),

                "Outlier_Percentage":
                    round(
                        len(outliers) /
                        len(self.df) * 100,
                        2
                    )

            })

        pd.DataFrame(results).to_csv(

            self.output_directory /
            "outlier_analysis.csv",

            index=False

        )

        #######################################################################
    # Feature Target Relationship
    #######################################################################

    def feature_target_relationship(self):

        self.logger.info(
            "Generating feature-target relationship..."
        )

        numerical_df = self.df.select_dtypes(
            include="number"
        ).copy()

        target = self.df[self.TARGET_COLUMN]

        encoder = LabelEncoder()

        encoded_target = encoder.fit_transform(target)

        mi_scores = mutual_info_classif(
            numerical_df,
            encoded_target,
            random_state=0
        )

        relationship = pd.DataFrame({

            "Feature": numerical_df.columns,

            "Mutual_Information": mi_scores

        })

        relationship.sort_values(
            by="Mutual_Information",
            ascending=False,
            inplace=True
        )

        relationship.to_csv(
            self.output_directory /
            "feature_target_relationship.csv",
            index=False
        )

    #######################################################################
    # Feature Importance
    #######################################################################

    def feature_importance(self):

        self.logger.info(
            "Generating feature importance..."
        )

        numerical_df = self.df.select_dtypes(
            include="number"
        ).copy()

        target = self.df[self.TARGET_COLUMN]

        encoder = LabelEncoder()

        encoded_target = encoder.fit_transform(target)

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            numerical_df,
            encoded_target
        )

        importance = pd.DataFrame({

            "Feature": numerical_df.columns,

            "Importance": model.feature_importances_

        })

        importance.sort_values(
            by="Importance",
            ascending=False,
            inplace=True
        )

        importance.to_csv(
            self.output_directory /
            "feature_importance.csv",
            index=False
        )

    #######################################################################
    # EDA Summary
    #######################################################################

    def eda_summary(self):

        self.logger.info(
            "Generating EDA summary..."
        )

        importance = pd.read_csv(
            self.output_directory /
            "feature_importance.csv"
        )

        correlation = pd.read_csv(
            self.correlation_directory /
            "pearson.csv",
            index_col=0
        )

        highest_correlations = (
            correlation.abs()
            .unstack()
            .sort_values(ascending=False)
            .drop_duplicates()
            .head(10)
        )

        summary = {

            "top_predictors":
                importance.head(5)["Feature"].tolist(),

            "weak_predictors":
                importance.tail(5)["Feature"].tolist(),

            "highest_correlations": {
                f"{feature1} <-> {feature2}": float(value)
                for (feature1, feature2), value in highest_correlations.items()
            },

            "recommended_next_step":
                "Proceed to Feature Engineering.",

            "possible_target_leakage":
                [
                    "Forecasted_4Hr_Spoilage_Risk_Pct"
                ]

        }

        with open(
            self.output_directory /
            "eda_summary.json",
            "w"
        ) as file:

            json.dump(
                summary,
                file,
                indent=4
            )

    #######################################################################
    # Run
    #######################################################################

    def run(self):

        self.load_dataset()

        self.dataset_overview()

        self.target_analysis()

        self.numerical_statistics()

        self.classwise_statistics()

        self.distribution_analysis()

        self.correlation_analysis()

        self.class_comparison()

        self.outlier_analysis()

        self.feature_target_relationship()

        self.feature_importance()

        self.eda_summary()

        self.logger.info(
            "EDA completed successfully."
        )


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    eda = ClassificationEDA(
        version="v1"
    )

    eda.run()