import os

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

class FeatureSelector:

    def __init__(self, X, y):

        self.X = X.copy()
        self.y = y.copy()

    def correlation_analysis(self):

        correlation = self.X.corr()

        correlation.to_csv(
            "outputs/feature_correlation.csv"
        )

        print("Feature correlation saved.")

    def mutual_information(self):

        mi = mutual_info_regression(
            self.X,
            self.y
        )

        mi_df = pd.DataFrame({

            "Feature": self.X.columns,

            "Mutual_Information": mi

        })

        mi_df = mi_df.sort_values(

            by="Mutual_Information",

            ascending=False

        )

        print("\nMutual Information\n")

        print(mi_df)

        return mi_df

    def feature_importance(self):

        model = RandomForestRegressor(

            n_estimators=100,

            random_state=42

        )

        model.fit(

            self.X,

            self.y

        )

        importance_df = pd.DataFrame({

            "Feature": self.X.columns,

            "Importance": model.feature_importances_

        })

        importance_df = importance_df.sort_values(

            by="Importance",

            ascending=False

        )

        importance_df.to_csv(

            "outputs/feature_importance.csv",

            index=False

        )

        print("\nFeature Importance\n")

        print(importance_df)

        return importance_df

    def save_selected_features(

        self,

        importance_df

    ):

        selected = importance_df[

            importance_df["Importance"] > 0.01

        ]

        with open(

            "outputs/selected_features.txt",

            "w"

        ) as file:

            for feature in selected["Feature"]:

                file.write(feature + "\n")

        print("\nSelected features saved.")

    def select_features(self):

        os.makedirs(

            "outputs",

            exist_ok=True

        )

        self.correlation_analysis()

        self.mutual_information()

        importance = self.feature_importance()

        self.save_selected_features(

            importance

        )

        rfe = self.recursive_feature_elimination()

        return importance, rfe
    
    def recursive_feature_elimination(self):

        estimator = RandomForestRegressor(

            n_estimators=100,

            random_state=42

        )

        rfe = RFE(

            estimator=estimator,

            n_features_to_select=8

        )

        rfe.fit(

            self.X,

            self.y

        )

        rfe_df = pd.DataFrame({

            "Feature": self.X.columns,

            "Selected": rfe.support_,

            "Ranking": rfe.ranking_

        })

        rfe_df = rfe_df.sort_values(

            by="Ranking"

        )

        rfe_df.to_csv(

            "outputs/rfe_features.csv",

            index=False

        )

        print("\nRecursive Feature Elimination\n")

        print(rfe_df)

        return rfe_df