"""
Hyperparameter Tuning

Uses RandomizedSearchCV to find the best parameters for
different regression models.
"""

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor


class HyperparameterTuner:

    def __init__(

        self,

        X_train,

        y_train

    ):

        self.X_train = X_train

        self.y_train = y_train

    #######################################################
    # Random Forest
    #######################################################

    def tune_random_forest(self):

        print("\nTuning Random Forest...")

        model = RandomForestRegressor(

            random_state=42,

            n_jobs=-1

        )

        params = {

            "n_estimators": [200, 300, 500, 700],

            "max_depth": [8, 10, 15, 20, None],

            "min_samples_split": [2, 5, 10],

            "min_samples_leaf": [1, 2, 4]

        }

        search = RandomizedSearchCV(

            estimator=model,

            param_distributions=params,

            n_iter=20,

            cv=5,

            scoring="r2",

            random_state=42,

            n_jobs=-1

        )

        search.fit(

            self.X_train,

            self.y_train

        )

        print("\nBest Random Forest Parameters")

        print(search.best_params_)

        print("Best CV Score :", search.best_score_)

        return search.best_estimator_

    #######################################################
    # XGBoost
    #######################################################

    def tune_xgboost(self):

        print("\nTuning XGBoost...")

        model = XGBRegressor(

            objective="reg:squarederror",

            random_state=42,

            n_jobs=-1

        )

        params = {

            "n_estimators": [300, 500, 700],

            "learning_rate": [0.01, 0.03, 0.05, 0.1],

            "max_depth": [4, 6, 8, 10],

            "subsample": [0.7, 0.8, 0.9, 1.0],

            "colsample_bytree": [0.7, 0.8, 0.9, 1.0]

        }

        search = RandomizedSearchCV(

            estimator=model,

            param_distributions=params,

            n_iter=20,

            cv=5,

            scoring="r2",

            random_state=42,

            n_jobs=-1

        )

        search.fit(

            self.X_train,

            self.y_train

        )

        print("\nBest XGBoost Parameters")

        print(search.best_params_)

        print("Best CV Score :", search.best_score_)

        return search.best_estimator_

    #######################################################
    # LightGBM
    #######################################################

    def tune_lightgbm(self):

        print("\nTuning LightGBM...")

        model = LGBMRegressor(

            random_state=42,

            verbosity=-1

        )

        params = {

            "n_estimators": [300, 500, 700],

            "learning_rate": [0.01, 0.03, 0.05, 0.1],

            "max_depth": [5, 8, 10, 15],

            "num_leaves": [31, 50, 80, 120],

            "subsample": [0.7, 0.8, 0.9, 1.0],

            "colsample_bytree": [0.7, 0.8, 0.9, 1.0]

        }

        search = RandomizedSearchCV(

            estimator=model,

            param_distributions=params,

            n_iter=20,

            cv=5,

            scoring="r2",

            random_state=42,

            n_jobs=-1

        )

        search.fit(

            self.X_train,

            self.y_train

        )

        print("\nBest LightGBM Parameters")

        print(search.best_params_)

        print("Best CV Score :", search.best_score_)

        return search.best_estimator_