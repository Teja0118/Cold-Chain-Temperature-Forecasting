"""
Regression Trainer
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from .model_evaluator import ModelEvaluator
from .model_saver import ModelSaver
from .model_comparator import ModelComparator

class RegressionTrainer:

    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        # Store training and testing data
        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test

    def train_linear_regression(self):
        model = LinearRegression()
        model.fit(
            self.X_train,
            self.y_train
        )
        predictions = model.predict(
            self.X_test
        )

        metrics = ModelEvaluator.evaluate_regression(
            "Linear Regression",
            self.y_test,
            predictions
        )

        ModelSaver.save_model(
            model,
            "linear_regression.pkl"
        )

        ModelComparator.save_metrics(
            "Linear Regression",
            metrics
        )

        return model, metrics
    
    def train_random_forest(self):
        """
        Train a Random Forest Regressor.

        Why Random Forest?
        ------------------
        - Captures non-linear relationships.
        - Handles feature interactions automatically.
        - Less sensitive to outliers.
        - Usually performs much better than Linear Regression
        for structured/tabular datasets.
        """

        print("\nTraining Random Forest Regressor...")

        # Create the model with initial hyperparameters.
        # These values provide a good balance between
        # accuracy and training time.
        # Tuned Random Forest Model
        model = RandomForestRegressor(

            n_estimators=300,

            max_depth=10,

            min_samples_split=10,

            min_samples_leaf=4,

            random_state=42,

            n_jobs=-1

        )

        # Train the model
        model.fit(

            self.X_train,

            self.y_train

        )

        # Predict on unseen data
        predictions = model.predict(

            self.X_test

        )

        # Evaluate performance
        metrics = ModelEvaluator.evaluate_regression(

            "Random Forest Regressor",

            self.y_test,

            predictions

        )

        # Save trained model
        ModelSaver.save_model(

            model,

            "random_forest_regressor.pkl"

        )

        ModelComparator.save_metrics(
            "Random Forest",
            metrics
        )

        return model, metrics
    
    def train_xgboost(self):
        """
        Train an XGBoost Regressor.

        XGBoost generally performs exceptionally well on structured
        tabular datasets by using gradient boosting over decision trees.
        """
        print("Training XGBoost Regressor...")

        # Tuned XGBoost Model
        model = XGBRegressor(

            n_estimators=300,

            learning_rate=0.03,

            max_depth=4,

            subsample=0.7,

            colsample_bytree=1.0,

            objective="reg:squarederror",

            random_state=42,

            n_jobs=-1

        )

        # Train model
        model.fit(
            self.X_train,
            self.y_train
        )

        # Predictions
        predictions = model.predict(
            self.X_test
        )

        # Evaluate
        metrics = ModelEvaluator.evaluate_regression(
            "XGBoost Regressor",
            self.y_test,
            predictions
        )

        # Save model
        ModelSaver.save_model(
            model,
            "xgboost_regressor.pkl"
        )

        # Add to comparision table
        ModelComparator.save_metrics(
            "XGBoost",
            metrics
        )

        return model, metrics
    
    def train_lightgbm(self):
        """
        Train a LightGBM Regressor.

        LightGBM is a gradient boosting framework optimized for
        speed and memory efficiency. It generally trains faster
        than XGBoost while providing comparable performance.
        """
        print("Train LightGBM Regressor...")

        # Initial LightGBM model
        # Tuned LightGBM Model
        model = LGBMRegressor(

            n_estimators=700,

            learning_rate=0.01,

            max_depth=10,

            num_leaves=31,

            subsample=0.8,

            colsample_bytree=1.0,

            random_state=42,

            n_jobs=-1,

            verbosity=-1

        )

        # Train the model
        model.fit(
            self.X_train,
            self.y_train
        )

        # Predict
        predictions = model.predict(
            self.X_test
        )

        # Evaluate
        metrics = ModelEvaluator.evaluate_regression(
            "LightGBM Regressor",
            self.y_test,
            predictions
        )

        # Save model
        ModelSaver.save_model(
            model,
            "lightgbm_regressor.pkl"
        )

        # Update comparsion table
        ModelComparator.save_metrics(
            "LightGBM",
            metrics
        )

        return model, metrics
