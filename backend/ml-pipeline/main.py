import joblib

from src.analysis.target_analyzer import TargetAnalyzer


def main():

    # -------------------------------------------------------
    # Load Saved Train/Test Data
    # -------------------------------------------------------

    y_reg_train = joblib.load(

        "models/y_reg_train.pkl"

    )

    # -------------------------------------------------------
    # Target Variable Analysis
    # -------------------------------------------------------

    target_analysis = TargetAnalyzer(

        y_reg_train

    )

    target_analysis.analyze()


if __name__ == "__main__":
    main()



"""
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.eda import EDA
from src.balancing.dataset_balancer import DatasetBalancer
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.feature_selection.feature_selector import FeatureSelector
from src.training.train_test_splitter import TrainTestSplitter
from src.training.regression_trainer import RegressionTrainer
from src.training.model_comparator import ModelComparator
from src.training.hyperparameter_tuner import HyperparameterTuner
from src.training.model_saver import ModelSaver
from src.analysis.target_analyzer import TargetAnalyzer

DATASET_PATH = "data/raw/supply_chain_iot_cold_chain_temp_forecast_80k.csv"

def main():

    # -------------------------------------------------------
    # STEP 1 - DATA PREPARATION (Already Completed)
    # Uncomment only if the dataset changes.
    # -------------------------------------------------------

    
    # Load Dataset
    loader = DataLoader(DATASET_PATH)
    df = loader.load_data()

    # EDA
    EDA.first_rows(df)
    EDA.last_rows(df)
    EDA.dataset_info(df)
    EDA.missing_values(df)
    EDA.duplicate_rows(df)
    EDA.statistical_summary(df)
    EDA.categorical_summary(df)

    # Dataset Balancing
    balancer = DatasetBalancer(df)

    balanced_df = balancer.balance()

    balancer.save_dataset(
        "data/balanced/cold_chain_dataset_balanced.csv"
    )

    

    
    # Preprocessing
    preprocessor = DataPreprocessor(balanced_df)

    X, y_reg, y_cls = preprocessor.preprocess()

    # Feature Engineering
    feature_engineer = FeatureEngineer(X)

    X_engineered = feature_engineer.engineer_features()

    # Feature Selection
    selector = FeatureSelector(
        X_engineered,
        y_reg
    )

    importance_df, rfe_df = selector.select_features()

    selected_columns = [

        "Cargo_Type",
        "Ambient_External_Temp_C",
        "Actual_Internal_Temp_C",
        "HVAC_Power_Consumption_Watts",
        "Seal_Integrity_Index",
        "Temperature_Deviation_C",
        "HVAC_Efficiency",
        "HVAC_Stress_Index"

    ]

    X_final = X_engineered[selected_columns]

    # Train-Test Split
    splitter = TrainTestSplitter(
        X_final,
        y_reg,
        y_cls
    )

    (
        X_train,
        X_test,
        y_reg_train,
        y_reg_test,
        y_cls_train,
        y_cls_test
    ) = splitter.split()
    

    # -------------------------------------------------------
    # STEP 2 - LOAD PREVIOUSLY SAVED TRAIN/TEST DATA
    # -------------------------------------------------------

    import joblib

    X_train = joblib.load("models/X_train.pkl")
    X_test = joblib.load("models/X_test.pkl")

    y_reg_train = joblib.load("models/y_reg_train.pkl")
    y_reg_test = joblib.load("models/y_reg_test.pkl")

    
    target_analysis = TargetAnalyzer(

        y_reg_train

    )

    target_analysis.analyze()

    # -------------------------------------------------------
    # STEP 3 - MODEL TRAINING
    # -------------------------------------------------------

    regression_trainer = RegressionTrainer(

        X_train,

        X_test,

        y_reg_train,

        y_reg_test

    )

    linear_model, linear_metrics = (

        regression_trainer.train_linear_regression()

    )

    # -------------------------------------------------------
    # Random Forest Regression
    # -------------------------------------------------------

    rf_model, rf_metrics = (

        regression_trainer.train_random_forest()

    )

    # -------------------------------------------------------
    # XGBoost Regression
    # -------------------------------------------------------

    xgb_model, xgb_metrics = (

        regression_trainer.train_xgboost()

    )

    # -------------------------------------------------------
    # LightGBM Regression
    # -------------------------------------------------------

    lgbm_model, lgbm_metrics = (

        regression_trainer.train_lightgbm()

    )

    # Display the current best regression model
    ModelComparator.best_model()


    feature_columns = [

        "Cargo_Type",

        "Ambient_External_Temp_C",

        "Actual_Internal_Temp_C",

        "HVAC_Power_Consumption_Watts",

        "Seal_Integrity_Index",

        "Temperature_Deviation_C",

        "HVAC_Efficiency",

        "HVAC_Stress_Index"

    ]

    # Since LightGBM is currently the best model,
    # save it as the production model.
    ModelSaver.save_production_model(

        lgbm_model,

        feature_columns

    )

    
    
    
    # -------------------------------------------------------
    # Hyperparameter Tuning
    # -------------------------------------------------------

    tuner = HyperparameterTuner(

        X_train,

        y_reg_train

    )


    # best_rf = tuner.tune_random_forest()

    best_xgb = tuner.tune_xgboost()

    # best_lgbm = tuner.tune_lightgbm()

    

if __name__ == "__main__":
    main()

"""