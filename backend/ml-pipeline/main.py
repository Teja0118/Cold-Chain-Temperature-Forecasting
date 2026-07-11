from src.preprocessing.data_loader import DataLoader
from src.preprocessing.eda import EDA
from src.balancing.dataset_balancer import DatasetBalancer
from src.preprocessing.data_preprocessor import DataPreprocessor
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.feature_selection.feature_selector import FeatureSelector

DATASET_PATH = "data/raw/supply_chain_iot_cold_chain_temp_forecast_80k.csv"

def main():
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

    # Dataset balancing
    balancer = DatasetBalancer(df)

    balanced_df = balancer.balance()

    balancer.save_dataset(
        "data/balanced/cold_chain_dataset_balanced.csv"
    )

    preprocessor = DataPreprocessor(balanced_df)

    X, y_reg, y_cls = preprocessor.preprocess()

    feature_engineer = FeatureEngineer(X)

    X_engineered = feature_engineer.engineer_features()

    selector = FeatureSelector(

        X_engineered,

        y_reg

    )

    importance_df, rfe_df = selector.select_features()

if __name__ == "__main__":
    main()