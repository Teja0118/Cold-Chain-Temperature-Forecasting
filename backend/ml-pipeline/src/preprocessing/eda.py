class EDA:
    @staticmethod
    def dataset_info(df):
        print("Dataset Information:")
        print(df.info())

    @staticmethod
    def first_rows(df):
        print("First 5 rows:")
        print(df.head())

    @staticmethod
    def last_rows(df):
        print("Last 5 rows:")
        print(df.tail())

    @staticmethod
    def missing_values(df):
        print("Missing Values:")
        print(df.isnull().sum())

    @staticmethod
    def duplicate_rows(df):
        print("Duplicate rows:")
        print(df.duplicated().sum())

    @staticmethod
    def statistical_summary(df):
        print("Statistical Summary:")
        print(df.describe(include="all"))

    @staticmethod
    def categorical_summary(df):
        print("Categorical Features:")

        categorical_columns = df.select_dtypes(include=["object"]).columns

        for column in categorical_columns:
            print(f"\n{column}")
            print(df[column].value_counts())