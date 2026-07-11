import pandas as pd

class DataLoader:
    """
    Load the dataset from the specified file path
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Load CSV dataset into a Pandas DataFrame
        """
        try:
            df = pd.read_csv(self.file_path)

            print("Dataset Loaded Successfully")
            print(f"Rows:  {df.shape[0]}")
            print(f"Columns: {df.shape[1]}")

            return df

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

