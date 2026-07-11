import joblib
from sklearn.preprocessing import LabelEncoder

class Encoder:
        
        @staticmethod
        def encode_column(df, column_name, output_path):
            encoder = LabelEncoder()
            df[column_name] = encoder.fit_transform(df[column_name])
            joblib.dump(encoder, output_path)
            print(f"{column_name} encoded.")
            return df