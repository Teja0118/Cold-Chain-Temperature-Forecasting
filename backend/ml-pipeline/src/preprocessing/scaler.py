import joblib
from sklearn.preprocessing import StandardScaler

class Scaler:

    @staticmethod
    def scale_features(df, columns, output_path):
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        joblib.dump(scaler, output_path)
        print("Numerical features scaled.")
        return df