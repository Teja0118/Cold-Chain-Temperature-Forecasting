import os
import joblib

from sklearn.model_selection import train_test_split


class TrainTestSplitter:

    def __init__(self, X, y_reg, y_cls):

        self.X = X
        self.y_reg = y_reg
        self.y_cls = y_cls

    def split(self):

        (
            X_train,
            X_test,
            y_reg_train,
            y_reg_test,
            y_cls_train,
            y_cls_test

        ) = train_test_split(

            self.X,

            self.y_reg,

            self.y_cls,

            test_size=0.2,

            random_state=42,

            stratify=self.y_cls

        )

        os.makedirs("models", exist_ok=True)

        joblib.dump(X_train, "models/X_train.pkl")
        joblib.dump(X_test, "models/X_test.pkl")

        joblib.dump(y_reg_train, "models/y_reg_train.pkl")
        joblib.dump(y_reg_test, "models/y_reg_test.pkl")

        joblib.dump(y_cls_train, "models/y_cls_train.pkl")
        joblib.dump(y_cls_test, "models/y_cls_test.pkl")

        print("\nTrain-Test Split Completed")
        print(f"Training Samples : {len(X_train)}")
        print(f"Testing Samples  : {len(X_test)}")

        return (
            X_train,
            X_test,
            y_reg_train,
            y_reg_test,
            y_cls_train,
            y_cls_test
        )