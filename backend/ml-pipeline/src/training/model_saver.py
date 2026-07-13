"""
Model Saving Utility

Saves trained models to disk
"""

import os
import joblib

class ModelSaver:

    @staticmethod
    def save_model(model, filename):
        os.makedirs("models", exist_ok=True)
        path = f"models/{filename}"
        joblib.dump(model,path)
        print(f"Model Saved: {path}")

    