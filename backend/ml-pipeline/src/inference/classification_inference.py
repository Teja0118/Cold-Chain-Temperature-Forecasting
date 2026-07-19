"""
===============================================================================
Classification Inference
===============================================================================

Loads the trained classification model and predicts the logistics action
for incoming shipment data.

===============================================================================
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class ClassificationInference:

    def __init__(
        self,
        model_path: str,
        encoder_path: str,
    ):

        self.model_path = Path(model_path)

        self.encoder_path = Path(encoder_path)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

        self.logger = logging.getLogger(__name__)

        self.load_model()

    ####################################################################
    # Load Model
    ####################################################################

    def load_model(self):

        self.logger.info(
            "Loading classification model..."
        )

        self.model = joblib.load(
            self.model_path
        )

        self.encoder = joblib.load(
            self.encoder_path
        )

        self.logger.info(
            "Model loaded successfully."
        )

    ####################################################################
    # Prepare Input
    ####################################################################

    def prepare_input(
        self,
        input_data: dict,
    ):

        dataframe = pd.DataFrame(
            [input_data]
        )

        return dataframe

    ####################################################################
    # Predict Class
    ####################################################################

    def predict(
        self,
        input_data: dict,
    ):

        dataframe = self.prepare_input(
            input_data
        )

        prediction = self.model.predict(
            dataframe
        )[0]

        probabilities = self.model.predict_proba(
            dataframe
        )[0]

        predicted_class = self.encoder.inverse_transform(
            [prediction]
        )[0]

        confidence = float(
            np.max(probabilities)
        )

        probability_scores = {}

        for class_name, probability in zip(

            self.encoder.classes_,

            probabilities

        ):

            probability_scores[class_name] = round(
                float(probability),
                4
            )

        return {

            "predicted_class": predicted_class,

            "confidence": round(
                confidence,
                4
            ),

            "probabilities": probability_scores

        }
    
        ####################################################################
    # Risk Level
    ####################################################################

    def determine_risk_level(
        self,
        predicted_class: str,
    ):

        mapping = {

            "SAFE_Maintain_Course": "LOW",

            "WARNING_Request_HVAC_Remote_Reset": "MEDIUM",

            "EMERGENCY": "HIGH"

        }

        return mapping.get(
            predicted_class,
            "UNKNOWN"
        )

    ####################################################################
    # Operational Recommendation
    ####################################################################

    def generate_recommendation(
        self,
        predicted_class: str,
    ):

        recommendations = {

            "SAFE_Maintain_Course": [
                "Continue shipment.",
                "Monitor temperature normally.",
                "No corrective action required."
            ],

            "WARNING_Request_HVAC_Remote_Reset": [
                "Request remote HVAC reset.",
                "Increase telemetry monitoring.",
                "Notify operations team."
            ],

            "EMERGENCY": [
                "Stop shipment immediately.",
                "Escalate to emergency response.",
                "Inspect refrigeration unit.",
                "Protect cargo from spoilage."
            ]

        }

        return recommendations.get(
            predicted_class,
            ["No recommendation available."]
        )

    ####################################################################
    # Prediction Summary
    ####################################################################

    def predict_with_summary(
        self,
        input_data: dict,
    ):

        result = self.predict(
            input_data
        )

        risk_level = self.determine_risk_level(
            result["predicted_class"]
        )

        recommendation = self.generate_recommendation(
            result["predicted_class"]
        )

        return {

            "predicted_class":
                result["predicted_class"],

            "risk_level":
                risk_level,

            "confidence":
                result["confidence"],

            "probabilities":
                result["probabilities"],

            "recommendation":
                recommendation

        }


##############################################################################
# Main
##############################################################################

if __name__ == "__main__":

    MODEL_PATH = (
        "models/classification/best_tuned_model.pkl"
    )

    ENCODER_PATH = (
        "models/classification/label_encoder.pkl"
    )

    inference = ClassificationInference(
        model_path=MODEL_PATH,
        encoder_path=ENCODER_PATH
    )

    sample = {

        "Ambient_External_Temp_C": 37.5,

        "Internal_Set_Point_C": 4.0,

        "Actual_Internal_Temp_C": 7.8,

        "HVAC_Power_Consumption_Watts": 3450,

        "Seal_Integrity_Index": 0.82,

        "Temperature_Deviation": 3.8,

        "Ambient_Load": 33.5,

        "HVAC_Efficiency": 0.0011,

        "Cooling_Stress": 115575,

        "Seal_Loss": 0.18,

        "Thermal_Stress_Index": 127.3,

        "HVAC_Load_Ratio": 100.0,

        "Temperature_Risk_Index": 0.684,

        "Cooling_Performance_Score": 718.75,

        "Overall_Risk_Score": 115575.864

    }

    result = inference.predict_with_summary(
        sample
    )

    print(result)