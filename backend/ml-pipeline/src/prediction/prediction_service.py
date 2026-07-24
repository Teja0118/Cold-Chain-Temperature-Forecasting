"""
Prediction Service

Inference Pipeline

Input
    ↓
Regression Model
    ↓
Predicted Spoilage Risk
    ↓
Classification Model
    ↓
Warning Threshold
    ↓
Emergency Override
    ↓
Prediction Response
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.prediction.model_loader import ModelLoader
from src.prediction.prediction_models import (
    PredictionRequest,
    PredictionResponse,
)
from src.prediction.prediction_utils import (
    validate_request,
    request_to_dataframe,
    add_predicted_risk,
    extract_confidence,
    extract_probabilities,
    apply_warning_threshold,
    is_emergency_condition,
)

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Production Prediction Service.

    Loads trained models once and performs
    end-to-end logistics action prediction.

    Flow

    Request
        ↓
    Validation
        ↓
    Regression
        ↓
    Predicted Risk
        ↓
    Classification
        ↓
    Emergency Override
        ↓
    Response
    """

    def __init__(self):

        loader = ModelLoader()

        self.pipeline = loader.classifier

        self.regression_model = loader.regression

        self.label_encoder = loader.label_encoder

        self.warning_threshold = loader.warning_threshold

        self.model_name = loader.model_name

        self.input_columns = loader.input_columns

        logger.info(
            "Prediction Service initialized "
            "(Model=%s, Threshold=%.2f)",
            self.model_name,
            self.warning_threshold,
        )

    def _predict_risk(
        self,
        frame: pd.DataFrame,
    ) -> float:
        """
        Predict spoilage risk using the regression model.
        """

        predicted_risk = float(
            self.regression_model.predict(
                frame[self.input_columns]
            )[0]
        )

        predicted_risk = max(
            0.0,
            min(100.0, predicted_risk)
        )

        logger.info(
            "Predicted Spoilage Risk : %.2f",
            predicted_risk,
        )

        return predicted_risk

    """

    def _predict_probabilities(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:
        
        # Predict class probabilities.

        probabilities = self.pipeline.predict_proba(frame)

        return probabilities
    """

    def _predict_probabilities(
        self,
        frame: pd.DataFrame,
    ) -> np.ndarray:

        probabilities = self.pipeline.predict_proba(frame)

        """ 
        print("\nRaw Probabilities")
        print(probabilities)
        """

        return probabilities
        

    def _decode_prediction(
        self,
        prediction: int,
    ) -> str:
        """
        Convert encoded class to label.
        """

        return self.label_encoder.inverse_transform(
            [prediction]
        )[0]

    def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:
        """
        Perform end-to-end prediction.

        Flow

        Request
            ↓
        Validation
            ↓
        Regression Prediction
            ↓
        Predicted Risk Feature
            ↓
        Classification Prediction
            ↓
        Warning Threshold
            ↓
        Emergency Override
            ↓
        Response
        """

        start_time = time.perf_counter()

        logger.info("Starting prediction...")

        # ---------------------------------------------
        # Validate Input
        # ---------------------------------------------

        validate_request(request)

        # ---------------------------------------------
        # Convert Request to DataFrame
        # ---------------------------------------------

        frame = request_to_dataframe(request)

        # ---------------------------------------------
        # Predict Spoilage Risk
        # ---------------------------------------------

        predicted_risk = self._predict_risk(frame)

        # ---------------------------------------------
        # Append Predicted Risk
        # ---------------------------------------------

        frame = add_predicted_risk(
            frame,
            predicted_risk,
        )

        # ---------------------------------------------
        # Predict Class Probabilities
        # ---------------------------------------------

        probabilities = self._predict_probabilities(
            frame
        )

        # ---------------------------------------------
        # Apply Warning Threshold
        # ---------------------------------------------

        prediction = apply_warning_threshold(
            probabilities=probabilities,
            label_encoder=self.label_encoder,
            warning_threshold=self.warning_threshold,
        )

        # ---------------------------------------------
        # Emergency Override
        # ---------------------------------------------

        if bool(
            is_emergency_condition(frame).iloc[0]
        ):

            logger.info(
                "Emergency Override Triggered."
            )

            prediction = int(
                self.label_encoder.transform(
                    ["EMERGENCY"]
                )[0]
            )

        # ---------------------------------------------
        # Decode Prediction
        # ---------------------------------------------

        predicted_action = self._decode_prediction(
            prediction
        )

        # ---------------------------------------------
        # Confidence
        # ---------------------------------------------

        confidence = extract_confidence(
            probabilities
        )

        probability_dict = extract_probabilities(
            probabilities,
            self.label_encoder.classes_,
        )

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "Prediction completed in %.2f ms",
            elapsed_ms,
        )

        logger.info(
            "Predicted Action : %s",
            predicted_action,
        )

        return PredictionResponse(
            predicted_spoilage_risk_pct=predicted_risk,
            predicted_action=predicted_action,
            confidence=confidence,
            probabilities=probability_dict,
            warning_threshold=self.warning_threshold,
            model_name=self.model_name,
            prediction_latency_ms=round(elapsed_ms, 2),
        )