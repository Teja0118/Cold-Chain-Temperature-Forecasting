from __future__ import annotations

import pandas as pd

from src.prediction.prediction_models import (
    PredictionRequest,
)


INPUT_COLUMNS = [
    "Cargo_Type",
    "Ambient_External_Temp_C",
    "Internal_Set_Point_C",
    "Actual_Internal_Temp_C",
    "HVAC_Power_Consumption_Watts",
    "Seal_Integrity_Index",
]


PREDICTED_RISK = "Predicted_Spoilage_Risk_Pct"


def validate_request(request: PredictionRequest) -> None:
    """
    Validate prediction request values.
    """

    if request.ambient_external_temp_c < -40 or request.ambient_external_temp_c > 70:
        raise ValueError(
            "Ambient_External_Temp_C must be between -40 and 70."
        )

    if request.internal_set_point_c < -40 or request.internal_set_point_c > 20:
        raise ValueError(
            "Internal_Set_Point_C must be between -40 and 20."
        )

    if request.actual_internal_temp_c < -40 or request.actual_internal_temp_c > 70:
        raise ValueError(
            "Actual_Internal_Temp_C must be between -40 and 70."
        )

    if request.hvac_power_consumption_watts < 0:
        raise ValueError(
            "HVAC_Power_Consumption_Watts cannot be negative."
        )

    if not 0 <= request.seal_integrity_index <= 1:
        raise ValueError(
            "Seal_Integrity_Index must be between 0 and 1."
        )


def request_to_dataframe(
    request: PredictionRequest,
) -> pd.DataFrame:
    """
    Convert PredictionRequest to a single-row DataFrame.
    """

    return pd.DataFrame(
        [
            {
                "Cargo_Type": request.cargo_type,
                "Ambient_External_Temp_C": request.ambient_external_temp_c,
                "Internal_Set_Point_C": request.internal_set_point_c,
                "Actual_Internal_Temp_C": request.actual_internal_temp_c,
                "HVAC_Power_Consumption_Watts": request.hvac_power_consumption_watts,
                "Seal_Integrity_Index": request.seal_integrity_index,
            }
        ]
    )


def add_predicted_risk(
    frame: pd.DataFrame,
    predicted_risk: float,
) -> pd.DataFrame:
    """
    Append predicted spoilage risk to the feature set.
    """

    output = frame.copy()

    output[PREDICTED_RISK] = predicted_risk

    return output


def extract_probabilities(
    probabilities,
    classes,
) -> dict:
    """
    Convert predict_proba output to a JSON-friendly dictionary.
    """

    return {
        class_name: round(float(probability), 6)
        for class_name, probability in zip(
            classes,
            probabilities[0],
        )
    }

def extract_confidence(
    probabilities,
) -> float:
    """
    Return the highest prediction confidence.
    """

    return round(
        float(probabilities.max()),
        6,
    )

def is_emergency_condition(
    frame: pd.DataFrame,
) -> pd.Series:
    """
    Determine whether the physical conditions
    indicate an Emergency.
    """

    deviation = (
        frame["Actual_Internal_Temp_C"]
        - frame["Internal_Set_Point_C"]
    )

    emergency = (
        (deviation >= 6)
        & (frame["Seal_Integrity_Index"] < 0.60)
        & (frame["Ambient_External_Temp_C"] >= 38)
    )

    return emergency

def apply_warning_threshold(
    probabilities,
    label_encoder,
    warning_threshold: float,
) -> int:
    """
    Apply the calibrated Warning threshold.

    If the Warning probability is greater than or
    equal to the calibrated threshold, return the
    Warning class.

    Otherwise return the Safe class.
    """

    safe_code = int(
        label_encoder.transform(
            ["SAFE_Maintain_Course"]
        )[0]
    )

    warning_code = int(
        label_encoder.transform(
            ["WARNING_Request_HVAC_Remote_Reset"]
        )[0]
    )

    if (
        probabilities[0][warning_code]
        >= warning_threshold
    ):
        return warning_code

    return safe_code