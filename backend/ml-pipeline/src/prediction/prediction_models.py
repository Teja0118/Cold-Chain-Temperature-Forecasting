from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class PredictionRequest:
    """
    Input required for a single shipment prediction.
    """

    cargo_type: str
    ambient_external_temp_c: float
    internal_set_point_c: float
    actual_internal_temp_c: float
    hvac_power_consumption_watts: float
    seal_integrity_index: float


@dataclass(slots=True)
class PredictionResponse:
    """
    Output returned by the prediction service.
    """

    predicted_spoilage_risk_pct: float

    predicted_action: str

    confidence: float

    probabilities: Dict[str, float]

    warning_threshold: float
    model_name: str
    prediction_latency_ms: float