"""
Test script for PredictionService.

Run:

python -m src.prediction.test_prediction_service
"""

from src.prediction.prediction_models import PredictionRequest
from src.prediction.prediction_service import PredictionService


def run_test(name: str, request: PredictionRequest):

    print("=" * 80)
    print(f"TEST : {name}")
    print("=" * 80)

    service = PredictionService()

    try:

        response = service.predict(request)

        print(f"Predicted Spoilage Risk : {response.predicted_spoilage_risk_pct:.2f}%")
        print(f"Predicted Action        : {response.predicted_action}")
        print(f"Confidence             : {response.confidence:.6f}")

        print("\nClass Probabilities")

        for action, probability in response.probabilities.items():
            print(f"  {action:<40} : {probability:.6f}")

        print(f"\nWarning Threshold      : {response.warning_threshold:.6f}")
        print(f"Model                  : {response.model_name}")
        print(f"Latency               : {response.prediction_latency_ms:.2f} ms")

    except ValueError as ex:

        print(f"\nValidation Failed : {ex}")

    except Exception as ex:

        print(f"\nPrediction Failed : {ex}")

        raise

    print()


if __name__ == "__main__":

    # -------------------------------------------------------
    # Test 1
    # Normal Shipment
    # -------------------------------------------------------

    run_test(
        "SAFE Shipment",
        PredictionRequest(
            cargo_type="Frozen Food",
            ambient_external_temp_c=24,
            internal_set_point_c=-18,
            actual_internal_temp_c=-18,
            hvac_power_consumption_watts=1200,
            seal_integrity_index=0.98,
        ),
    )

    # -------------------------------------------------------
    # Test 2
    # Warning
    # -------------------------------------------------------

    run_test(
        "WARNING Shipment",
        PredictionRequest(
            cargo_type="Frozen Food",
            ambient_external_temp_c=38,
            internal_set_point_c=-18,
            actual_internal_temp_c=-14,
            hvac_power_consumption_watts=1700,
            seal_integrity_index=0.72,
        ),
    )

    # -------------------------------------------------------
    # Test 3
    # Emergency
    # -------------------------------------------------------

    run_test(
        "EMERGENCY Shipment",
        PredictionRequest(
            cargo_type="Frozen Food",
            ambient_external_temp_c=42,
            internal_set_point_c=-18,
            actual_internal_temp_c=-8,
            hvac_power_consumption_watts=2200,
            seal_integrity_index=0.40,
        ),
    )

    # -------------------------------------------------------
    # Test 4
    # Invalid Seal
    # -------------------------------------------------------

    run_test(
        "Invalid Seal",
        PredictionRequest(
            cargo_type="Frozen Food",
            ambient_external_temp_c=25,
            internal_set_point_c=-18,
            actual_internal_temp_c=-18,
            hvac_power_consumption_watts=1200,
            seal_integrity_index=1.5,
        ),
    )

    # -------------------------------------------------------
    # Test 5
    # Invalid HVAC
    # -------------------------------------------------------

    run_test(
        "Invalid HVAC",
        PredictionRequest(
            cargo_type="Frozen Food",
            ambient_external_temp_c=25,
            internal_set_point_c=-18,
            actual_internal_temp_c=-18,
            hvac_power_consumption_watts=-100,
            seal_integrity_index=0.95,
        ),
    )