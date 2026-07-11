import random

class EmergencyDataGenerator:

    @staticmethod
    def generate_emergency_row(row):
        new_row = row.copy()

        # High outside temperature
        new_row["Ambient_External_Temp_C"] = round(
            random.uniform(40,45), 1
        )

        # Actual temperature much higher than set point
        set_point = new_row["Internal_Set_Point_C"]

        new_row["Actual_Internal_Temp_C"] = round(
            set_point + random.uniform(6, 12), 1
        )

        # HVAC working very hard
        new_row["HVAC_Power_Consumption_Watts"] = round(
            random.uniform(1800, 2500), 1
        )

        # Poor seal
        new_row["Seal_Integrity_Index"] = round(
            random.uniform(0.25, 0.60), 3
        )

        # Very high spoilage risk
        new_row["Forecasted_4Hr_Spoilage_Risk_Pct"] = round(
            random.uniform(80, 100), 1
        )

        new_row[
            "Logistics_Action_Recommendation"
        ] = "EMERGENCY_Reroute"

        return new_row
    
    @staticmethod
    def generate_warning_row(row):

        new_row = row.copy()

        # Slightly higher ambient temperature
        new_row["Ambient_External_Temp_C"] = round(
            random.uniform(35, 40), 1
        )

        set_point = new_row["Internal_Set_Point_C"]

        # Slight temperature deviation
        new_row["Actual_Internal_Temp_C"] = round(
            set_point + random.uniform(2, 5), 1
        )

        # HVAC working harder
        new_row["HVAC_Power_Consumption_Watts"] = round(
            random.uniform(1400, 1800), 1
        )

        # Moderate seal degradation
        new_row["Seal_Integrity_Index"] = round(
            random.uniform(0.60, 0.80), 3
        )

        # Medium spoilage risk
        new_row["Forecasted_4Hr_Spoilage_Risk_Pct"] = round(
            random.uniform(45, 75), 1
        )

        new_row[
            "Logistics_Action_Recommendation"
        ] = "WARNING_Request_HVAC_Remote_Reset"

        return new_row
