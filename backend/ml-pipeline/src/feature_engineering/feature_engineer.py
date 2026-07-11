class FeatureEngineer:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

    def create_temperature_deviation(self):

        self.df["Temperature_Deviation_C"] = (

            self.df["Actual_Internal_Temp_C"]

            -

            self.df["Internal_Set_Point_C"]

        )

        print("Temperature_Deviation_C created.")

    def create_cooling_load(self):

        self.df["Cooling_Load_C"] = (

            self.df["Ambient_External_Temp_C"]

            -

            self.df["Internal_Set_Point_C"]

        )

        print("Cooling_Load_C created.")

    def create_hvac_efficiency(self):

        self.df["HVAC_Efficiency"] = (

            self.df["Cooling_Load_C"]

            /

            self.df["HVAC_Power_Consumption_Watts"]

        )

        print("HVAC_Efficiency created.")

    def create_seal_loss_score(self):

        self.df["Seal_Loss_Score"] = (

            1

            -

            self.df["Seal_Integrity_Index"]

        )

        print("Seal_Loss_Score created.")

    def create_hvac_stress_index(self):

        self.df["HVAC_Stress_Index"] = (

            self.df["HVAC_Power_Consumption_Watts"]

            *

            self.df["Seal_Loss_Score"]

        )

        print("HVAC_Stress_Index created.")

    def save_dataset(self):

        output_path = "data/processed/X_features_engineered.csv"

        self.df.to_csv(

            output_path,

            index=False

        )

        print(f"\nEngineered dataset saved to {output_path}")

    def engineer_features(self):

        self.create_temperature_deviation()

        self.create_cooling_load()

        self.create_hvac_efficiency()

        self.create_seal_loss_score()

        self.create_hvac_stress_index()

        self.save_dataset()

        return self.df