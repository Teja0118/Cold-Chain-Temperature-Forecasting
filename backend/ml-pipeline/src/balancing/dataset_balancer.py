import random
import pandas as pd

from .utils import EmergencyDataGenerator


class DatasetBalancer:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

        self.next_container_number = (
            self.df["Container_ID"]
            .str.extract(r"(\d+)")
            .astype(int)
            .max()
            .iloc[0]
        )

    def print_distribution(self):

        print("\nCurrent Class Distribution\n")

        print(
            self.df[
                "Logistics_Action_Recommendation"
            ].value_counts()
        )

    def generate_emergency_samples(
        self,
        num_samples=5000
    ):

        warning_rows = self.df[
            self.df[
                "Logistics_Action_Recommendation"
            ] == "WARNING_Request_HVAC_Remote_Reset"
        ]

        generated_rows = []

        for _ in range(num_samples):

            row = warning_rows.sample(
                n=1,
                replace=True
            ).iloc[0]

            new_row = EmergencyDataGenerator.generate_emergency_row(
                row
            )

            new_row["Container_ID"] = self.generate_container_id()

            generated_rows.append(new_row)

        emergency_df = pd.DataFrame(generated_rows)

        self.df = pd.concat(
            [self.df, emergency_df],
            ignore_index=True
        )

    def save_dataset(
        self,
        output_path
    ):

        self.df.to_csv(
            output_path,
            index=False
        )

        print("\nBalanced dataset saved.")

        print(output_path)

    def balance(self):

        self.print_distribution()

        self.generate_warning_samples()

        self.generate_emergency_samples()

        print("\nUpdated Class Distribution\n")

        print(
            self.df[
                "Logistics_Action_Recommendation"
            ].value_counts()
        )

        return self.df
    
    def generate_warning_samples(self, target_count=5000):

        warning_rows = self.df[
            self.df["Logistics_Action_Recommendation"]
            == "WARNING_Request_HVAC_Remote_Reset"
        ]

        current_count = len(warning_rows)

        samples_needed = target_count - current_count

        if samples_needed <= 0:
            return

        generated_rows = []

        for _ in range(samples_needed):

            row = warning_rows.sample(
                n=1,
                replace=True
            ).iloc[0]

            new_row = EmergencyDataGenerator.generate_warning_row(
                row
            )

            new_row["Container_ID"] = self.generate_container_id()

            generated_rows.append(new_row)

        warning_df = pd.DataFrame(generated_rows)

        self.df = pd.concat(
            [self.df, warning_df],
            ignore_index=True
        )
    
    def generate_container_id(self):

        self.next_container_number += 1

        return f"COLD_CNT_{self.next_container_number:06d}"
    
