import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

NPI_data_path = os.getenv("NPI_data_path")
NPI_data = pd.read_excel(NPI_data_path)
merging_columns = [
    "date_examination",
    "light_off_performed",
    "npi_left",
    "npi_right",
]
for column in merging_columns:
    if column == "date_examination":
        NPI_data[f"{column}_merged"] = (
            NPI_data[f"{column}"]
            .combine_first(NPI_data[f"date_of_examination_2"])
        )
    else:
        NPI_data[f"{column}_merged"] = (
            NPI_data[f"{column}"]
            .combine_first(NPI_data[f"{column}_2"])
        )

columns_to_keep = ["record_id", "redcap_repeat_instance"] + [f"{col}_merged" for col in merging_columns]
NPI_data_cleaned = NPI_data[columns_to_keep] # Remove suffix from light_off_performed_merged column
NPI_data_cleaned["light_off_performed_merged"] = (
    NPI_data_cleaned["light_off_performed_merged"]
    .str.replace(r"\s*\(.*\)$", "", regex=True)
)
NPI_data_cleaned["redcap_repeat_instance"] = ( # Ensure first recording is marked as 0
    NPI_data_cleaned["redcap_repeat_instance"]
    .fillna(0)
)
mask = NPI_data_cleaned["record_id"] == 213
cols = ["npi_left_merged"]

NPI_data_cleaned.loc[mask, cols] = (
    NPI_data_cleaned.loc[mask, cols].replace(0, np.nan)
)
NPI_data_cleaned["redcap_repeat_instance"]  += 1 # Shift follow-up numbers to start from 1
NPI_data_cleaned = NPI_data_cleaned[
    NPI_data_cleaned["light_off_performed_merged"] == "Yes"
]
NPI_data_cleaned = (
    NPI_data_cleaned
    .sort_values(
        by=["record_id", "redcap_repeat_instance"],
        ascending=[True, True]
    )
)
NPI_data_cleaned["redcap_repeat_instance"] = (
    NPI_data_cleaned
    .groupby("record_id")
    .cumcount()
    .add(1)
)