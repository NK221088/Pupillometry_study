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

# dates_path = os.getenv("dates_data_path")
# dates = pd.read_csv(dates_path)

# individuel_dates = np.array([])
# followup_numbers = np.array([])
# id_columns = np.array([])

# for id in np.unique(dates["record_id"]):
#     if id == 72:
#         print("debug")
#     ind_data = dates[dates['record_id'] == id][:int(dates[dates['record_id'] == id]["redcap_repeat_instance"].iloc[-1])+1] if not np.isnan(dates[dates['record_id'] == id]["redcap_repeat_instance"].iloc[-1]) else dates[dates['record_id'] == id][:1]
#     ind_dates = np.concatenate([ind_data["date_examination"].iloc[:1].values, ind_data["date_examination_v2"].iloc[1:].values])
#     followup_number = np.concatenate([np.array([0]), ind_data["redcap_repeat_instance"][1:].values.astype(int)])
#     id_column = ind_data["record_id"].values
    
#     individuel_dates = np.concatenate([individuel_dates, ind_dates])
#     followup_numbers = np.concatenate([followup_numbers, followup_number + 1])
#     id_columns = np.concatenate([id_columns, id_column])
    

# dates_data = {
#     "Subject ID": id_columns,
#     "Day": followup_numbers,
#     "individuel_dates": individuel_dates,
# }
# dates_data_original = pd.DataFrame(dates_data)