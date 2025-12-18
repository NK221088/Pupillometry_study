import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

HC_left_path = os.getenv("dates_data_path")
dates = pd.read_csv(HC_left_path)

individuel_dates = np.array([])
followup_numbers = np.array([])
id_columns = np.array([])

for id in np.unique(dates["record_id"]):
    if id == 72:
        print("debug")
    ind_data = dates[dates['record_id'] == id][:int(dates[dates['record_id'] == id]["redcap_repeat_instance"].iloc[-1])+1] if not np.isnan(dates[dates['record_id'] == id]["redcap_repeat_instance"].iloc[-1]) else dates[dates['record_id'] == id][:1]
    ind_dates = np.concatenate([ind_data["date_examination"].iloc[:1].values, ind_data["date_examination_v2"].iloc[1:].values])
    followup_number = np.concatenate([np.array([0]), ind_data["redcap_repeat_instance"][1:].values.astype(int)])
    id_column = ind_data["record_id"].values
    
    individuel_dates = np.concatenate([individuel_dates, ind_dates])
    followup_numbers = np.concatenate([followup_numbers, followup_number + 1])
    id_columns = np.concatenate([id_columns, id_column])
    

dates_data = {
    "Subject ID": id_columns,
    "Day": followup_numbers,
    "individuel_dates": individuel_dates,
}
dates_data_original = pd.DataFrame(dates_data)