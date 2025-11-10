import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np


load_dotenv()

HC_left_path = os.getenv("HC_left_data_path")
HC_right_path = os.getenv("HC_right_data_path")
patient_left_path =os.getenv("patient_left_data_path")
patient_right_path = os.getenv("patient_right_data_path")

zero_start_time = 0
light_on_time = 3
LOR_early_start_time = 6
LOR_early_end_time = 8
LOR_late_start_time = 8
LOE_late_end_time = 13

HC_left_data = pd.read_excel(HC_left_path, index_col=0, sheet_name=None)
HC_right_data = pd.read_excel(HC_right_path, index_col=0, sheet_name=None)

patient_left_data = pd.read_excel(patient_left_path, index_col=0, sheet_name=None)
patient_left_zero_indices = {sheet_name: np.where(patient_left_data[sheet_name].index == 0)[0][0] for sheet_name in patient_left_data.keys()} # Finding the first time index
patient_left_closest_timestamp_LOR_early_start_time = {sheet_name: np.argmin(np.abs((np.array([idx for idx in patient_left_data[sheet_name].index if isinstance(idx, (int, float))]) - LOR_early_start_time))) for sheet_name in patient_left_data.keys()} # Find the time index closest to LOR early start time
patient_left_text_data = {sheet_name: patient_left_data[sheet_name].iloc[:patient_left_zero_indices[sheet_name]] for sheet_name in patient_left_data.keys()} # Extract text data -> Everything before first time index
patient_left_numeric_data = {sheet_name: patient_left_data[sheet_name].iloc[patient_left_zero_indices[sheet_name]:].apply(pd.to_numeric, errors='coerce') for sheet_name in patient_left_data.keys()}

patient_right_data = pd.read_excel(patient_right_path, index_col=0, sheet_name=None)
patient_right_zero_indices = {sheet_name: np.where(patient_right_data[sheet_name].index == 0)[0][0] for sheet_name in patient_right_data.keys()} # Finding the first time index
patient_right_closest_timestamp_LOR_early_start_time = {sheet_name: np.argmin(np.abs((np.array([idx for idx in patient_right_data[sheet_name].index if isinstance(idx, (int, float))]) - LOR_early_start_time))) for sheet_name in patient_right_data.keys()} # Find the time index closest to LOR early start time
patient_right_text_data = {sheet_name: patient_right_data[sheet_name].iloc[:patient_right_zero_indices[sheet_name]] for sheet_name in patient_right_data.keys()} # Extract text data -> Everything before first time index
patient_right_numeric_data = {sheet_name: patient_right_data[sheet_name].iloc[patient_right_zero_indices[sheet_name]:].apply(pd.to_numeric, errors='coerce') for sheet_name in patient_right_data.keys()}

patient_left_PLR_interval_data = {sheet_name: patient_left_numeric_data[sheet_name][(patient_left_numeric_data[sheet_name].index >= light_on_time) & (patient_left_numeric_data[sheet_name].index <= LOR_early_start_time)] for sheet_name in patient_left_data.keys()}
patient_right_PLR_interval_data = {sheet_name: patient_right_numeric_data[sheet_name][(patient_right_numeric_data[sheet_name].index >= light_on_time) & (patient_right_numeric_data[sheet_name].index <= LOR_early_start_time)] for sheet_name in patient_right_data.keys()}

patient_left_LOR_interval_data = {sheet_name: patient_left_numeric_data[sheet_name][(patient_left_numeric_data[sheet_name].index >= LOR_early_start_time) & (patient_left_numeric_data[sheet_name].index <= LOE_late_end_time)] for sheet_name in patient_left_data.keys()}
patient_right_LOR_interval_data = {sheet_name: patient_right_numeric_data[sheet_name][(patient_right_numeric_data[sheet_name].index >= LOR_early_start_time) & (patient_right_numeric_data[sheet_name].index <= LOE_late_end_time)] for sheet_name in patient_right_data.keys()}

# Find 50 % numeric value
patient_left_highest_values = {sheet_name: patient_left_numeric_data[sheet_name].iloc[:patient_left_closest_timestamp_LOR_early_start_time[sheet_name]].max() for sheet_name in patient_left_data.keys()}
patient_left_50_percent_values = {sheet_name: patient_left_highest_values[sheet_name] * 0.5 for sheet_name in patient_left_data.keys()}

patient_right_highest_values = {sheet_name: patient_right_numeric_data[sheet_name].iloc[:patient_right_closest_timestamp_LOR_early_start_time[sheet_name]].max() for sheet_name in patient_right_data.keys()}
patient_right_50_percent_values = {sheet_name: patient_right_highest_values[sheet_name] * 0.5 for sheet_name in patient_right_data.keys()}

# Find timestamp closest to numeric value
patient_left_closest_timestamp_PLR = {sheet_name: (patient_left_PLR_interval_data[sheet_name] - patient_left_50_percent_values[sheet_name]).abs().idxmin() for sheet_name in patient_left_data.keys()}
patient_right_closest_timestamp_PLR = {sheet_name: (patient_right_PLR_interval_data[sheet_name] - patient_right_50_percent_values[sheet_name]).abs().idxmin() for sheet_name in patient_right_data.keys()}
patient_left_closest_timestamp_LOR = {sheet_name: (patient_left_LOR_interval_data[sheet_name] - patient_left_50_percent_values[sheet_name]).abs().idxmin() for sheet_name in patient_left_data.keys()}
patient_right_closest_timestamp_LOR = {sheet_name: (patient_right_LOR_interval_data[sheet_name] - patient_right_50_percent_values[sheet_name]).abs().idxmin() for sheet_name in patient_right_data.keys()}

patient_left_increase_during_LOR_late = {
    sheet_name: (
        patient_left_LOR_interval_data[sheet_name].iloc[-1] - patient_left_LOR_interval_data[sheet_name].iloc[0]
        if not patient_left_LOR_interval_data[sheet_name].iloc[-1].isna().all()
        else patient_left_LOR_interval_data[sheet_name].iloc[-2] - patient_left_LOR_interval_data[sheet_name].iloc[0]
    )
    for sheet_name in patient_left_data.keys()
}
patient_right_increase_during_LOR_late = {
    sheet_name: (
        patient_right_LOR_interval_data[sheet_name].iloc[-1] - patient_right_LOR_interval_data[sheet_name].iloc[0]
        if not patient_right_LOR_interval_data[sheet_name].iloc[-1].isna().all()
        else patient_right_LOR_interval_data[sheet_name].iloc[-2] - patient_right_LOR_interval_data[sheet_name].iloc[0]
    )
    for sheet_name in patient_right_data.keys()
}

patient_left_timespan_during_LOR_late = {
    sheet_name: (
        patient_left_LOR_interval_data[sheet_name].index[-1] - (patient_left_LOR_interval_data[sheet_name].index.to_series() - LOR_late_start_time).abs().idxmin()
        if not patient_left_LOR_interval_data[sheet_name].iloc[-1].isna().all()
        else patient_left_LOR_interval_data[sheet_name].index[-2] - (patient_left_LOR_interval_data[sheet_name].index.to_series() - LOR_late_start_time).abs().idxmin()
    )
    for sheet_name in patient_left_data.keys()
}
patient_right_timespan_during_LOR_late = {
    sheet_name: (
        patient_right_LOR_interval_data[sheet_name].index[-1] - (patient_right_LOR_interval_data[sheet_name].index.to_series() - LOR_late_start_time).abs().idxmin()
        if not patient_right_LOR_interval_data[sheet_name].iloc[-1].isna().all()
        else patient_right_LOR_interval_data[sheet_name].index[-2] - (patient_right_LOR_interval_data[sheet_name].index.to_series() - LOR_late_start_time).abs().idxmin()
    )
    for sheet_name in patient_right_data.keys()
}

patient_left_LOR_gradient = {sheet_name: patient_left_increase_during_LOR_late[sheet_name] / patient_left_timespan_during_LOR_late[sheet_name] for sheet_name in patient_left_data.keys()}
patient_right_LOR_gradient = {sheet_name: patient_right_increase_during_LOR_late[sheet_name] / patient_right_timespan_during_LOR_late[sheet_name] for sheet_name in patient_right_data.keys()}

patient_left_GCS_metrics = pd.concat([
    patient_left_text_data[sheet_name].loc["GCS"] 
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_FOUR_metrics = pd.concat([
    patient_left_text_data[sheet_name].loc["FOUR"] 
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_SECONDS_metrics = pd.concat([
    patient_left_text_data[sheet_name].loc["SECONDS"] 
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())
SECONDS_conversion_dict = {
"C": 0,
"U": 1,
"M-": 2,
"M+": 3,
"E": 4
}
patient_left_SECONDS_metrics = patient_left_SECONDS_metrics.replace(SECONDS_conversion_dict)

patient_right_GCS_metrics = pd.concat([
    patient_right_text_data[sheet_name].loc["GCS"] 
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_FOUR_metrics = pd.concat([
    patient_right_text_data[sheet_name].loc["FOUR"] 
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_SECONDS_metrics = pd.concat([
    patient_right_text_data[sheet_name].loc["SECONDS"] 
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())
patient_right_SECONDS_metrics = patient_right_SECONDS_metrics.replace(SECONDS_conversion_dict)

patient_left_first_50_metrics = pd.concat([
    patient_left_closest_timestamp_PLR[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_second_50_metrics = pd.concat([
    patient_left_closest_timestamp_LOR[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_LOR_late_gradient_metrics = pd.concat([
    patient_left_LOR_gradient[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_right_first_50_metrics = pd.concat([
    patient_right_closest_timestamp_PLR[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_second_50_metrics = pd.concat([
    patient_right_closest_timestamp_LOR[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_LOR_late_gradient_metrics = pd.concat([
    patient_right_LOR_gradient[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())