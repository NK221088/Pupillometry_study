import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv()

HC_left_path = os.getenv("HC_left_data_path")
HC_right_path = os.getenv("HC_right_data_path")
patient_left_path =os.getenv("patient_left_data_path")
patient_right_path = os.getenv("patient_right_data_path")

HC_left_data = pd.read_excel(HC_left_path, index_col=0, sheet_name=None)


HC_right_data = pd.read_excel(HC_right_path, index_col=0, sheet_name=None)


patient_left_data = pd.read_excel(patient_left_path, index_col=0, sheet_name=None)
patient_left_text_data = {sheet_name: patient_left_data[sheet_name].iloc[:8] for sheet_name in patient_left_data.keys()}
patient_left_numeric_data = {sheet_name: patient_left_data[sheet_name].iloc[8:] for sheet_name in patient_left_data.keys()}

patient_right_data = pd.read_excel(patient_right_path, index_col=0, sheet_name=None)
patient_right_text_data = {sheet_name: patient_right_data[sheet_name].iloc[:8] for sheet_name in patient_right_data.keys()}
patient_right_numeric_data = {sheet_name: patient_right_data[sheet_name].iloc[8:] for sheet_name in patient_right_data.keys()}

zero_start_time = 0
LOR_early_start_time = 6
LOR_early_end_time = 8
LOR_late_start_time = 8
LOE_late_end_time = 13

patient_left_PLR_interval_data = {sheet_name: patient_left_numeric_data[sheet_name][(patient_left_numeric_data[sheet_name].index >= zero_start_time) & (patient_left_numeric_data[sheet_name].index <= LOR_early_start_time)] for sheet_name in patient_left_data.keys()}
patient_right_PLR_interval_data = {sheet_name: patient_right_numeric_data[sheet_name][(patient_right_numeric_data[sheet_name].index >= zero_start_time) & (patient_right_numeric_data[sheet_name].index <= LOR_early_start_time)] for sheet_name in patient_right_data.keys()}

patient_left_LOR_interval_data = {sheet_name: patient_left_numeric_data[sheet_name][(patient_left_numeric_data[sheet_name].index >= LOR_early_start_time) & (patient_left_numeric_data[sheet_name].index <= LOE_late_end_time)] for sheet_name in patient_left_data.keys()}
patient_right_LOR_interval_data = {sheet_name: patient_right_numeric_data[sheet_name][(patient_right_numeric_data[sheet_name].index >= LOR_early_start_time) & (patient_right_numeric_data[sheet_name].index <= LOE_late_end_time)] for sheet_name in patient_right_data.keys()}

# Find 50 % numeric value
patient_left_highest_values = {sheet_name: patient_left_numeric_data[sheet_name].max() for sheet_name in patient_left_data.keys()}
patient_left_50_percent_values = {sheet_name: patient_left_highest_values[sheet_name] * 0.5 for sheet_name in patient_left_data.keys()}

patient_right_highest_values = {sheet_name: patient_right_numeric_data[sheet_name].max() for sheet_name in patient_right_data.keys()}
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

patient_left_GCS_metrics = df = pd.concat([
    patient_left_text_data[sheet_name].loc["GCS"] 
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_FOUR_metrics = df = pd.concat([
    patient_left_text_data[sheet_name].loc["FOUR"] 
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_SECONDS_metrics = df = pd.concat([
    patient_left_text_data[sheet_name].loc["SECONDS"] 
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_right_GCS_metrics = df = pd.concat([
    patient_right_text_data[sheet_name].loc["GCS"] 
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_FOUR_metrics = df = pd.concat([
    patient_right_text_data[sheet_name].loc["FOUR"] 
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_SECONDS_metrics = df = pd.concat([
    patient_right_text_data[sheet_name].loc["SECONDS"] 
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_left_first_50_metrics = df = pd.concat([
    patient_left_closest_timestamp_PLR[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_second_50_metrics = df = pd.concat([
    patient_left_closest_timestamp_LOR[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_LOR_late_gradient_metrics = df = pd.concat([
    patient_left_LOR_gradient[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_right_first_50_metrics = df = pd.concat([
    patient_right_closest_timestamp_PLR[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_second_50_metrics = df = pd.concat([
    patient_right_closest_timestamp_LOR[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_LOR_late_gradient_metrics = df = pd.concat([
    patient_right_LOR_gradient[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())