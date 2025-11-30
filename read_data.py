import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np


load_dotenv()

HC_left_path = os.getenv("HC_left_data_path")
HC_right_path = os.getenv("HC_right_data_path")
patient_left_path = os.getenv("patient_left_data_path")
patient_right_path = os.getenv("patient_right_data_path")

zero_start_time = 0
light_on_time = 3
LOR_early_start_time = 6
LOR_early_end_time = 8
LOR_late_start_time = 8
LOR_late_end_time = 13

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

##################################################################################
# Arousal gradient
patient_left_arousal_interval_data = {sheet_name: patient_left_numeric_data[sheet_name][(patient_left_numeric_data[sheet_name].index >= zero_start_time) & (patient_left_numeric_data[sheet_name].index <= light_on_time)] for sheet_name in patient_left_data.keys()}
patient_right_arousal_interval_data = {sheet_name: patient_right_numeric_data[sheet_name][(patient_right_numeric_data[sheet_name].index >= zero_start_time) & (patient_right_numeric_data[sheet_name].index <= light_on_time)] for sheet_name in patient_right_data.keys()}

patient_left_arousal_timespan = {sheet_name: patient_left_arousal_interval_data[sheet_name].index[-1] - patient_left_arousal_interval_data[sheet_name].index[0] for sheet_name in patient_left_data.keys()} # Simply compute the timespan as the diffrence between the last and first value for the selected data
patient_right_arousal_timespan = {sheet_name: patient_right_arousal_interval_data[sheet_name].index[-1] - patient_right_arousal_interval_data[sheet_name].index[0] for sheet_name in patient_right_data.keys()}
patient_left_arousal_movement = {sheet_name: (patient_left_arousal_interval_data[sheet_name].iloc[0] - patient_left_arousal_interval_data[sheet_name].max()).clip(lower=0) for sheet_name in patient_left_data.keys()} # Compute the size difference betweeen the last and first value
patient_right_arousal_movement = {sheet_name: (patient_right_arousal_interval_data[sheet_name].iloc[0] - patient_right_arousal_interval_data[sheet_name].max()).clip(lower=0) for sheet_name in patient_right_data.keys()}
patient_left_arousal_gradient = {sheet_name: patient_left_arousal_movement[sheet_name] / patient_left_arousal_timespan[sheet_name] for sheet_name in patient_left_data.keys()}
patient_right_arousal_gradient = {sheet_name: patient_right_arousal_movement[sheet_name] / patient_right_arousal_timespan[sheet_name] for sheet_name in patient_right_data.keys()}

##################################################################################
# Max PLR:
patient_left_PLR_interval_data = {sheet_name: patient_left_numeric_data[sheet_name][(patient_left_numeric_data[sheet_name].index >= light_on_time) & (patient_left_numeric_data[sheet_name].index <= LOR_early_start_time)] for sheet_name in patient_left_data.keys()}
patient_right_PLR_interval_data = {sheet_name: patient_right_numeric_data[sheet_name][(patient_right_numeric_data[sheet_name].index >= light_on_time) & (patient_right_numeric_data[sheet_name].index <= LOR_early_start_time)] for sheet_name in patient_right_data.keys()}

patient_left_max_PLR = {sheet_name: patient_left_PLR_interval_data[sheet_name].iloc[0] - patient_left_PLR_interval_data[sheet_name].min() for sheet_name in patient_left_data.keys()} 
patient_right_max_PLR = {sheet_name: patient_right_PLR_interval_data[sheet_name].iloc[0] - patient_right_PLR_interval_data[sheet_name].min() for sheet_name in patient_right_data.keys()}

##################################################################################
# Find 50 % numeric value
patient_left_highest_values = {sheet_name: patient_left_numeric_data[sheet_name].iloc[:patient_left_closest_timestamp_LOR_early_start_time[sheet_name]].max() for sheet_name in patient_left_data.keys()} # Find the highest value
patient_left_50_percent_values = {sheet_name: patient_left_highest_values[sheet_name] * 0.5 for sheet_name in patient_left_data.keys()} # Compute half the max value

patient_right_highest_values = {sheet_name: patient_right_numeric_data[sheet_name].iloc[:patient_right_closest_timestamp_LOR_early_start_time[sheet_name]].max() for sheet_name in patient_right_data.keys()} # Find the highest value
patient_right_50_percent_values = {sheet_name: patient_right_highest_values[sheet_name] * 0.5 for sheet_name in patient_right_data.keys()} # Compute half the max value

# Find timestamp closest to numeric value
patient_left_closest_timestamp_PLR = {sheet_name: (patient_left_PLR_interval_data[sheet_name] - patient_left_50_percent_values[sheet_name]).abs().idxmin() for sheet_name in patient_left_data.keys()}
patient_right_closest_timestamp_PLR = {sheet_name: (patient_right_PLR_interval_data[sheet_name] - patient_right_50_percent_values[sheet_name]).abs().idxmin() for sheet_name in patient_right_data.keys()}

# Define the LOR interval as the next 50 % expand value must be in this interval
patient_left_LOR_interval_data = {sheet_name: patient_left_numeric_data[sheet_name][(patient_left_numeric_data[sheet_name].index >= LOR_early_start_time) & (patient_left_numeric_data[sheet_name].index <= LOR_late_end_time)] for sheet_name in patient_left_data.keys()}
patient_right_LOR_interval_data = {sheet_name: patient_right_numeric_data[sheet_name][(patient_right_numeric_data[sheet_name].index >= LOR_early_start_time) & (patient_right_numeric_data[sheet_name].index <= LOR_late_end_time)] for sheet_name in patient_right_data.keys()}

patient_left_closest_timestamp_LOR = {sheet_name: (patient_left_LOR_interval_data[sheet_name] - patient_left_50_percent_values[sheet_name]).abs().idxmin() for sheet_name in patient_left_data.keys()}
patient_right_closest_timestamp_LOR = {sheet_name: (patient_right_LOR_interval_data[sheet_name] - patient_right_50_percent_values[sheet_name]).abs().idxmin() for sheet_name in patient_right_data.keys()}

##################################################################################
# LOR Early Gradient:
patient_left_LOR_early_interval_data = {sheet_name: patient_left_numeric_data[sheet_name][(patient_left_numeric_data[sheet_name].index >= LOR_early_start_time) & (patient_left_numeric_data[sheet_name].index <= LOR_early_end_time)] for sheet_name in patient_left_data.keys()}
patient_right_LOR_early_interval_data = {sheet_name: patient_right_numeric_data[sheet_name][(patient_right_numeric_data[sheet_name].index >= LOR_early_start_time) & (patient_right_numeric_data[sheet_name].index <= LOR_early_end_time)] for sheet_name in patient_right_data.keys()}

# Find the increase during LOR early for left and right respectively
patient_left_increase_during_LOR_early = {
    sheet_name: (patient_left_LOR_early_interval_data[sheet_name].iloc[0] - patient_left_LOR_early_interval_data[sheet_name].max()) for sheet_name in patient_left_data.keys()}

patient_right_increase_during_LOR_early = {
    sheet_name: (patient_right_LOR_early_interval_data[sheet_name].iloc[0] - patient_right_LOR_early_interval_data[sheet_name].max()) for sheet_name in patient_right_data.keys()}

# Find timespans for left and right respectively
patient_left_timespan_during_LOR_early = {
    sheet_name: patient_left_LOR_early_interval_data[sheet_name].index[-1] - patient_left_LOR_early_interval_data[sheet_name].index[0] for sheet_name in patient_left_data.keys()
}
patient_right_timespan_during_LOR_early = {
    sheet_name: patient_right_LOR_early_interval_data[sheet_name].index[-1] - patient_right_LOR_early_interval_data[sheet_name].index[0] for sheet_name in patient_right_data.keys()
}

# Compute gradient
patient_left_LOR_early_gradient = {sheet_name: patient_left_increase_during_LOR_early[sheet_name] / patient_left_timespan_during_LOR_early[sheet_name] for sheet_name in patient_left_data.keys()}
patient_right_LOR_early_gradient = {sheet_name: patient_right_increase_during_LOR_early[sheet_name] / patient_right_timespan_during_LOR_early[sheet_name] for sheet_name in patient_right_data.keys()}

##################################################################################
# LOR Late Gradient:
patient_left_LOR_late_interval_data = {sheet_name: patient_left_numeric_data[sheet_name][(patient_left_numeric_data[sheet_name].index >= LOR_late_start_time) & (patient_left_numeric_data[sheet_name].index <= LOR_late_end_time)] for sheet_name in patient_left_data.keys()}
patient_right_LOR_late_interval_data = {sheet_name: patient_right_numeric_data[sheet_name][(patient_right_numeric_data[sheet_name].index >= LOR_late_start_time) & (patient_right_numeric_data[sheet_name].index <= LOR_late_end_time)] for sheet_name in patient_right_data.keys()}

# Find the increase during LOR early for left and right respectively
patient_left_increase_during_LOR_late = {
    sheet_name: (patient_left_LOR_late_interval_data[sheet_name].iloc[0] - patient_left_LOR_late_interval_data[sheet_name].max()) for sheet_name in patient_left_data.keys()}

patient_right_increase_during_LOR_late = {
    sheet_name: (patient_right_LOR_late_interval_data[sheet_name].iloc[0] - patient_right_LOR_late_interval_data[sheet_name].max()) for sheet_name in patient_right_data.keys()}

# Find timespans for left and right respectively
patient_left_timespan_during_LOR_late = {
    sheet_name: patient_left_LOR_late_interval_data[sheet_name].index[-1] - patient_left_LOR_late_interval_data[sheet_name].index[0] for sheet_name in patient_left_data.keys()
}
patient_right_timespan_during_LOR_late = {
    sheet_name: patient_right_LOR_late_interval_data[sheet_name].index[-1] - patient_right_LOR_late_interval_data[sheet_name].index[0] for sheet_name in patient_right_data.keys()
}

# Compute gradient
patient_left_LOR_late_gradient = {sheet_name: patient_left_increase_during_LOR_late[sheet_name] / patient_left_timespan_during_LOR_late[sheet_name] for sheet_name in patient_left_data.keys()}
patient_right_LOR_late_gradient = {sheet_name: patient_right_increase_during_LOR_late[sheet_name] / patient_right_timespan_during_LOR_late[sheet_name] for sheet_name in patient_right_data.keys()}

##################################################################################
# Diagnostic metrics:
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

##################################################################################
# Find patients with pupil size constantly below 3.5:
patient_left_35 = {sheet_name: patient_left_numeric_data[sheet_name].max() < 3.5 for sheet_name in patient_left_data.keys()} # Find the highest value
patient_right_35 = {sheet_name: patient_right_numeric_data[sheet_name].max() < 3.5 for sheet_name in patient_right_data.keys()} # Find the highest value

##################################################################################
# Store data left:
patient_left_arousal_metrics = pd.concat([
    patient_left_arousal_movement[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_max_PLR_metrics = pd.concat([
    patient_left_max_PLR[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_first_50_metrics = pd.concat([
    patient_left_closest_timestamp_PLR[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_second_50_metrics = pd.concat([
    patient_left_closest_timestamp_LOR[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_LOR_early_gradient_metrics = pd.concat([
    patient_left_increase_during_LOR_early[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_LOR_late_gradient_metrics = pd.concat([
    patient_left_increase_during_LOR_late[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

patient_left_35_metrics = pd.concat([
    patient_left_35[sheet_name]
    for sheet_name in patient_left_data.keys()
], axis=1, keys=patient_left_data.keys())

subject_IDs = [[subjectID] * len(patient_left_LOR_late_gradient_metrics.loc[subjectID].dropna()) for subjectID in patient_left_LOR_late_gradient_metrics.index]
days = np.concatenate([np.arange(1, len(IDs)+1, 1) for IDs in subject_IDs]).flatten()
subject_IDs = np.concatenate(subject_IDs)
patient_left_GCS_scores = np.concatenate([patient_left_GCS_metrics.loc[index].dropna().values for index in patient_left_GCS_metrics.index]).astype(np.float64)
patient_left_FOUR_scores = np.concatenate([patient_left_FOUR_metrics.loc[index].dropna().values for index in patient_left_FOUR_metrics.index]).astype(np.float64)
patient_left_SECONDS_scores = np.concatenate([patient_left_SECONDS_metrics.loc[index].dropna().values for index in patient_left_SECONDS_metrics.index]).astype(np.float64)

patient_left_arousal_scores = np.concatenate([patient_left_arousal_metrics.loc[index].dropna().values for index in patient_left_arousal_metrics.index]).astype(np.float64)
patient_left_max_PLR_scores = np.concatenate([patient_left_max_PLR_metrics.loc[index].dropna().values for index in patient_left_max_PLR_metrics.index]).astype(np.float64)
patient_left_first_50_scores = np.concatenate([patient_left_first_50_metrics.loc[index].dropna().values for index in patient_left_first_50_metrics.index]).astype(np.float64)
patient_left_second_50_scores = np.concatenate([patient_left_second_50_metrics.loc[index].dropna().values for index in patient_left_second_50_metrics.index]).astype(np.float64)
patient_left_LOR_early_gradient_scores = np.concatenate([patient_left_LOR_early_gradient_metrics.loc[index].dropna().values for index in patient_left_LOR_early_gradient_metrics.index]).astype(np.float64)
patient_left_LOR_late_gradient_scores = np.concatenate([patient_left_LOR_late_gradient_metrics.loc[index].dropna().values for index in patient_left_LOR_late_gradient_metrics.index]).astype(np.float64)
patient_left_35_scores = np.concatenate([patient_left_35_metrics.loc[index].dropna().values for index in patient_left_35_metrics.index]).astype(np.float64)

data = {
    "Subject ID": subject_IDs,
    "Day": days,
    "GCS scores": patient_left_GCS_scores,
    "FOUR scores": patient_left_FOUR_scores,
    "SECONDS scores": patient_left_SECONDS_scores,
    "Arousal gradient": patient_left_arousal_scores,
    "Max PLR": patient_left_max_PLR_scores,
    "First 50% scores": patient_left_first_50_scores,
    "Second 50% scores": patient_left_second_50_scores,
    "LOR early gradient": patient_left_LOR_early_gradient_scores,
    "LOR late gradient": patient_left_LOR_late_gradient_scores,
    "Under 3.5 mm.": patient_left_35_scores,
}
left_data_original = pd.DataFrame(data)
left_data_original = left_data_original.sort_values(["Subject ID", "Day"])

##################################################################################
# Store data right:
patient_right_arousal_metrics = pd.concat([
    patient_right_arousal_movement[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_max_PLR_metrics = pd.concat([
    patient_right_max_PLR[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_first_50_metrics = pd.concat([
    patient_right_closest_timestamp_PLR[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_second_50_metrics = pd.concat([
    patient_right_closest_timestamp_LOR[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_LOR_early_gradient_metrics = pd.concat([
    patient_right_increase_during_LOR_early[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_LOR_late_gradient_metrics = pd.concat([
    patient_right_increase_during_LOR_late[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

patient_right_35_metrics = pd.concat([
    patient_right_35[sheet_name]
    for sheet_name in patient_right_data.keys()
], axis=1, keys=patient_right_data.keys())

from dates import dates_data_original
# Merge the two dataframes on both 'Subject ID' and 'Day'
left_data_with_dates = left_data_original.merge(
    dates_data_original[['Subject ID', 'Day', 'individuel_dates']], 
    on=['Subject ID', 'Day'], 
    how='left'
)