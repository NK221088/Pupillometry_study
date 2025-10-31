import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

HC_left_path = os.getenv("HC_left_data_path")
HC_right_path = os.getenv("HC_right_data_path")
patient_left_path =os.getenv("patient_left_data_path")
patient_right_path = os.getenv("patient_right_data_path")

HC_left_data = pd.read_excel(HC_left_path, index_col=0)


HC_right_data = pd.read_excel(HC_right_path, index_col=0)


patient_left_data = pd.read_excel(patient_left_path, index_col=0)
patient_left_text_data = patient_left_data.iloc[:8]
patient_left_numeric_data = patient_left_data[8:]

patient_right_data = pd.read_excel(patient_right_path, index_col=0)
patient_right_text_data = patient_right_data.iloc[:8]
patient_right_numeric_data = patient_right_data[8:]

zero_start_time = 0
LOR_early_start_time = 6
LOR_early_end_time = 8
LOR_late_start_time = 8
LOE_late_end_time = 13

patient_left_PLR_interval_data = patient_left_numeric_data[(patient_left_numeric_data.index >= zero_start_time) & (patient_left_numeric_data.index <= LOR_early_start_time)]
patient_right_PLR_interval_data = patient_right_numeric_data[(patient_right_numeric_data.index >= zero_start_time) & (patient_right_numeric_data.index <= LOR_early_start_time)]

patient_left_LOR_interval_data = patient_left_numeric_data[(patient_left_numeric_data.index >= LOR_early_start_time) & (patient_left_numeric_data.index <= LOE_late_end_time)]
patient_right_LOR_interval_data = patient_right_numeric_data[(patient_right_numeric_data.index >= LOR_early_start_time) & (patient_right_numeric_data.index <= LOE_late_end_time)]

# Find 50 % numeric value
patient_left_highest_values = patient_left_numeric_data.max()
patient_left_50_percent_values = patient_left_highest_values * 0.5

patient_right_highest_values = patient_right_numeric_data.max()
patient_right_50_percent_values = patient_right_highest_values * 0.5

# Find timestamp closest to numeric value
patient_left_closest_timestamp_PLR = (patient_left_PLR_interval_data - patient_left_50_percent_values).abs().idxmin()
patient_right_closest_timestamp_PLR = (patient_right_PLR_interval_data - patient_left_50_percent_values).abs().idxmin()
patient_left_closest_timestamp_POR = (patient_left_LOR_interval_data - patient_right_50_percent_values).abs().idxmin()
patient_right_closest_timestamp_POR = (patient_right_LOR_interval_data - patient_right_50_percent_values).abs().idxmin()

patient_left_increase_during_LOR_late = patient_left_LOR_interval_data.iloc[-1] - patient_left_LOR_interval_data.iloc[0]
patient_right_increase_during_LOR_late = patient_right_LOR_interval_data.iloc[-1] - patient_right_LOR_interval_data.iloc[0]

patient_left_timespan_during_LOR_late = patient_left_LOR_interval_data.index[-1] - (patient_left_LOR_interval_data.index.to_series() - LOR_late_start_time).abs().idxmin()
patient_right_timespan_during_LOR_late = patient_right_LOR_interval_data.index[-1] - (patient_right_LOR_interval_data.index.to_series() - LOR_late_start_time).abs().idxmin()

patient_left_LOR_gradient = patient_left_increase_during_LOR_late / patient_left_timespan_during_LOR_late
patient_right_LOR_gradient = patient_right_increase_during_LOR_late / patient_right_timespan_during_LOR_late

patient_left_consciousness_metrics = pd.DataFrame({
    'GCS': patient_left_text_data.loc["GCS"],
    'FOUR': patient_left_text_data.loc["FOUR"],
    'SECONDS': patient_left_text_data.loc["SECONDS"]
    })

patient_right_consciousness_metrics = pd.DataFrame({
    'GCS': patient_right_text_data.loc["GCS"],
    'FOUR': patient_right_text_data.loc["FOUR"],
    'SECONDS': patient_right_text_data.loc["SECONDS"]
    })

patient_left_pupil_metrics = pd.DataFrame({
    'First 50%': patient_left_closest_timestamp_PLR,
    'Second 50%': patient_left_closest_timestamp_POR,
    'LOR Late Gradient': patient_left_LOR_gradient
    })

patient_right_pupil_metrics = pd.DataFrame({
    'First 50%': patient_right_closest_timestamp_PLR,
    'Second 50%': patient_right_closest_timestamp_POR,
    'LOR Late Gradient': patient_right_LOR_gradient
    })

plt.scatter(patient_left_pupil_metrics.iloc[1].values, patient_left_consciousness_metrics.iloc[1].values[:2])

print("test")

