import sys
import os

# Add the project root (one folder up) to Python's module search path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from NPi.final_plots import patient_left_individual_raw_data, patient_left_individual_text_data
import numpy as np

zero_start_time = 0
light_on_time = 3
LOR_early_start_time = 6
LOR_early_end_time = 8
LOR_late_start_time = 8
LOR_late_end_time = 13

time_stamps = patient_left_individual_raw_data[1].index.astype(float)
patient_left_closest_timestamp_LOR_early_start_time = (
    patient_left_individual_raw_data[1]
    .index
    .to_series()
    .sub(LOR_early_start_time)
    .abs()
    .idxmin()
)


##################################################################################
# Arousal gradient

def extract_arousal_gradients(data_dict, zero_start_time, light_on_time):
    arousal_gradients = {}

    for key, df in data_dict.items():
        interval = df.loc[
            (df.index >= zero_start_time) &
            (df.index <= light_on_time)
        ]

        timespan = interval.index[-1] - interval.index[0]
        movement = (interval.max() - interval.iloc[0]).clip(lower=0)

        arousal_gradients[key] = movement / timespan

    return arousal_gradients


##################################################################################
# Max PLR:

def extract_max_PLR(data_dict, light_on_time, LOR_early_start_time):
    max_PLRs = {}

    for key, df in data_dict.items():
        interval = df.loc[
            (df.index >= light_on_time) &
            (df.index <= LOR_early_start_time)
        ]

        baseline = interval.iloc[0]
        min_value = interval.min()

        max_PLRs[key] = baseline - min_value

    return max_PLRs

##################################################################################
# Find 50 % numeric value

def extract_50_percent_timestamps(
    data_dict,
    patient_left_closest_timestamp_LOR_early_start_time,
    light_on_time,
    LOR_early_start_time,
    LOR_late_end_time
):
    _50_per_PLR_times = {}
    _50_per_LOR_times = {}

    for key, df in data_dict.items():
        interval = df.loc[:patient_left_closest_timestamp_LOR_early_start_time]

        PLR_interval = df.loc[
            (df.index >= light_on_time) &
            (df.index <= LOR_early_start_time)
        ]

        LOR_interval = df.loc[
            (df.index >= LOR_early_start_time) &
            (df.index <= LOR_late_end_time)
        ]

        max_value = interval.max()
        _50_value = max_value * 0.5

        _50_per_PLR_times[key] = (
            (PLR_interval - _50_value).abs().idxmin()
        )

        _50_per_LOR_times[key] = (
            (LOR_interval - _50_value).abs().idxmin()
        )

    return _50_per_PLR_times, _50_per_LOR_times
