import sys
import os

# Add the project root (one folder up) to Python's module search path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Data_loading.read_data import patient_left_individual_raw_data, patient_left_individual_text_data

from Data_loading.read_NPi_data import NPI_data_cleaned
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
day_save_path = os.getenv("save_path_day_data")
save_path_extracted_features = os.getenv("save_path_extracted_features")

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
        movement = (interval.max() - interval.iloc[0])

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
        interval = df.loc[:patient_left_closest_timestamp_LOR_early_start_time].dropna(axis=1)

        PLR_interval = df.loc[
            (df.index >= light_on_time) &
            (df.index <= LOR_early_start_time)
        ].dropna(axis=1)

        LOR_interval = df.loc[
            (df.index >= LOR_early_start_time) &
            (df.index <= LOR_late_end_time)
        ].dropna(axis=1)

        max_value = interval.max()
        _50_value = max_value * 0.5

        _50_per_PLR_times[key] = (
            (PLR_interval - _50_value).abs().idxmin()
        )

        _50_per_LOR_times[key] = (
            (LOR_interval - _50_value).abs().idxmin()
        )

    return _50_per_PLR_times, _50_per_LOR_times

##################################################################################
# LOR Early Gradient:

def extract_LOR_early_gradients(
    data_dict,
    LOR_early_start_time,
    LOR_early_end_time,
    ):
    LOR_early_gradients = {}

    for key, df in data_dict.items():
        interval = df.loc[
            (df.index >= LOR_early_start_time) &
            (df.index <= LOR_early_end_time)
        ]

        timespan = interval.index[-1] - interval.index[0]
        movement = (interval.max() - interval.iloc[0])

        LOR_early_gradients[key] = movement / timespan

    return LOR_early_gradients

##################################################################################
# LOR late Gradient:

def extract_LOR_late_gradients(
    data_dict,
    LOR_late_start_time,
    LOR_late_end_time,
    ):
    LOR_late_gradients = {}

    for key, df in data_dict.items():
        interval = df.loc[
            (df.index >= LOR_late_start_time) &
            (df.index <= LOR_late_end_time)
        ]

        timespan = interval.index[-1] - interval.index[0]
        movement = (interval.max() - interval.iloc[0])

        LOR_late_gradients[key] = movement / timespan

    return LOR_late_gradients

left_arousal_gradients = extract_arousal_gradients(patient_left_individual_raw_data, zero_start_time, light_on_time)
left_max_PLRs = extract_max_PLR(patient_left_individual_raw_data, light_on_time, LOR_early_start_time)
left_50_per_PLR_times, left_50_per_LOR_times = extract_50_percent_timestamps(patient_left_individual_raw_data, patient_left_closest_timestamp_LOR_early_start_time, light_on_time, LOR_early_start_time, LOR_late_end_time)
left_LOR_early_gradients = extract_LOR_early_gradients(patient_left_individual_raw_data, LOR_early_start_time, LOR_early_end_time)
left_LOR_late_gradients = extract_LOR_late_gradients(patient_left_individual_raw_data, LOR_late_start_time, LOR_late_end_time)
def dict_to_metric_df(metric_dict, value_name):
    return (
        pd.concat(metric_dict,
                  names=["record_id", "redcap_repeat_instance"])
          .rename(value_name)
          .reset_index()
    )
metric_dfs = [
    dict_to_metric_df(left_arousal_gradients, "left_arousal_gradient"),
    dict_to_metric_df(left_max_PLRs, "left_max_PLR"),
    dict_to_metric_df(left_LOR_early_gradients, "left_LOR_early_gradient"),
    dict_to_metric_df(left_LOR_late_gradients, "left_LOR_late_gradient"),
]
metric_dfs += [
    dict_to_metric_df(left_50_per_PLR_times, "left_50pct_PLR_time"),
    dict_to_metric_df(left_50_per_LOR_times, "left_50pct_LOR_time"),
]
left_seconds = {
    record_id: df.loc["SECONDS"]
    for record_id, df in patient_left_individual_text_data.items()
    if "SECONDS" in df.index
}
left_seconds_df = dict_to_metric_df(
    left_seconds,
    "left_SECONDS"
)
# metric_dfs += [left_seconds_df]

from functools import reduce

left_metrics_df = reduce(
    lambda l, r: l.merge(
        r,
        on=["record_id", "redcap_repeat_instance"],
        how="outer"
    ),
    metric_dfs
)
left_metrics_df = left_metrics_df.merge(
    NPI_data_cleaned[
        ["record_id", "redcap_repeat_instance", "date_examination_merged"]
    ],
    on=["record_id", "redcap_repeat_instance"],
    how="left"
)
SECONDS_conversion_dict = {
"C": 0,
"U": 1,
"M-": 2,
"M+": 3,
"E": 4
}
# left_metrics_df = left_metrics_df.replace(SECONDS_conversion_dict)
survival = {
    record_id: df.loc["90-day survival"].iloc[0]
    for record_id, df in patient_left_individual_text_data.items()
    if "90-day survival" in df.index
}
ICU_outcome = {
    record_id: df.loc["Outcome at ICU"].iloc[0]
    for record_id, df in patient_left_individual_text_data.items()
    if "Outcome at ICU" in df.index
}
df_outcome = (
    pd.DataFrame.from_dict(
        ICU_outcome,
        orient="index",
        columns=["ICU_outcome"],
    )
    .reset_index()
    .rename(columns={"index": "record_id"})
)

left_metrics_df.to_csv(
    os.path.join(save_path_extracted_features, f'pupilometry_features.csv'))
df_outcome.to_csv(
    os.path.join(save_path_extracted_features, f'ICU_outcome.csv'))


for day in np.unique(left_metrics_df["redcap_repeat_instance"]):
    day_df = pd.concat(
        {
            patient_id: df[day]          # column 1 = day one
            for patient_id, df in patient_left_individual_raw_data.items()
            if day in df.columns
        },
        axis=1
    )

    day_df.columns.name = "patient_id"
    day_df = day_df.drop(list(day_df.columns[day_df.isna().any()]), axis=1)  # drop patients with incomplete data

    day_df.to_csv(
    os.path.join(day_save_path, f'day{day}_left_raw_data.csv'))