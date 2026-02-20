import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np
import re
from read_NPi_data import NPI_data_cleaned

load_dotenv()

HC_left_path = os.getenv("HC_left_data_path")
HC_right_path = os.getenv("HC_right_data_path")
patient_left_path = os.getenv("patient_left_250_data_path")
patient_right_path = os.getenv("patient_right_250_data_path")

LOR_early_start_time = 6

HC_right_data = pd.read_excel(HC_right_path, index_col=0, sheet_name=None)

#######################################################################################################################################################
# HCs left
#######################################################################################################################################################

HC_left_data = pd.read_excel(HC_left_path, index_col=0, header=2, sheet_name="Ark1")
HC_left_text_data = pd.read_excel(HC_left_path, index_col=0, header=None, nrows=2)

HC_left_data_individual_raw_data = {HC_id: HC_left_data[HC_id] for HC_id in HC_left_data.columns}
HC_left_data_individual_raw_data = {HC_id: HC_left_text_data[HC_id] for HC_id in HC_left_text_data.columns}

#######################################################################################################################################################
# HCs right
#######################################################################################################################################################

HC_right_data = pd.read_excel(HC_right_path, index_col=0, header=2, sheet_name="Ark1")
HC_right_text_data = pd.read_excel(HC_right_path, index_col=0, header=None, nrows=2)

HC_right_data_individual_raw_data = {HC_id: HC_right_data[HC_id] for HC_id in HC_right_data.columns}
HC_right_data_individual_raw_data = {HC_id: HC_right_text_data[HC_id] for HC_id in HC_right_text_data.columns}

#######################################################################################################################################################
# Patient left
#######################################################################################################################################################

patient_left_data = pd.read_excel(patient_left_path, index_col=0, sheet_name=None)
patient_left_zero_indices = {sheet_name: np.where(patient_left_data[sheet_name].index == 0)[0][0] for sheet_name in patient_left_data.keys()} # Finding the first time index
patient_left_closest_timestamp_LOR_early_start_time = {sheet_name: np.argmin(np.abs((np.array([idx for idx in patient_left_data[sheet_name].index if isinstance(idx, (int, float))]) - LOR_early_start_time))) for sheet_name in patient_left_data.keys()} # Find the time index closest to LOR early start time
patient_left_text_data = {sheet_name: patient_left_data[sheet_name].iloc[:patient_left_zero_indices[sheet_name]] for sheet_name in patient_left_data.keys()} # Extract text data -> Everything before first time index
patient_left_numeric_data = {sheet_name: patient_left_data[sheet_name].iloc[patient_left_zero_indices[sheet_name]:].apply(pd.to_numeric, errors='coerce') for sheet_name in patient_left_data.keys()}

patient_left_data = {
    i: value
    for i, value in enumerate(patient_left_data.values(), start=1)
}

all_patient_ids = list(patient_left_data[1].keys()) # Define all patient IDs

patient_left_numeric_data = {
    i: value
    for i, value in enumerate(patient_left_numeric_data.values(), start=1)
}
patient_left_text_data = {
    i: value
    for i, value in enumerate(patient_left_text_data.values(), start=1)
}

patient_left_raw_values = {sheet_name: patient_left_numeric_data[sheet_name] for sheet_name in patient_left_data.keys()}

patient_left_individual_raw_data = {patient_id: pd.concat([
    patient_left_raw_values[sheet_name][patient_id] if patient_id in patient_left_raw_values[sheet_name].columns else pd.Series(dtype='float64')
    for sheet_name in patient_left_data.keys()
    ], axis=1, keys=patient_left_data.keys()) for patient_id in all_patient_ids}

patient_left_individual_text_data = {patient_id: pd.concat([
    patient_left_text_data[sheet_name][patient_id] if patient_id in patient_left_text_data[sheet_name].columns else pd.Series(dtype='float64')
    for sheet_name in patient_left_text_data.keys()
    ], axis=1, keys=patient_left_text_data.keys()) for patient_id in all_patient_ids}

patient_left_individual_text_data = {patient_id: pd.concat([
    patient_left_text_data[sheet_name][patient_id] if patient_id in patient_left_text_data[sheet_name].columns else pd.Series(dtype='float64')
    for sheet_name in patient_left_text_data.keys()
    ], axis=1, keys=patient_left_text_data.keys()) for patient_id in all_patient_ids}

#######################################################################################################################################################
# Patient right
#######################################################################################################################################################

patient_right_data = pd.read_excel(patient_right_path, index_col=0, sheet_name=None)
patient_right_zero_indices = {sheet_name: np.where(patient_right_data[sheet_name].index == 0)[0][0] for sheet_name in patient_right_data.keys()} # Finding the first time index
patient_right_closest_timestamp_LOR_early_start_time = {sheet_name: np.argmin(np.abs((np.array([idx for idx in patient_right_data[sheet_name].index if isinstance(idx, (int, float))]) - LOR_early_start_time))) for sheet_name in patient_right_data.keys()} # Find the time index closest to LOR early start time
patient_right_text_data = {sheet_name: patient_right_data[sheet_name].iloc[:patient_right_zero_indices[sheet_name]] for sheet_name in patient_right_data.keys()} # Extract text data -> Everything before first time index
patient_right_numeric_data = {sheet_name: patient_right_data[sheet_name].iloc[patient_right_zero_indices[sheet_name]:].apply(pd.to_numeric, errors='coerce') for sheet_name in patient_right_data.keys()}

patient_right_data = {
    i: value
    for i, value in enumerate(patient_right_data.values(), start=1)
}
patient_right_numeric_data = {
    i: value
    for i, value in enumerate(patient_right_numeric_data.values(), start=1)
}
patient_right_text_data = {
    i: value
    for i, value in enumerate(patient_right_text_data.values(), start=1)
}

patient_right_raw_values = {sheet_name: patient_right_numeric_data[sheet_name] for sheet_name in patient_right_data.keys()}

patient_right_individual_raw_data = {patient_id: pd.concat([
    patient_right_raw_values[sheet_name][patient_id] if patient_id in patient_right_raw_values[sheet_name].columns else pd.Series(dtype='float64')
    for sheet_name in patient_right_data.keys()
    ], axis=1, keys=patient_right_data.keys()) for patient_id in all_patient_ids}

patient_right_individual_text_data = {patient_id: pd.concat([
    patient_right_text_data[sheet_name][patient_id] if patient_id in patient_right_text_data[sheet_name].columns else pd.Series(dtype='float64')
    for sheet_name in patient_right_text_data.keys()
    ], axis=1, keys=patient_right_text_data.keys()) for patient_id in all_patient_ids}

patient_right_individual_text_data = {patient_id: pd.concat([
    patient_right_text_data[sheet_name][patient_id] if patient_id in patient_right_text_data[sheet_name].columns else pd.Series(dtype='float64')
    for sheet_name in patient_right_text_data.keys()
    ], axis=1, keys=patient_right_text_data.keys()) for patient_id in all_patient_ids}


#######################################################################################################################################################
# Sorting patient data
#######################################################################################################################################################

for patient_id in all_patient_ids:
    if patient_id == 74:
        continue
    visit_order = (
        NPI_data_cleaned[NPI_data_cleaned["record_id"] == patient_id]
        .sort_values("date_examination_merged")
        ["redcap_repeat_instance"]
        .tolist()
    )

    patient_left_individual_raw_data[patient_id] = (
        patient_left_individual_raw_data[patient_id]
        .loc[:, visit_order]
    )

    patient_left_individual_text_data[patient_id] = (
        patient_left_individual_text_data[patient_id]
        .loc[:, visit_order]
    )

    patient_right_individual_raw_data[patient_id] = (
        patient_right_individual_raw_data[patient_id]
        .loc[:, visit_order]
    )

    patient_right_individual_text_data[patient_id] = (
        patient_right_individual_text_data[patient_id]
        .loc[:, visit_order]
    )

print("debug")