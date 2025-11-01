import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import numpy as np

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

spearman_results_left = {}
spearman_results_right = {}
model_results_left = {}

save_path_scatter = os.getenv("save_path_scatter")
save_path_spearman = os.getenv("save_path_spearman")
save_path_models = os.getenv("save_path_models")
for idx_left, idx_right in zip(patient_left_first_50_metrics.index, patient_right_first_50_metrics.index):
    
    print(f"Plotting subject {idx_left}/{patient_left_first_50_metrics.index[-1]}")
    
    f, axs = plt.subplots(3, 3, figsize=(30, 10))
    f.suptitle(f"Subject {idx_left} — GCS/FOUR vs PLR Metrics", fontsize=18, fontweight='bold')

    # Row headers
    f.text(0.05, 0.83, "Same-day", fontsize=16, fontweight='bold', va='center')
    f.text(0.05, 0.53, "Lag 1 day", fontsize=16, fontweight='bold', va='center')
    f.text(0.05, 0.23, "Lag 2 days", fontsize=16, fontweight='bold', va='center')

    # Column headers
    axs[0][0].set_title("Time to reach 50% PLR (decrease)")
    axs[0][1].set_title("Time to reach 50% PLR (increase)")
    axs[0][2].set_title("LOR,Late Gradient")

    longest_length_left = min([len(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)), len(patient_left_GCS_metrics.loc[idx_left].dropna().values.astype(float)), len(patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float))])
    longest_length_right = min([len(patient_right_first_50_metrics.loc[idx_right].dropna().values.astype(float)), len(patient_right_GCS_metrics.loc[idx_right].dropna().values.astype(float)), len(patient_right_FOUR_metrics.loc[idx_right].dropna().values.astype(float))])
    
    axs[0][0].scatter(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], patient_left_GCS_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], color="blue", marker="o", label="Left, GCS")
    axs[0][0].scatter(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], color="red", marker="o", label="Left, FOUR")
    # axs[0][0].scatter(patient_right_first_50_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right], patient_right_GCS_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right], color="blue", marker="s", label="Right, GCS")
    # axs[0][0].scatter(patient_right_first_50_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right], patient_right_FOUR_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right], color="red", marker="s", label="Right, FOUR")
    axs[0][0].legend()

    axs[1][0].scatter(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1], patient_left_GCS_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:], color="blue", marker="o", label="Left, GCS")
    axs[1][0].scatter(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:], color="red", marker="o", label="Left, FOUR")
    # axs[1][0].scatter(patient_right_first_50_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][:-1], patient_right_GCS_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][1:], color="blue", marker="s", label="Right, GCS")
    # axs[1][0].scatter(patient_right_first_50_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][:-1], patient_right_FOUR_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][1:], color="red", marker="s", label="Right, FOUR")
    axs[1][0].legend()

    axs[2][0].scatter(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2], patient_left_GCS_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:], color="blue", marker="o", label="Left, GCS")
    axs[2][0].scatter(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:], color="red", marker="o", label="Left, FOUR")
    # axs[1][0].scatter(patient_right_first_50_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][:-1], patient_right_GCS_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][1:], color="blue", marker="s", label="Right, GCS")
    # axs[1][0].scatter(patient_right_first_50_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][:-1], patient_right_FOUR_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][1:], color="red", marker="s", label="Right, FOUR")
    axs[2][0].legend()
    
    patient_left_spearman_results_first_50_0_lag = stats.spearmanr(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left])
    patient_left_spearman_results_first_50_1_lag = stats.spearmanr(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:])
    patient_left_spearman_results_first_50_2_lag = stats.spearmanr(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:])
    patient_left_first_50_results = [patient_left_spearman_results_first_50_0_lag, patient_left_spearman_results_first_50_1_lag, patient_left_spearman_results_first_50_2_lag]
    
    if len(np.unique(patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left])) >= 2 & len(np.unique(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left])) >= 2:
        patient_left_model_first_50_0_lag = OrderedModel(
        endog=patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left],
        exog=patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left],
        distr="logit"
        )
    else:
        patient_left_model_first_50_0_lag = None
    if len(np.unique(patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1])) >= 2 & len(np.unique(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:])) >= 2:
        patient_left_model_first_50_1_lag = OrderedModel(
        endog=patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1],
        exog=patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:],
        distr="logit"
        )
    else:
        patient_left_model_first_50_1_lag = None
    if len(np.unique(patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2])) >= 2 & len(np.unique(patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:])) >= 2:
        patient_left_model_first_50_2_lag = OrderedModel(
        endog=patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2],
        exog=patient_left_first_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:],
        distr="logit"
        )
    else:
        patient_left_model_first_50_2_lag = None
    patient_left_model_res_first_50_0_lag = patient_left_model_first_50_0_lag.fit(method='bfgs') if patient_left_model_first_50_0_lag is not None else None
    patient_left_model_res_first_50_1_lag = patient_left_model_first_50_1_lag.fit(method='bfgs') if patient_left_model_first_50_1_lag is not None else None
    patient_left_model_res_first_50_2_lag = patient_left_model_first_50_2_lag.fit(method='bfgs') if patient_left_model_first_50_2_lag is not None else None
    patient_left_first_50_model_results = [patient_left_model_res_first_50_0_lag, patient_left_model_res_first_50_1_lag, patient_left_model_res_first_50_2_lag]
    
    axs[0][1].scatter(patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], patient_left_GCS_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], color="blue", marker="o", label="Left, GCS")
    axs[0][1].scatter(patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], color="red", marker="o", label="Left, FOUR")
    # axs[0][1].scatter(patient_right_second_50_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right], patient_right_GCS_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right], color="blue", marker="s", label="Right, GCS")
    # axs[0][1].scatter(patient_right_second_50_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right], patient_right_FOUR_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right], color="red", marker="s", label="Right, FOUR")
    axs[0][1].legend()

    axs[1][1].scatter(patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1], patient_left_GCS_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:], color="blue", marker="o", label="Left, GCS")
    axs[1][1].scatter(patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:], color="red", marker="o", label="Left, FOUR")
    # axs[1][1].scatter(patient_right_second_50_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][:-1], patient_right_GCS_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][1:], color="blue", marker="s", label="Right, GCS")
    # axs[1][1].scatter(patient_right_second_50_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][:-1], patient_right_FOUR_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][1:], color="red", marker="s", label="Right, FOUR")
    axs[1][1].legend()

    axs[2][1].scatter(patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2], patient_left_GCS_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:], color="blue", marker="o", label="Left, GCS")
    axs[2][1].scatter(patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:], color="red", marker="o", label="Left, FOUR")
    # axs[1][1].scatter(patient_right_second_50_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][:-1], patient_right_GCS_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][1:], color="blue", marker="s", label="Right, GCS")
    # axs[1][1].scatter(patient_right_second_50_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][:-1], patient_right_FOUR_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][1:], color="red", marker="s", label="Right, FOUR")
    axs[2][1].legend()
    
    patient_left_spearman_results_second_50_0_lag = stats.spearmanr(patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left])
    patient_left_spearman_results_second_50_1_lag = stats.spearmanr(patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:])
    patient_left_spearman_results_second_50_2_lag = stats.spearmanr(patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:])
    patient_left_second_50_results = [patient_left_spearman_results_second_50_0_lag, patient_left_spearman_results_second_50_1_lag, patient_left_spearman_results_second_50_2_lag]

    if len(np.unique(patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left])) >= 2 & len(np.unique(patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left])) >= 2:
        patient_left_model_second_50_0_lag = OrderedModel(
        endog=patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left],
        exog=patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left],
        distr="logit"
        )
    else:
        patient_left_model_second_50_0_lag = None
    if len(np.unique(patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1])) >= 2 & len(np.unique(patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:])) >= 2:
        patient_left_model_second_50_1_lag = OrderedModel(
        endog=patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1],
        exog=patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:],
        distr="logit"
        )
    else:
        patient_left_model_second_50_1_lag = None
    if len(np.unique(patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2])) >= 2 & len(np.unique(patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:])) >= 2:
        patient_left_model_second_50_2_lag = OrderedModel(
        endog=patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2],
        exog=patient_left_second_50_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:],
        distr="logit"
        )
    else:
        patient_left_model_second_50_2_lag = None
    patient_left_model_res_second_50_0_lag = patient_left_model_second_50_0_lag.fit(method='bfgs') if patient_left_model_second_50_0_lag is not None else None
    patient_left_model_res_second_50_1_lag = patient_left_model_second_50_1_lag.fit(method='bfgs') if patient_left_model_second_50_1_lag is not None else None
    patient_left_model_res_second_50_2_lag = patient_left_model_second_50_2_lag.fit(method='bfgs') if patient_left_model_second_50_2_lag is not None else None
    patient_left_second_50_model_results = [patient_left_model_res_second_50_0_lag, patient_left_model_res_second_50_1_lag, patient_left_model_res_second_50_2_lag]
    
    axs[0][2].scatter(patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], patient_left_GCS_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], color="blue", marker="o", label="Left, GCS")
    axs[0][2].scatter(patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], color="red", marker="o", label="Left, FOUR")
    # axs[0][2].scatter(patient_right_LOR_late_gradient_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right], patient_right_GCS_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right], color="blue", marker="s", label="Right, GCS")
    # axs[0][2].scatter(patient_right_LOR_late_gradient_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right], patient_right_FOUR_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right], color="red", marker="s", label="Right, FOUR")
    axs[0][2].legend()

    axs[1][2].scatter(patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1], patient_left_GCS_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:], color="blue", marker="o", label="Left, GCS")
    axs[1][2].scatter(patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:], color="red", marker="o", label="Left, FOUR")
    # axs[1][2].scatter(patient_right_LOR_late_gradient_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][:-1], patient_right_GCS_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][1:], color="blue", marker="s", label="Right, GCS")
    # axs[1][2].scatter(patient_right_LOR_late_gradient_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][:-1], patient_right_FOUR_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][1:], color="red", marker="s", label="Right, FOUR")
    axs[1][2].legend()

    axs[2][2].scatter(patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2], patient_left_GCS_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:], color="blue", marker="o", label="Left, GCS")
    axs[2][2].scatter(patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:], color="red", marker="o", label="Left, FOUR")
    # axs[1][2].scatter(patient_right_LOR_late_gradient_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][:-1], patient_right_GCS_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][1:], color="blue", marker="s", label="Right, GCS")
    # axs[1][2].scatter(patient_right_LOR_late_gradient_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][:-1], patient_right_FOUR_metrics.loc[idx_right].dropna().values.astype(float)[:longest_length_right][1:], color="red", marker="s", label="Right, FOUR")
    axs[2][2].legend()

    patient_left_spearman_results_LOR_late_gradient_0_lag = stats.spearmanr(patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left])
    patient_left_spearman_results_LOR_late_gradient_1_lag = stats.spearmanr(patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:])
    patient_left_spearman_results_LOR_late_gradient_2_lag = stats.spearmanr(patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2], patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:])
    patient_left_LOR_late_gradient_results = [patient_left_spearman_results_LOR_late_gradient_0_lag, patient_left_spearman_results_LOR_late_gradient_1_lag, patient_left_spearman_results_LOR_late_gradient_2_lag]

    if len(np.unique(patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left])) >= 2 & len(np.unique(patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left])) >= 2:
        patient_left_model_LOR_late_gradient_0_lag = OrderedModel(
        endog=patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left],
        exog=patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left],
        distr="logit"
        )
    else:
        patient_left_model_LOR_late_gradient_0_lag = None
    if len(np.unique(patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1])) >= 2 & len(np.unique(patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:])) >= 2:
        patient_left_model_LOR_late_gradient_1_lag = OrderedModel(
        endog=patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-1],
        exog=patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][1:],
        distr="logit"
        )
    else:
        patient_left_model_LOR_late_gradient_1_lag = None
    if len(np.unique(patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2])) >= 2 & len(np.unique(patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:])) >= 2:
        patient_left_model_LOR_late_gradient_2_lag = OrderedModel(
        endog=patient_left_FOUR_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][:-2],
        exog=patient_left_LOR_late_gradient_metrics.loc[idx_left].dropna().values.astype(float)[:longest_length_left][2:],
        distr="logit"
        )
    else:
        patient_left_model_LOR_late_gradient_2_lag = None
    patient_left_model_res_LOR_late_gradient_0_lag = patient_left_model_LOR_late_gradient_0_lag.fit(method='bfgs') if patient_left_model_LOR_late_gradient_0_lag is not None else None
    patient_left_mode_res_LOR_late_gradient_1_lag = patient_left_model_LOR_late_gradient_1_lag.fit(method='bfgs') if patient_left_model_LOR_late_gradient_1_lag is not None else None
    patient_left_mode_res_LOR_late_gradient_2_lag = patient_left_model_LOR_late_gradient_2_lag.fit(method='bfgs') if patient_left_model_LOR_late_gradient_2_lag is not None else None
    patient_left_LOR_late_gradient_model_results = [patient_left_model_res_LOR_late_gradient_0_lag, patient_left_mode_res_LOR_late_gradient_1_lag, patient_left_mode_res_LOR_late_gradient_2_lag]
    
    
    f.tight_layout(rect=[0.1, 0, 1, 0.95])  # leaves space for row headers and suptitle

    f.savefig(
        save_path_scatter + f'\\subject_{idx_left}.pdf',
        dpi=300,                     
        bbox_inches='tight',
        format='pdf'
    )
    plt.close(f)
    
    spearman_results_left[idx_left] = [patient_left_first_50_results, patient_left_second_50_results, patient_left_LOR_late_gradient_results]
    model_results_left[idx_left] = [patient_left_first_50_model_results, patient_left_second_50_model_results, patient_left_LOR_late_gradient_model_results]

for idx_left in patient_left_first_50_metrics.index:
    lag = [0,1,2]
    df = pd.DataFrame({
        "Lag": lag,
        "Spearman_r_first_50": [spearman_results_left[idx_left][0][i].statistic for i in lag],
        "Spearman_r_second_50": [spearman_results_left[idx_left][1][i].statistic for i in lag],
        "Spearman_r_LOR_late_gradient": [spearman_results_left[idx_left][2][i].statistic for i in lag],
        "Spearman_pvalue_first_50": [spearman_results_left[idx_left][0][i].pvalue for i in lag],
        "Spearman_pvalue_second_50": [spearman_results_left[idx_left][1][i].pvalue for i in lag],
        "Spearman_pvalue_LOR_late_gradient": [spearman_results_left[idx_left][2][i].pvalue for i in lag],
    })

    f, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    f.subplots_adjust(hspace=0.3, wspace=0.3)

    # Row and column titles
    col_titles = ["first_50", "second_50", "LOR_late_gradient"]
    row_titles = [r"Spearman $\rho$", "p-value"]

    for i, col in enumerate(col_titles):
        axes[0, i].set_title(col, fontsize=11, pad=10)

    # Add shared row labels using figure text
    f.text(0.04, 0.75, row_titles[0], va='center', rotation='vertical', fontsize=12)
    f.text(0.04, 0.25, row_titles[1], va='center', rotation='vertical', fontsize=12)

    # Shared x-axis label
    f.text(0.5, 0.04, "Lag", ha='center', fontsize=12)

    sns.boxplot(x="Lag", y="Spearman_r_first_50", data=df, ax=axes[0][0])
    sns.boxplot(x="Lag", y="Spearman_r_second_50", data=df, ax=axes[0][1])
    sns.boxplot(x="Lag", y="Spearman_r_LOR_late_gradient", data=df, ax=axes[0][2])
    sns.boxplot(x="Lag", y="Spearman_pvalue_first_50", data=df, ax=axes[1][0])
    sns.boxplot(x="Lag", y="Spearman_pvalue_second_50", data=df, ax=axes[1][1])
    sns.boxplot(x="Lag", y="Spearman_pvalue_LOR_late_gradient", data=df, ax=axes[1][2])

    # Remove all individual axis labels
    for ax in axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        
    f.suptitle("Spearman Correlation Results", fontsize=13, y=0.98)
    f.savefig(
    save_path_spearman + f'\\subject_{idx_left}_spearman_results.pdf',
    dpi=300,                     
    bbox_inches='tight',
    format='pdf'
    )
    plt.close(f)

    df_model = pd.DataFrame({
        "Lag": lag,
        "Model_slope_first_50": [model_results_left[idx_left][0][i].params[0] if model_results_left[idx_left][0][i] is not None else None for i in lag],
        "Model_slope_second_50": [model_results_left[idx_left][1][i].params[0] if model_results_left[idx_left][1][i] is not None else None for i in lag],
        "Model_slope_LOR_late_gradient": [model_results_left[idx_left][2][i].params[0] if model_results_left[idx_left][2][i] is not None else None for i in lag],
        "Model_pvalue_first_50": [model_results_left[idx_left][0][i].pvalues[0] if model_results_left[idx_left][0][i] is not None else None for i in lag],
        "Model_pvalue_second_50": [model_results_left[idx_left][1][i].pvalues[0] if model_results_left[idx_left][1][i] is not None else None for i in lag],
        "Model_pvalue_LOR_late_gradient": [model_results_left[idx_left][2][i].pvalues[0] if model_results_left[idx_left][2][i] is not None else None for i in lag],
    })

    f, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    f.subplots_adjust(hspace=0.3, wspace=0.3)

    # Row and column titles
    col_titles = ["first_50", "second_50", "LOR_late_gradient"]
    row_titles = [r"Slope of model", "p-value"]

    for i, col in enumerate(col_titles):
        axes[0, i].set_title(col, fontsize=11, pad=10)

    # Add shared row labels using figure text
    f.text(0.04, 0.75, row_titles[0], va='center', rotation='vertical', fontsize=12)
    f.text(0.04, 0.25, row_titles[1], va='center', rotation='vertical', fontsize=12)

    # Shared x-axis label
    f.text(0.5, 0.04, "Lag", ha='center', fontsize=12)

    sns.boxplot(x="Lag", y="Model_slope_first_50", data=df_model, ax=axes[0][0])
    sns.boxplot(x="Lag", y="Model_slope_second_50", data=df_model, ax=axes[0][1])
    sns.boxplot(x="Lag", y="Model_slope_LOR_late_gradient", data=df_model, ax=axes[0][2])
    sns.boxplot(x="Lag", y="Model_pvalue_first_50", data=df_model, ax=axes[1][0])
    sns.boxplot(x="Lag", y="Model_pvalue_second_50", data=df_model, ax=axes[1][1])
    sns.boxplot(x="Lag", y="Model_pvalue_LOR_late_gradient", data=df_model, ax=axes[1][2])

    # Remove all individual axis labels
    for ax in axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        
    f.suptitle("Ordinal Logistic Regression Results", fontsize=13, y=0.98)
    f.savefig(
    save_path_models + f'\\subject_{idx_left}_model_results.pdf',
    dpi=300,                     
    bbox_inches='tight',
    format='pdf'
    )
    plt.close(f)
    

print("test")