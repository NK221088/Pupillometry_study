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
  
savepath = os.getenv("save_path")
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

    f.tight_layout(rect=[0.1, 0, 1, 0.95])  # leaves space for row headers and suptitle

    f.savefig(
        savepath + f'\\subject_{idx_left}.pdf',
        dpi=300,                     
        bbox_inches='tight',
        format='pdf'
    )
    plt.close(f)
print("test")

