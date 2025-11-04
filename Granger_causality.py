import matplotlib.pyplot as plt
import nitime.timeseries as ts
import nitime.analysis as nta
import numpy as np
import pandas as pd
import seaborn as sns

from read_data import (patient_left_FOUR_metrics, patient_left_first_50_metrics, patient_left_second_50_metrics, patient_left_LOR_late_gradient_metrics)

patient_left_scores = {}
for index in patient_left_FOUR_metrics.index:
    patient_left_scores[index] = np.vstack([patient_left_FOUR_metrics.loc[index].dropna(), patient_left_LOR_late_gradient_metrics.loc[index].dropna()])
    patient_left_scores[index] = patient_left_scores[index].astype(np.float64)


mock_scores = {}
mock_indices = np.arange(1,101,1)
for index in mock_indices:
    mock_scores[index] = np.vstack([np.random.randint(1, 13, size=(1, 10)), np.random.rand(1, 10)])

granger_results = []
for subject, subject_scores in patient_left_scores.items():
    
    if subject_scores.shape[1] < 3:
        print(f"Skipping {subject}: Insufficient data points.  ({subject_scores.shape[1]})")
        continue
    # Check for zero variance:
    
    FOUR_score = subject_scores[0, :]
    gradient = subject_scores[1, :]
    
    var_four = np.var(FOUR_score)
    var_gradient = np.var(gradient)
    
    if var_four == 0 or var_gradient == 0:
        print(f"Skipping {subject}: Zero variance for at least one parameter")
        continue

    gradient_FOUR_score = np.vstack([gradient, FOUR_score])
    time_series_gradient_FOUR = ts.TimeSeries(gradient_FOUR_score, sampling_interval=24, time_unit="h")
    granger_gradient_FOUR = nta.GrangerAnalyzer(time_series_gradient_FOUR, order=1)
    gradient_to_FOUR = granger_gradient_FOUR.causality_xy[0, 1, :]
    
    FOUR_only = np.vstack([FOUR_score, FOUR_score])
    time_series_FOUR = ts.TimeSeries(FOUR_only, sampling_interval=24, time_unit="h")
    granger_FOUR = nta.GrangerAnalyzer(time_series_FOUR, order=1)
    FOUR_to_FOUR = granger_gradient_FOUR.causality_xy[0, 1, :]
    
    if np.isnan(np.mean(FOUR_to_FOUR)) or np.isnan(np.mean(gradient_to_FOUR)):
        print(f"Skipping {subject}: Either Granger causalities are np.nan.")
        continue
    
    granger_results.append({
        "Subject ID": subject,
        "FOUR to FOUR": FOUR_to_FOUR,
        "Gradient to FOUR": gradient_to_FOUR
    })
        
df_results = pd.DataFrame(granger_results)
df_results.set_index("Subject ID", inplace=True)
print(df_results)

sns.set_style("whitegrid")
fig, ax = plt.subplots(3, 4, figsize=(30, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for idx, index in enumerate(df_results.index[:12]):
    row = idx // 4
    col = idx % 4
    
    caus_xy = df_results.loc[index].values[0]
    caus_yx = df_results.loc[index].values[1]
    
    plotting_df = pd.DataFrame({
        "Granger Causality values": np.concatenate([caus_xy, caus_yx]),
        "Metric": (["FOUR → LOR Late Gradient"] * len(caus_xy)) +
                  (["LOR Late Gradient → FOUR"] * len(caus_yx))
    })
    
    sns.boxplot(x="Metric", y="Granger Causality values", data=plotting_df, ax=ax[row, col])
    
    ax[row, col].set_title(f"Subject {index}")
    ax[row, col].set_ylabel("Granger Causality")
    ax[row, col].tick_params(axis='x', rotation=15)

    if row < 2:  # since 0-indexed rows → bottom row = row 2
        ax[row, col].set_xlabel("")
        ax[row, col].set_xticklabels([])


plt.tight_layout(pad=3.0)
plt.show()
print("test")