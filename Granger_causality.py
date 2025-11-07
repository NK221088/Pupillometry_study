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
    granger_gradient_FOUR = nta.GrangerAnalyzer(time_series_gradient_FOUR, order=1, n_freqs=1)
    gradient_to_FOUR = granger_gradient_FOUR.causality_xy[0, 1, :]
    
    if np.isnan(np.mean(gradient_to_FOUR)):
        print(f"Skipping {subject}: Granger causalities is np.nan.")
        continue
    
    granger_results.append({
        "Subject ID": subject,
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
    
    gradient_to_FOUR = df_results.loc[index].values[0]
    
    plotting_df = pd.DataFrame({
        "Granger Causality values": gradient_to_FOUR,
        "Metric": "LOR Late Gradient → FOUR"
    })
    
    sns.boxplot(x="Metric", y="Granger Causality values", data=plotting_df, ax=ax[row, col])
    
    ax[row, col].set_title(f"Subject {index}")
    ax[row, col].set_ylabel("Granger Causality")
    ax[row, col].tick_params(axis='x')

    if row < 2:  # since 0-indexed rows → bottom row = row 2
        ax[row, col].set_xlabel("")
        ax[row, col].set_xticklabels([])


plt.tight_layout(pad=3.0)
plt.show()
print("test")