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
    var0 = np.var(subject_scores[0])
    var1 = np.var(subject_scores[1])
    
    if var0 == 0 or var1 == 0:
        print(f"Skipping {subject}: Zero variance for at least one parameter")
        continue
        
    time_series = ts.TimeSeries(subject_scores, sampling_interval=24, time_unit="h")
    granger = nta.GrangerAnalyzer(time_series, order=1)
    granger_four_to_LOR_late_gradient = np.mean(granger.causality_xy[0, 1, :])
    granger_LOR_late_gradient_to_four = np.mean(granger.causality_yx[0, 1, :])
    if np.isnan(granger_four_to_LOR_late_gradient) or np.isnan(granger_LOR_late_gradient_to_four):
        print(f"Skipping {subject}: Both Granger causalities are np.nan.")
        continue
    granger_results.append({
        "Subject ID": subject,
        "four_to_LOR_late_gradient": granger_four_to_LOR_late_gradient,
        "LOR_late_gradient_to_four": granger_LOR_late_gradient_to_four
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
    Causalities = df_results.loc[index].tolist()
    for i, (label, value) in enumerate(zip(df_results.columns, Causalities)):
        ax[row, col].hlines(
            y=value, 
            xmin=i-0.3,  # Start of line
            xmax=i+0.3,  # End of line
            colors='steelblue', 
            linewidth=4,
            alpha=0.7
        )
    ax[row, col].set_title(f"Subject {index}", fontsize=14, fontweight='bold', pad=10)

plt.tight_layout()
plt.show()
print("test")