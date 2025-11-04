from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from read_data import (patient_left_FOUR_metrics, patient_left_GCS_metrics, patient_left_first_50_metrics, patient_left_second_50_metrics, patient_left_LOR_late_gradient_metrics)

wakefullness_metric = "patient_left_GCS_metrics" # "patient_left_FOUR_metrics" 

if wakefullness_metric == "patient_left_FOUR_metrics":
    patient_left_scores = {}
    for index in patient_left_FOUR_metrics.index:
        patient_left_scores[index] = np.vstack([patient_left_FOUR_metrics.loc[index].dropna(), patient_left_LOR_late_gradient_metrics.loc[index].dropna()])
        patient_left_scores[index] = patient_left_scores[index].astype(np.float64)

else:
    patient_left_scores = {}
    for index in patient_left_GCS_metrics.index:
        patient_left_scores[index] = np.vstack([patient_left_GCS_metrics.loc[index].dropna(), patient_left_LOR_late_gradient_metrics.loc[index].dropna()])
        patient_left_scores[index] = patient_left_scores[index].astype(np.float64)


mock_scores = {}
mock_indices = np.arange(1,101,1)
for index in mock_indices:
    mock_scores[index] = np.vstack([np.random.randint(1, 13, size=(1, 10)), np.random.rand(1, 10)])

granger_results = []
max_lag = 2
for subject, subject_scores in patient_left_scores.items():
    
    if subject_scores.shape[1] <= (3 * max_lag + 1): # 3 * maxlag + 1 (constant term)
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

    FOUR_FOUR_model = AutoReg(FOUR_score, lags=max_lag).fit()
    gradient_to_FOUR_model = grangercausalitytests(np.column_stack([FOUR_score, gradient]), maxlag=max_lag)
    
    granger_results.append({
        "Subject ID": subject,
        "FOUR to FOUR": FOUR_FOUR_model.pvalues[1],
        "Gradient to FOUR": gradient_to_FOUR_model[max_lag][0]['ssr_ftest'][1]
    })
        
df_results = pd.DataFrame(granger_results)
df_results.set_index("Subject ID", inplace=True)

number_of_overlapping_patients = len(list(
    set(df_results[df_results["Gradient to FOUR"] < 0.05].index) &
    set(df_results[df_results["FOUR to FOUR"] < 0.05].index)
))
df_results_summarized = pd.concat([df_results[df_results["FOUR to FOUR"] < 0.05][:5], df_results[df_results["Gradient to FOUR"] < 0.05]]).reset_index().drop_duplicates(subset='Subject ID').set_index('Subject ID')

sns.set_style("whitegrid")
col_size = 2
fig, ax = plt.subplots(3, col_size, figsize=(30, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for idx, index in enumerate(df_results_summarized.index):
    row = idx // col_size
    col = idx % col_size
    
    FOUR_to_FOUR = df_results.loc[index].values[0]
    gradient_to_FOUR = df_results.loc[index].values[1]
    
    plotting_df = pd.DataFrame({
    'Test': ['AR(1): y.L1', 'Granger: F-test'],
    'p_value': [FOUR_to_FOUR, gradient_to_FOUR]
    })
    sns.scatterplot(data=plotting_df, x='Test', y='p_value', s=200, color='steelblue', ax=ax[row, col])
    ax[row, col].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    
    ax[row, col].set_title(f"Subject {index}")
    ax[row, col].set_ylim(0, 1)
    ax[row, col].set_xlim(-1, 2)
    ax[row, col].tick_params(axis='x', rotation=15)
    ax[row, col].margins(x=0.3)

    if row < 2:  # since 0-indexed rows → bottom row = row 2
        ax[row, col].set_xlabel("")
        ax[row, col].set_xticklabels([])
    
    if col > 0:
        ax[row, col].set_ylabel("")
        ax[row, col].set_yticklabels([])

fig.suptitle(f"Autoregressive comparison using {wakefullness_metric[13:]}", fontsize=18, fontweight="bold")
plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])
plt.savefig(f"granger_causality_results_{wakefullness_metric[13:]}_lag_{max_lag}.pdf")
print("test")