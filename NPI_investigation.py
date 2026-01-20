from pearson import Pearson_corrfunc
from spearman import Spearman_corrfunc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

NPi_measurements_data_path = os.getenv("NPI_measurements_data_path")
# NPI_data = pd.read_csv(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Arbejde_Rigshospitalet\Pupillometry\NPI_investigation\NPI_measurements.csv")
NPI_data = pd.read_csv(NPi_measurements_data_path)
merging_columns = [
    "date_examination",
    "light_off_performed",
    "npi_left",
    "npi_right",
    "pupil_size_left",
    "pupil_size_right",
    "pupil_min_left",
    "pupil_min_right",
    "ch_left",
    "ch_right",
    "const_velocity_left",
    "const_velocity_right",
    "max_const_velocity_left",
    "max_const_velocity_right",
    "latency_left",
    "latency_right",
    "dilat_velocity_left",
    "dilat_velocity_right",
]
for column in merging_columns:
    if column == "date_examination":
        NPI_data[f"{column}_merged"] = (
            NPI_data[f"{column}"]
            .combine_first(NPI_data[f"date_of_examination_2"])
        )
    else:
        NPI_data[f"{column}_merged"] = (
            NPI_data[f"{column}"]
            .combine_first(NPI_data[f"{column}_2"])
        )

columns_to_keep = ["record_id", "redcap_repeat_instance"] + [f"{col}_merged" for col in merging_columns]
NPI_data_cleaned = NPI_data[columns_to_keep] # Remove suffix from light_off_performed_merged column
NPI_data_cleaned["light_off_performed_merged"] = (
    NPI_data_cleaned["light_off_performed_merged"]
    .map({1.0: "Yes", 0.0: "No"})
)

NPI_data_cleaned["redcap_repeat_instance"] = ( # Ensure first recording is marked as 0
    NPI_data_cleaned["redcap_repeat_instance"]
    .fillna(0)
)
mask = NPI_data_cleaned["record_id"] == 213
cols = ["npi_left_merged"]

NPI_data_cleaned.loc[mask, cols] = (
    NPI_data_cleaned.loc[mask, cols].replace(0, np.nan)
)
NPI_data_cleaned["redcap_repeat_instance"]  += 1 # Shift follow-up numbers to start from 1
NPI_data_cleaned = NPI_data_cleaned[
    NPI_data_cleaned["light_off_performed_merged"] == "Yes"
]
NPI_data_cleaned = (
    NPI_data_cleaned
    .sort_values(
        by=["record_id", "redcap_repeat_instance"],
        ascending=[True, True]
    )
)
NPI_data_cleaned["redcap_repeat_instance"] = (
    NPI_data_cleaned
    .groupby("record_id")
    .cumcount()
    .add(1)
)

NPI_distribution_plots_path = os.getenv("NPI_distribution_plots_path")
# Create the pairplot
columns_to_plot = [
    c for c in columns_to_keep
    if c not in [
        'record_id',
        'redcap_repeat_instance',
        'date_examination_merged',
        'light_off_performed_merged'
    ]
]
NPI_data_cleaned = NPI_data_cleaned.drop(
    NPI_data_cleaned.index[120] 
)
left_columns = ["record_id", "redcap_repeat_instance"] + [col for col in NPI_data_cleaned.columns if '_left' in col]
right_columns = ["record_id", "redcap_repeat_instance"] + [col for col in NPI_data_cleaned.columns if '_right' in col]
left_NPI_data_cleaned = NPI_data_cleaned.copy()[left_columns]
right_NPI_data_cleaned = NPI_data_cleaned.copy()[right_columns]
left_NPI_data_cleaned.columns = left_NPI_data_cleaned.columns.str.replace('_left_merged', '').str.replace('_', ' ').str.replace("npi", "NPi")
right_NPI_data_cleaned.columns = right_NPI_data_cleaned.columns.str.replace('_right_merged', '').str.replace('_', ' ').str.replace("npi", "NPi")

g = sns.PairGrid(left_NPI_data_cleaned, 
                 x_vars=left_NPI_data_cleaned.columns[2:],
                 y_vars=left_NPI_data_cleaned.columns[2:])

# Map all plots: scatter with regression line
g.map(sns.scatterplot, alpha=0.6)
g.map(sns.regplot, scatter=False, color='red', line_kws={'linewidth': 1.5})

# Add correlation coefficients to each subplot
for i, y_var in enumerate(g.y_vars):
    for j, x_var in enumerate(g.x_vars):
        ax = g.axes[i, j]
        x_data = left_NPI_data_cleaned[x_var].values
        y_data = left_NPI_data_cleaned[y_var].values
        
        # Call your correlation function
        mask = ~np.isnan(x_data) & ~np.isnan(y_data)
        if mask.sum() >= 2:
            r, p = stats.spearmanr(x_data[mask], y_data[mask])
            
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            
            ax.text(0.05, 0.95, rf'$\rho$ = {r:.2f}{sig}',
                   transform=ax.transAxes,
                   ha='left', va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

plt.savefig(
    os.path.join(NPI_distribution_plots_path, f'Correlation_histograms_NPi.pdf'),
    dpi=600,                     
    bbox_inches='tight',
    format='pdf'
)
plt.close()