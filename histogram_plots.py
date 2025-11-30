from read_data import left_data_original
from read_data import left_data_with_dates
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import pandas as pd

def Pearson_corrfunc(x, y, **kwargs):
    """Calculate and display correlation coefficient"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return
    
    r, p = stats.pearsonr(x[mask], y[mask])
    
    # Determine significance stars
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    ax = plt.gca()
    ax.annotate(f'r = {r:.2f}{sig}',
                xy=(0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center', fontsize=12)

def Spearman_corrfunc(x, y, **kwargs):
    """Calculate and display correlation coefficient"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return
    
    r, p = stats.spearmanr(x[mask], y[mask])
    
    # Determine significance stars (consistent with Pearson version)
    if p < 0.001:
        sig = '***'
    elif p < 0.01:
        sig = '**'
    elif p < 0.05:
        sig = '*'
    else:
        sig = 'ns'
    
    ax = plt.gca()
    ax.annotate(rf'$\rho$ = {r:.2f}{sig}',
                xy=(0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center', fontsize=12)

# Create the pairplot
columns_to_plot = ["Arousal gradient", "Max PLR", "LOR early gradient", "LOR late gradient", "FOUR scores", "SECONDS scores", "GCS scores"]

# Patient selection
selected_data = left_data_original[left_data_original["Under 3.5 mm."] == 1]


g = sns.PairGrid(selected_data[columns_to_plot], diag_sharey=False)

# Lower triangle: scatter plots with regression line
g.map_lower(sns.scatterplot, alpha=0.6)
g.map_lower(sns.regplot, scatter=False, color='red', line_kws={'linewidth': 1.5})

# Diagonal: histograms
g.map_diag(sns.histplot, kde=True)

# Upper triangle: correlation coefficients
g.map_upper(Spearman_corrfunc)

plt.tight_layout()

save_path_variance = os.getenv("save_path_variance")

plt.savefig(
    os.path.join(save_path_variance, f'Correlation_histograms.pdf'),
    dpi=300,                     
    bbox_inches='tight',
    format='pdf'
)

#######################################################################
# With one day delay

first_and_second_day_data = left_data_with_dates[left_data_with_dates["Day"].isin([1, 2])]
first_and_second_day_data['individuel_dates'] = pd.to_datetime(first_and_second_day_data['individuel_dates'])
first_and_second_day_data['date_diff'] = first_and_second_day_data.groupby('Subject ID')['individuel_dates'].diff().dt.days

# Filter for consecutive dates only (where date_diff == 1 for Day 2)
consecutive_only = first_and_second_day_data[
    (first_and_second_day_data['Day'] == 1) | 
    ((first_and_second_day_data['Day'] == 2) & (first_and_second_day_data['date_diff'] == 1))
]

# Pivot the data: columns from Day 1 and Day 2
day1_data = consecutive_only[consecutive_only['Day'] == 1].set_index('Subject ID')[columns_to_plot]
day2_data = consecutive_only[consecutive_only['Day'] == 2].set_index('Subject ID')[columns_to_plot]

# Rename columns to distinguish Day 1 vs Day 2
day1_data = day1_data.add_suffix('_Day1')
day2_data = day2_data.add_suffix('_Day2')

# Merge them side by side
plot_data = day1_data.join(day2_data, how='inner')

g = sns.PairGrid(plot_data, 
                 x_vars=[col for col in plot_data.columns if '_Day1' in col],
                 y_vars=[col for col in plot_data.columns if '_Day2' in col])

# Map all plots: scatter with regression line
g.map(sns.scatterplot, alpha=0.6)
g.map(sns.regplot, scatter=False, color='red', line_kws={'linewidth': 1.5})

# Add correlation coefficients to each subplot
for i, y_var in enumerate(g.y_vars):
    for j, x_var in enumerate(g.x_vars):
        ax = g.axes[i, j]
        x_data = plot_data[x_var].values
        y_data = plot_data[y_var].values
        
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

save_path_variance = os.getenv("save_path_variance")

plt.savefig(
    os.path.join(save_path_variance, f'Correlation_histograms_day_delay.pdf'),
    dpi=300,                     
    bbox_inches='tight',
    format='pdf'
)