from read_data import left_data_original
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

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
    ax.annotate(f'ρ = {r:.2f}{sig}',
                xy=(0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center', fontsize=12)

# Create the pairplot
diagnostic_metric = "SECONDS"
columns_to_plot = ["Arousal gradient", "Max PLR", "LOR early gradient", "LOR late gradient", f"{diagnostic_metric} scores"]

g = sns.PairGrid(left_data_original[columns_to_plot], diag_sharey=False)

# Lower triangle: scatter plots
g.map_lower(sns.scatterplot, alpha=0.6)

# Diagonal: histograms
g.map_diag(sns.histplot, kde=True)

# Upper triangle: correlation coefficients
g.map_upper(Spearman_corrfunc)

plt.tight_layout()

save_path_variance = os.getenv("save_path_variance")

plt.savefig(
    os.path.join(save_path_variance, f'Correlation_histograms_{diagnostic_metric}_scores.pdf'),
    dpi=300,                     
    bbox_inches='tight',
    format='pdf'
)