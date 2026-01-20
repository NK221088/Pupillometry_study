import numpy as np

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