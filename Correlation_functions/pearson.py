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