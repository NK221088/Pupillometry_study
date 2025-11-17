from mixed_effect_approach import data
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd

save_path_variance = os.getenv("save_path_variance")
data.columns = data.columns.str.replace('_', ' ').str.replace(" scores", "")
target_correlation_matrix = data[data.columns[2:5]].corr()
features_correlation_matrix = data[data.columns[5:]].corr()
all_correlation_matrix = data[data.columns[2:]].corr()

fig, ax = plt.subplots(1, 3, figsize=(24, 16))
fig.suptitle('Correlation Matrix Heatmaps', fontsize=16, y=0.98)

# Set common scale for all heatmaps
vmin, vmax = -1, 1

# Create heatmaps without individual colorbars
sns.heatmap(target_correlation_matrix, annot=True, cmap='coolwarm', 
            fmt='.2f', linewidths=0.5, ax=ax[0], vmin=vmin, vmax=vmax, cbar=False)
ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30, ha='right', fontsize=8)
ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0, ha='right', fontsize=8)
ax[0].set_title('Target Variables')

sns.heatmap(features_correlation_matrix, annot=True, cmap='coolwarm', 
            fmt='.2f', linewidths=0.5, ax=ax[1], vmin=vmin, vmax=vmax, cbar=False)
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30, ha='right', fontsize=8)
ax[1].set_yticklabels(ax[1].get_yticklabels(), rotation=0, ha='right', fontsize=8)
ax[1].set_title('Features')

# Only the last heatmap gets a colorbar
im = sns.heatmap(all_correlation_matrix, annot=True, cmap='coolwarm', 
            fmt='.2f', linewidths=0.5, ax=ax[2], vmin=vmin, vmax=vmax, 
            cbar_kws={'label': 'Correlation'})
ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=30, ha='right', fontsize=8)
ax[2].set_yticklabels(ax[2].get_yticklabels(), rotation=0, ha='right', fontsize=8)
ax[2].set_title('All Variables')

# Adjust layout with more space on left and bottom
# plt.tight_layout(rect=[0.02, 0.01, 0.99, 0.02])
fig.subplots_adjust(
    left=0.04,    # Space on the left
    right=0.96,   # Space on the right
    top=0.92,     # Space at the top
    bottom=0.08,  # Space at the bottom
    wspace=0.25,   # Width space between subplots (horizontal spacing)
    hspace=0    # Height space between subplots (vertical spacing)
)
# plt.show()
plt.savefig(
    save_path_variance + f'\\Correlation_Matrix_Heatmaps.pdf',
    dpi=300,                     
    bbox_inches='tight',
    format='pdf'
)
plt.close(fig)

# Covariance analysis:

X_features = features_correlation_matrix
Sigma_features = np.cov(X_features.T)
eigenvalues_features, eigenvectors_features = np.linalg.eigh(Sigma_features)
results_df = pd.DataFrame(np.concatenate([eigenvalues_features.reshape(1,4), eigenvectors_features]), 
                          index=["Eigenvalues"] + list(data.columns[5:]),
                          columns=[f'PC{i+1}' for i in range(len(eigenvectors_features))])

# Plotting the PCs:
feature_names = X_features.columns.tolist()
variance_explained_features = eigenvalues_features / eigenvalues_features.sum() * 100
cumulative_variance_features = np.cumsum(variance_explained_features)
x_pos = np.arange(1, len(eigenvalues_features) + 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PCA Analysis Results', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
ax1.bar(x_pos, variance_explained_features, alpha=0.6, color='steelblue', label='Individual')
ax1.plot(x_pos, cumulative_variance_features, 'ro-', linewidth=2, markersize=8, label='Cumulative')
ax1.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
ax1.set_xlabel('Principal Component', fontsize=11)
ax1.set_ylabel('Variance Explained (%)', fontsize=11)
ax1.set_title('Scree Plot: Variance Explained by Each PC', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'PC{i}' for i in x_pos])
ax1.legend()
ax1.grid(alpha=0.3)

# Add percentage labels on bars
for i, v in enumerate(variance_explained_features):
    ax1.text(i + 1, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

# 2. Loading Plot (PC3 vs PC4 - the meaningful components)
ax2 = axes[0, 1]
pc3_loadings = eigenvectors_features[:, 2]
pc4_loadings = eigenvectors_features[:, 3]

ax2.scatter(pc3_loadings, pc4_loadings, s=100, alpha=0.6, color='coral')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel(f'PC3 ({variance_explained_features[2]:.1f}% variance)', fontsize=11)
ax2.set_ylabel(f'PC4 ({variance_explained_features[3]:.1f}% variance)', fontsize=11)
ax2.set_title('Loading Plot: PC3 vs PC4', fontweight='bold')
ax2.grid(alpha=0.3)

# Add feature labels with arrows
for i, feature in enumerate(feature_names):
    ax2.annotate(feature, 
                (pc3_loadings[i], pc4_loadings[i]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=1.5))

# 3. Heatmap of Loadings
ax3 = axes[1, 0]
loadings_df = pd.DataFrame(
    eigenvectors_features,
    index=feature_names,
    columns=[f'PC{i+1}' for i in range(4)]
)
sns.heatmap(loadings_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            ax=ax3, cbar_kws={'label': 'Loading Value'}, vmin=-1, vmax=1)
ax3.set_title('Feature Loadings Heatmap', fontweight='bold')
ax3.set_xlabel('Principal Component', fontsize=11)
ax3.set_ylabel('Feature', fontsize=11)

# 4. Eigenvalue Bar Plot
ax4 = axes[1, 1]
colors = ['lightcoral' if ev < 0.1 else 'steelblue' for ev in eigenvalues_features]
bars = ax4.bar(x_pos, eigenvalues_features, alpha=0.7, color=colors)
ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Kaiser criterion (λ=1)')
ax4.set_xlabel('Principal Component', fontsize=11)
ax4.set_ylabel('Eigenvalue (λ)', fontsize=11)
ax4.set_title('Eigenvalues by Component', fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([f'PC{i}' for i in x_pos])
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

# Add value labels
for i, (bar, ev) in enumerate(zip(bars, eigenvalues_features)):
    ax4.text(bar.get_x() + bar.get_width()/2, ev + 0.03, 
             f'{ev:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()