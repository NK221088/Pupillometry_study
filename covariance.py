from mixed_effect_approach import data
import numpy as np
import matplotlib.pyplot as plt

# Compute correlation matrix
corr = data[data.columns[4:]].corr()

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(corr, vmin=-1, vmax=1)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)

fig.colorbar(cax)
plt.tight_layout()
plt.show()

