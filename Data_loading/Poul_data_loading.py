import numpy as np
import pandas as pd

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
path = Path(os.getenv(rf"Pouls_data_path"))

def interpolate_zeros(row):
    s = row.copy().values.astype(float)
    artifact = np.zeros(len(s), dtype=bool)
    for i in range(len(s)):
        if s[i] == 0:
            artifact[i] = True
        elif 0 < i < len(s) - 1 and s[i-1] == 0 and s[i+1] == 0:
            artifact[i] = True
    result = s.copy()
    result[artifact] = np.nan
    return pd.Series(result).interpolate(method='linear').values

all_files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.startswith("R")]

dfs = []
for file in all_files:
    print(f"Processing: {file}")
    df = pd.read_csv(file, sep='\t', skiprows=1, low_memory=False)
    df = df[df["Protocol-Type"] == "PLR-Positive"]
    columns_to_keep = ['DateTime', 'PatientID', 'Pupil-Measured'] + list(df.columns[24:803])
    df = df[columns_to_keep]
    ts_cols = list(df.columns[3:])
    df['PatientID'] = df['PatientID'].astype(int)
    df[ts_cols] = df[ts_cols].astype(float)
    df[ts_cols] = df[ts_cols].apply(interpolate_zeros, axis=1, result_type='expand')
    df = df.drop_duplicates(subset=['DateTime', 'PatientID', 'Pupil-Measured'], keep='first')
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.drop_duplicates(subset=['DateTime', 'PatientID', 'Pupil-Measured'], keep='first')
print("debug")