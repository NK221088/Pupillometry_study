import numpy as np
import pandas as pd

import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
path = Path(os.getenv(rf"Pouls_data_path"))
save_path = Path(os.getenv(rf"Pouls_save_path"))

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
    columns_to_keep = ['DateTime', 'PatientID', 'Pupil-Measured'] + list(df.columns[24:804])
    df = df[columns_to_keep]
    ts_cols = list(df.columns[3:393])
    len_before = len(df)
    df = df[df["PatientID"].astype(str).str.split(".").str[0].str.match(r"^\d+$")]
    df['PatientID'] = df['PatientID'].astype(int)
    df = df[df["PatientID"].astype(str).str.len() > 6]
    df["PatientID"] = df["PatientID"].astype(str).str.zfill(10)
    print(f"Discarded {len_before - len(df)} rows with non-numeric PatientID in file: {file.split('\\')[-1]}")

    len_before = len(df)
    df[ts_cols] = df[ts_cols].astype(float)
    df = df[df[ts_cols].iloc[:, :8].max(axis=1) > 0]
    print(f"Discarded {len_before - len(df)} rows due to all first 8 values being zero in file: {file.split('\\')[-1]}")

    df[ts_cols] = df[ts_cols].apply(interpolate_zeros, axis=1, result_type='expand').round(2)

    len_before = len(df)
    df = df.drop_duplicates(subset=['DateTime', 'PatientID', 'Pupil-Measured'], keep='first')
    print(f"Discarded {len_before - len(df)} duplicate rows in file: {file.split('\\')[-1]}")
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# Make sure all dates are same format. OBS. Some measurements have seconds, those are kept.
df_dates = df_all["DateTime"].str.replace("-", "/").str.split(" ")
df_all["DateTime"] = df_dates.str[0].str[:6] + df_dates.str[0].str.split("/").str[-1].str[-2:] + " " + df_all["DateTime"].str.replace("-", "/").str.split(" ").str[-1]

df_all_len_before = len(df_all)
df_all = df_all.drop_duplicates(subset=['DateTime', 'PatientID', 'Pupil-Measured'], keep='first')
print(f"Discarded {df_all_len_before - len(df_all)} duplicate rows across all files.")

df_all.to_excel(os.path.join(save_path, "Poul_data_cleaned.xlsx"), index=False)