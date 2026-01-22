import sys
import os

# Add the project root (one folder up) to Python's module search path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
from Data_loading.feature_extraction import left_metrics_df, df_outcome

X = (
    left_metrics_df
    .query("redcap_repeat_instance == 1")
    .drop(columns=["redcap_repeat_instance", "date_examination_merged", "left_SECONDS"])
    .dropna()
)
y = (
    df_outcome
    .dropna(subset=["ICU_outcome"])
    .assign(outcome=lambda d: (d["ICU_outcome"] == "D").astype(int))
)

df_model = (
    X.merge(
        y[["record_id", "outcome"]],
        on="record_id",
        how="inner",
        validate="one_to_one",
    )
)

X = df_model.drop(columns=["record_id", "outcome"])
y = df_model["outcome"]

print("debug")