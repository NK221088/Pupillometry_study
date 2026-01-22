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

from Time_series_analysis.feature_extraction import left_metrics_df, df_outcome

print("debug")
X = (
    left_metrics_df
    .query("redcap_repeat_instance == 1")
    .drop(columns=["redcap_repeat_instance", "date_examination_merged"])
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

pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "logreg",
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,          # balance L1 / L2
                C=1.0,                 # inverse regularization strength
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            ),
        ),
    ]
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "roc_auc": "roc_auc",
    "auprc": "average_precision",
}


results = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
)

print("Mean AUROC:", results["test_roc_auc"].mean())
print("Mean AUPRC:", results["test_auprc"].mean())

pipeline.fit(X, y)

feature_names = X.columns
coefficients = pipeline.named_steps["logreg"].coef_[0]

coef_df = (
    pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )
    .sort_values("coefficient", key=np.abs, ascending=False)
)

print(coef_df)
