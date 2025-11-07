import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


from read_data import (patient_left_FOUR_metrics, patient_left_LOR_late_gradient_metrics)

subject_IDs = [[subjectID] * len(patient_left_LOR_late_gradient_metrics.loc[subjectID].dropna()) for subjectID in patient_left_LOR_late_gradient_metrics.index]
days = np.concatenate([np.arange(1, len(IDs)+1, 1) for IDs in subject_IDs]).flatten()
subject_IDs = np.concatenate(subject_IDs)
FOUR_scores = np.concatenate([patient_left_FOUR_metrics.loc[index].dropna().values for index in patient_left_FOUR_metrics.index]).astype(np.float64)
LOR_late_gradient_scores = np.concatenate([patient_left_LOR_late_gradient_metrics.loc[index].dropna().values for index in patient_left_LOR_late_gradient_metrics.index]).astype(np.float64)

data = {
    "subject_id": subject_IDs,
    "day": days,
    "four_score": FOUR_scores,
    "LOR_late_gradient_score": LOR_late_gradient_scores
}


data_original = pd.DataFrame(data)


max_lag = 7

for lag in range(1, max_lag+1):
    data = data_original.copy()
    data = data.sort_values(["subject_id", "day"])
    data["LOR_late_gradient_score_lag1"] = data.groupby("subject_id")["LOR_late_gradient_score"].shift(lag)
    data["four_score_lag1"] = data.groupby("subject_id")["four_score"].shift(lag)
    data["day_centered"] = data["day"] - data.groupby("subject_id")["day"].transform("min")
    df_lagged = data.dropna(subset=["LOR_late_gradient_score_lag1", "four_score_lag1"])


    model_ar = smf.mixedlm("four_score ~ four_score_lag1 + day_centered",
                        data=df_lagged,
                        groups=df_lagged["subject_id"])
    result_ar = model_ar.fit(reml=False)

    model_full = smf.mixedlm("four_score ~ four_score_lag1 + LOR_late_gradient_score_lag1 + day_centered",
                            data=df_lagged,
                            groups=df_lagged["subject_id"])
    result_full = model_full.fit(reml=False)

    lr_stat = -2 * (result_ar.llf - result_full.llf)
    p_value = stats.chi2.sf(lr_stat, 1)

    print(f"Lag: {lag}")
    print("result_ar converged:", result_ar.converged)
    print("result_full converged:", result_full.converged)
    print(f"p_value: {np.round(p_value,4)}")
    print(f"LR stat: {np.round(lr_stat, 4)}")