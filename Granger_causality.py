from read_data import (patient_left_FOUR_metrics, patient_left_first_50_metrics, patient_left_second_50_metrics, patient_left_LOR_late_gradient_metrics)
import nitime.timeseries as ts
import nitime.analysis as nta
import numpy as np

patient_left_scores = {}
for index in patient_left_FOUR_metrics.index:
    patient_left_scores[index] = np.vstack([patient_left_FOUR_metrics.loc[index].dropna(), patient_left_LOR_late_gradient_metrics.loc[index].dropna()])
    patient_left_scores[index] = patient_left_scores[index].astype(np.float64)

granger_results = {}
for subject, subject_scores in patient_left_scores.items():
    
    if subject_scores.shape[1] < 3:
        print(f"Skipping {subject}: Insufficient data points.  ({subject_scores.shape[1]})")
        continue
    # Check for zero variance:
    var0 = np.var(subject_scores[0])
    var1 = np.var(subject_scores[1])
    
    if var0 == 0 or var1 == 0:
        print(f"Skipping {subject}: Zero variance for at least one parameter")
        
    time_series = ts.TimeSeries(subject_scores, sampling_rate=1/86400)
    granger = nta.GrangerAnalyzer(time_series)
    granger_results[subject] = granger.causality_xy

print("test")