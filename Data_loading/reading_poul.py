import pandas as pd

data = pd.read_excel(r"L:\Auditdata\CONNECT-ME\Nikolai\pupillometry\Data\Pouls_data\test.xltx", header=1)
PLR_data = data[data["Protocol-Type"] == "PLR-Positive"]
num_data_points = 390
first_indice = 24
time_stamps = PLR_data[PLR_data["RecordID"] == PLR_data["RecordID"].values[0]].values[0][num_data_points+first_indice:num_data_points+first_indice+num_data_points]
columns_to_keep = ["DateTime", "PatientID", "Pupil-Measured"] + list(data.columns[first_indice:num_data_points+first_indice])
PLR_data = PLR_data[columns_to_keep]
PLR_data.columns = ["DateTime", "PatientID", "Pupil-Measured"] + list(time_stamps)
print("debug")