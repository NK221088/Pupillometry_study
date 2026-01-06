from dates import NPI_data_cleaned
from read_data import patient_left_data, patient_right_data, patient_left_numeric_data, patient_right_numeric_data, patient_left_text_data, patient_right_text_data
import pandas as pd
import matplotlib.pyplot as plt
import os

patient_left_data = {
    i: value
    for i, value in enumerate(patient_left_data.values(), start=1)
}
patient_left_numeric_data = {
    i: value
    for i, value in enumerate(patient_left_numeric_data.values(), start=1)
}
patient_left_text_data = {
    i: value
    for i, value in enumerate(patient_left_text_data.values(), start=1)
}

# Clinical metrics:atient_right_etiology = {sheet_name: patient_right_text_data[sheet_name].loc["Etiology"] for sheet_name in patient_right_data.keys()}
import re

patient_left_etiology_metrics = {
    patient_id: list(map(int, re.findall(r"\d+", str(
        patient_left_data[list(patient_left_data.keys())[0]][patient_id]
        .loc["Etiology"]
    ))))
    for patient_id in patient_left_data[list(patient_left_data.keys())[0]].columns
}

patient_left_consciousness_metrics = {
    patient_id: patient_left_data[list(patient_left_data.keys())[0]][patient_id].loc["SECONDS"]
    for patient_id in patient_left_data[list(patient_left_data.keys())[0]].columns
}

patient_left_sedation_metrics = {
    patient_id: patient_left_data[list(patient_left_data.keys())[0]][patient_id].loc["Sedation"]
    for patient_id in patient_left_data[list(patient_left_data.keys())[0]].columns
}


patient_left_raw_values = {sheet_name: patient_left_numeric_data[sheet_name] for sheet_name in patient_left_data.keys()}

all_patient_ids = list(patient_left_data[1].keys())

patient_left_individual_raw_data = {patient_id: pd.concat([
    patient_left_raw_values[sheet_name][patient_id] if patient_id in patient_left_raw_values[sheet_name].columns else pd.Series(dtype='float64')
    for sheet_name in patient_left_data.keys()
    ], axis=1, keys=patient_left_data.keys()) for patient_id in all_patient_ids}

patient_left_individual_text_data = {patient_id: pd.concat([
    patient_left_text_data[sheet_name][patient_id] if patient_id in patient_left_text_data[sheet_name].columns else pd.Series(dtype='float64')
    for sheet_name in patient_left_text_data.keys()
    ], axis=1, keys=patient_left_text_data.keys()) for patient_id in all_patient_ids}

patient_left_individual_text_data = {patient_id: pd.concat([
    patient_left_text_data[sheet_name][patient_id] if patient_id in patient_left_text_data[sheet_name].columns else pd.Series(dtype='float64')
    for sheet_name in patient_left_text_data.keys()
    ], axis=1, keys=patient_left_text_data.keys()) for patient_id in all_patient_ids}

for patient_id in all_patient_ids:
    visit_order = (
        NPI_data_cleaned[NPI_data_cleaned["record_id"] == patient_id]
        .sort_values("date_examination_merged")
        ["redcap_repeat_instance"]
        .tolist()
    )

    patient_left_individual_raw_data[patient_id] = (
        patient_left_individual_raw_data[patient_id]
        .loc[:, visit_order]
    )

    patient_left_individual_text_data[patient_id] = (
        patient_left_individual_text_data[patient_id]
        .loc[:, visit_order]
    )

etiology_coding = {
    0: "Cardiac cause",
    1: "cerebrovaskulær",
    2: "cerebrovaskulær",
    3: "cerebrovaskulær",
    4: "cerebrovaskulær",
    5: "TBI",
    8: "Other",
    9: "Other",
    10: "Other",
    13: "Other",
    14: "Other",
    15: "Other",
    16: "Other",
    20: "Other",
    21: "Other",
}

etiology_groups = {
    "Cardiac cause" :  [],
    "cerebrovaskulær" :  [],
    "TBI" :  [],
    "Other"  :  [],
}

for patient_id in patient_left_etiology_metrics:
    etiology_codes = patient_left_etiology_metrics[patient_id]
    etiology_groups[etiology_coding[int(etiology_codes[0])]].append(patient_id)

etiology_data = {k: pd.DataFrame() for k in etiology_groups}

for eti_state, patient_ids in etiology_groups.items():
    patient_series = []

    for patient_id in patient_ids:
        df = patient_left_individual_raw_data.get(patient_id)

        if df is None or df.shape[1] == 0:
            continue

        # safer if visit 1 is explicit
        if 1 not in df.columns:
            continue

        first_visit = df.loc[:, 1]
        first_visit.name = patient_id
        patient_series.append(first_visit)

    if patient_series:
        etiology_data[eti_state] = pd.concat(patient_series, axis=1)

group_colors = {
    "Cardiac cause": "#FFD700",
    "cerebrovaskulær": "tab:blue",
    "TBI": "tab:orange",
    "Other": "tab:purple",
}

plt.figure(figsize=(10, 6))

for eti_state, df in etiology_data.items():
    if df.empty:
        continue

    color = group_colors.get(eti_state, "gray")

    for patient_id in df.columns:
        plt.plot(
            df.index,
            df[patient_id],
            color=color,
            alpha=0.3,
            linewidth=1
        )

# legend
for eti_state, color in group_colors.items():
    plt.plot([], [], color=color, label=eti_state)

plt.xlabel("Time (s)")
plt.ylabel("Pupil size")
plt.title("Individual pupil responses (first visit)")
plt.legend(title="Etiology group")
plt.tight_layout()

fig_path = os.path.join(
    r"L:\AuditData\CONNECT-ME\Nikolai\pupillometry\Plots",
    "etiology_group_pupil_responses.pdf"
)
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

consciousness_coding = {
    'C': "Coma",
    'E': "eMCS",
    'M+': "MCS",
    'M-': "MCS",
    "M": "MCS",
    'U': "UWS",
}

consciousness_groups = {
    "Coma" :  [],
    "UWS" :  [],
    "MCS" :  [],
    "eMCS"  :  [],
}

for patient_id in patient_left_consciousness_metrics :
    etiology_codes = patient_left_consciousness_metrics [patient_id]
    consciousness_groups[consciousness_coding[etiology_codes[0]]].append(patient_id)

# sedation_coding = {
#     'C': "Coma",
#     'E': "eMCS",
#     'M+': "MCS",
#     'M-': "MCS",
#     "M": "MCS",
#     'U': "UWS",
# }

# sedation_groups = {
#     "Coma" :  [],
#     "UWS" :  [],
#     "MCS" :  [],
#     "eMCS"  :  [],
# }

# for patient_id in patient_left_sedation_metrics :
#     etiology_codes = patient_left_sedation_metrics [patient_id]
#     sedation_groups[sedation_coding[etiology_codes[0]]].append(patient_id)


consciousness_data = {
    "Coma": pd.DataFrame(),
    "UWS": pd.DataFrame(),
    "MCS": pd.DataFrame(),
    "eMCS": pd.DataFrame(),
}

for con_state, patient_ids in consciousness_groups.items():
    patient_series = []

    for patient_id in patient_ids:
        df = patient_left_individual_raw_data.get(patient_id)

        # skip if no data or no visits
        if df is None or df.shape[1] == 0:
            continue

        # extract FIRST visit
        first_visit = df.iloc[:, 0]
        first_visit.name = patient_id

        patient_series.append(first_visit)

    # only concatenate if we actually collected something
    if patient_series:
        consciousness_data[con_state] = pd.concat(
            patient_series,
            axis=1
        )

group_colors = {
    "Coma": "yellow",
    "UWS": "tab:blue",
    "MCS": "tab:green",
    "eMCS": "tab:orange",
}

plt.figure(figsize=(10, 6))

for con_state, df in consciousness_data.items():
    if df.empty:
        continue

    color = group_colors.get(con_state, "gray")

    for patient_id in df.columns:
        plt.plot(
            df.index,
            df[patient_id],
            color=color,
            alpha=0.3,      # transparency for overlap
            linewidth=1
        )

# legend (one entry per group)
for con_state, color in group_colors.items():
    plt.plot([], [], color=color, label=con_state)

plt.xlabel("Time (s)")
plt.ylabel("Pupil  size")
plt.title("Individual pupil responses (first visit)")
plt.legend(title="Consciousness group")
plt.tight_layout()
plt.show()
fig_path = os.path.join(rf"L:\AuditData\CONNECT-ME\Nikolai\pupillometry\Plots", "consciousness_group_pupil_responses.pdf")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")