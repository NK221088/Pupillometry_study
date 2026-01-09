from dates import NPI_data_cleaned
from read_data import patient_left_data, patient_right_data, patient_left_numeric_data, patient_right_numeric_data, patient_left_text_data, patient_right_text_data, HC_left_data,  HC_left_text_data, HC_left_numeric_data
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from matplotlib.ticker import MaxNLocator


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

patient_right_data = {
    i: value
    for i, value in enumerate(patient_right_data.values(), start=1)
}
patient_right_numeric_data = {
    i: value
    for i, value in enumerate(patient_right_numeric_data.values(), start=1)
}
patient_right_text_data = {
    i: value
    for i, value in enumerate(patient_right_text_data.values(), start=1)
}

# Clinical metrics:atient_right_etiology = {sheet_name: patient_right_text_data[sheet_name].loc["Etiology"] for sheet_name in patient_right_data.keys()}

patient_left_etiology_metrics = {
    patient_id: list(map(int, re.findall(r"\d+", str(
        patient_left_data[list(patient_left_data.keys())[0]][patient_id]
        .loc["Etiology"]
    ))))
    for patient_id in patient_left_data[list(patient_left_data.keys())[0]].columns
}

patient_left_sedation_metrics = {
    day:
    {patient_id: str(patient_left_data[day][patient_id].loc["Sedation"]).split(",")
    for patient_id in patient_left_data[day].columns}
    for day in patient_left_data.keys()
}

patient_left_consciousness_metrics = {
    day:
    {patient_id: patient_left_data[day][patient_id].loc["SECONDS"]
    for patient_id in patient_left_data[day].columns}
    for day in patient_left_data.keys()
}

patient_right_etiology_metrics = {
    patient_id: list(map(int, re.findall(r"\d+", str(
        patient_right_data[list(patient_right_data.keys())[0]][patient_id]
        .loc["Etiology"]
    ))))
    for patient_id in patient_right_data[list(patient_right_data.keys())[0]].columns
}

patient_right_sedation_metrics = {
    day:
    {patient_id: str(patient_right_data[day][patient_id].loc["Sedation"]).split(",")
    for patient_id in patient_right_data[day].columns}
    for day in patient_right_data.keys()
}

patient_right_consciousness_metrics = {
    day:
    {patient_id: patient_right_data[day][patient_id].loc["SECONDS"]
    for patient_id in patient_right_data[day].columns}
    for day in patient_right_data.keys()
}

patient_left_raw_values = {sheet_name: patient_left_numeric_data[sheet_name] for sheet_name in patient_left_data.keys()}
patient_right_raw_values = {sheet_name: patient_right_numeric_data[sheet_name] for sheet_name in patient_right_data.keys()}


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

patient_right_individual_raw_data = {patient_id: pd.concat([
    patient_right_raw_values[sheet_name][patient_id] if patient_id in patient_right_raw_values[sheet_name].columns else pd.Series(dtype='float64')
    for sheet_name in patient_right_data.keys()
    ], axis=1, keys=patient_right_data.keys()) for patient_id in all_patient_ids}

patient_right_individual_text_data = {patient_id: pd.concat([
    patient_right_text_data[sheet_name][patient_id] if patient_id in patient_right_text_data[sheet_name].columns else pd.Series(dtype='float64')
    for sheet_name in patient_right_text_data.keys()
    ], axis=1, keys=patient_right_text_data.keys()) for patient_id in all_patient_ids}

patient_right_individual_text_data = {patient_id: pd.concat([
    patient_right_text_data[sheet_name][patient_id] if patient_id in patient_right_text_data[sheet_name].columns else pd.Series(dtype='float64')
    for sheet_name in patient_right_text_data.keys()
    ], axis=1, keys=patient_right_text_data.keys()) for patient_id in all_patient_ids}

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

    patient_right_individual_raw_data[patient_id] = (
        patient_right_individual_raw_data[patient_id]
        .loc[:, visit_order]
    )

    patient_right_individual_text_data[patient_id] = (
        patient_right_individual_text_data[patient_id]
        .loc[:, visit_order]
    )
    
    
from collections import defaultdict
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import os
import pandas as pd

save_path_time = os.getenv("save_path_time_left")

# --------------------------------------------------
# Consciousness coding & colors
# --------------------------------------------------

consciousness_coding = {
    "C": "Coma",
    "E": "eMCS",
    "M+": "MCS+",
    "M-": "MCS-",
    "U": "UWS",
}

consciousness_colors = {
    "Coma":   "black",
    "UWS":    "lightgrey",
    "MCS-":   "orange",
    "MCS+":   "yellow",
    "eMCS":   "green",
}


# --------------------------------------------------
# Helper: assign consciousness state to NPI dataframe
# --------------------------------------------------

def assign_consciousness_state(
    NPI_df,
    raw_values,
    individual_raw_data,
    consciousness_metrics,
):
    consciousness_data_per_day = defaultdict(dict)

    for day in sorted(raw_values.keys()):
        consciousness_groups = {
            "Coma": [],
            "UWS": [],
            "MCS+": [],
            "MCS-": [],
            "eMCS": [],
        }

        day_metrics = consciousness_metrics.get(day, {})
        for patient_id, codes in day_metrics.items():
            if not codes:
                continue
            state = consciousness_coding.get(codes)
            if state:
                consciousness_groups[state].append(patient_id)

        for state, patient_ids in consciousness_groups.items():
            patient_series = []
            for pid in patient_ids:
                df = individual_raw_data.get(pid)
                if df is None or day not in df.columns:
                    continue
                s = df.loc[:, day]
                s.name = pid
                patient_series.append(s)

            if patient_series:
                consciousness_data_per_day[day][state] = pd.concat(
                    patient_series, axis=1
                )

    # write state labels into NPI_df
    NPI_df["SECONDs"] = pd.NA

    for day, day_dict in consciousness_data_per_day.items():
        for state, df in day_dict.items():
            mask = (
                (NPI_df["redcap_repeat_instance"] == day)
                & (NPI_df["record_id"].isin(df.columns))
            )
            NPI_df.loc[mask, "SECONDs"] = state

    return NPI_df


# --------------------------------------------------
# Helper: plot one NPI panel
# --------------------------------------------------

def plot_npi_panel(ax, df, side_label):
    ax.scatter(
        df["redcap_repeat_instance"],
        df[f"npi_{side_label}_merged"],
        c=df["SECONDs"].map(consciousness_colors),
        s=30,
        alpha=1,
        edgecolor="black",
        linewidth=0.3,
        zorder=3,
    )

    # clinical threshold
    ax.axhline(
        y=3,
        color="black",
        linestyle="--",
        linewidth=1,
        zorder=2,
    )

    # axis styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    ax.set_ylabel("NPI value", fontsize=13)
    ax.set_title(f"{side_label.capitalize()} Eye", fontsize=13)


# --------------------------------------------------
# Assign states for LEFT and RIGHT
# --------------------------------------------------

NPI_left = assign_consciousness_state(
    NPI_data_cleaned.copy(),
    patient_left_raw_values,
    patient_left_individual_raw_data,
    patient_left_consciousness_metrics,
)

NPI_right = assign_consciousness_state(
    NPI_data_cleaned.copy(),
    patient_right_raw_values,
    patient_right_individual_raw_data,
    patient_right_consciousness_metrics,
)

# --------------------------------------------------
# Plot: 2 rows × 1 column (Left / Right)
# --------------------------------------------------

NPI_left.to_csv(os.path.join(save_path_time, "NPI_left_with_states.csv"), index=False)
NPI_right.to_csv(os.path.join(save_path_time, "NPI_right_with_states.csv"), index=False)

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(6, 7),
    sharex=True,
)

plot_npi_panel(axes[0], NPI_left, "left")
plot_npi_panel(axes[1], NPI_right, "right")

# shared x-axis
axes[1].xaxis.set_major_locator(MultipleLocator(5))
axes[1].set_xlabel("Day", fontsize=13, labelpad=12)

# --------------------------------------------------
# Figure-level legend
# --------------------------------------------------

legend_handles = [
    plt.Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markerfacecolor=color,
        markeredgecolor="black",
        markersize=6,
        label=group,
    )
    for group, color in consciousness_colors.items()
]

fig.legend(
    handles=legend_handles,
    fontsize=11,
    title_fontsize=12,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=len(consciousness_colors),
    labelspacing=0.8,
    handletextpad=0.6,
)

# layout
fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.96])
fig.subplots_adjust(hspace=0.40)

fig_path = os.path.join(
    save_path_time,
    f"NPI_group_responses.jpg"
)
fig.savefig(fig_path, dpi=600, bbox_inches="tight")
plt.close(fig)

# # save
# fig_path = os.path.join(
#     os.getenv("save_path_time"),
#     "NPI_group_responses_left_right.jpg",
# )
# fig.savefig(fig_path, dpi=600, bbox_inches="tight")

"""
Left = False
if Left:
    etiology_metrics = patient_left_etiology_metrics
    sedation_metrics = patient_left_sedation_metrics
    consciousness_metrics = patient_left_consciousness_metrics
    raw_values = patient_left_raw_values
    individual_raw_data = patient_left_individual_raw_data
    save_path_time = os.getenv("save_path_time_left")
else:
    etiology_metrics = patient_right_etiology_metrics
    sedation_metrics = patient_right_sedation_metrics
    consciousness_metrics = patient_right_consciousness_metrics
    raw_values = patient_right_raw_values
    individual_raw_data = patient_right_individual_raw_data
    save_path_time = os.getenv("save_path_time_right")

consciousness_coding = {
    "C": "Coma",
    "E": "eMCS",
    "M+": "MCS+",
    "M-": "MCS-",
    "U": "UWS",
}

consciousness_colors = {
    "Coma":   "#4D4D4D",   # dark grey — baseline / deepest impairment
    "UWS":    "#0072B2",   # strong blue
    "MCS-":   "#009E73",   # bluish green
    "MCS+":   "#D55E00",   # vermillion
    "eMCS":   "#CC79A7",   # reddish purple
}


# --------------------------------------------------
# Collect data per day
# --------------------------------------------------

consciousness_data_per_day = defaultdict(dict)

for day in sorted(raw_values.keys()):

    consciousness_groups = {
        "Coma": [],
        "UWS": [],
        "MCS+": [],
        "MCS-": [],
        "eMCS": [],
    }

    day_consciousness_metrics = consciousness_metrics.get(day, {})
    for patient_id, codes in day_consciousness_metrics.items():
        if not codes:
            continue
        state = consciousness_coding.get(codes)
        if state is not None:
            consciousness_groups[state].append(patient_id)

    for con_state, patient_ids in consciousness_groups.items():
        patient_series = []

        for patient_id in patient_ids:
            df = individual_raw_data.get(patient_id)

            if df is None or day not in df.columns:
                continue

            series = df.loc[:, day]
            series.name = patient_id
            patient_series.append(series)

        if patient_series:
            consciousness_data_per_day[day][con_state] = pd.concat(
                patient_series, axis=1
            )

from matplotlib.ticker import MultipleLocator

NPI_data_cleaned["SECONDs"] = pd.NA

for day, day_dict in consciousness_data_per_day.items():
    for state, df in day_dict.items():
        patient_ids = df.columns

        mask = (
            (NPI_data_cleaned["redcap_repeat_instance"] == day) &
            (NPI_data_cleaned["record_id"].isin(patient_ids))
        )

        NPI_data_cleaned.loc[mask, "SECONDs"] = state


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(
    NPI_data_cleaned["redcap_repeat_instance"],
    NPI_data_cleaned[f"npi_{save_path_time.split(os.sep)[-1].lower()}_merged"],
    c=NPI_data_cleaned["SECONDs"].map(consciousness_colors),
    s=30,
    alpha=1,
    edgecolor="black",
    linewidth=0.3,
    zorder=3,
)

# Clinical threshold
ax.axhline(
    y=3,
    color="black",
    linestyle="--",
    linewidth=1,
    alpha=1,
    zorder=2,
)

# Axes formatting
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=11)

ax.set_xlabel("Day", fontsize=13, labelpad=12)
ax.set_ylabel("NPI value", fontsize=13)


# --------------------------------------------------
# Figure-level legend (outside plot)
# --------------------------------------------------

legend_handles = [
    plt.Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markerfacecolor=color,
        markeredgecolor="black",
        markersize=6,
        label=group,
    )
    for group, color in consciousness_colors.items()
]

fig.legend(
    handles=legend_handles,
    title="Consciousness group\n",
    fontsize=11,
    title_fontsize=12,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=len(consciousness_colors),
    labelspacing=0.8,
    handletextpad=0.6,
)

# Reserve space for legend
fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.85])

# Save
fig_path = os.path.join(
    save_path_time,
    f"NPI_group_responses_{save_path_time.split(os.sep)[-1]}.jpg"
)
fig.savefig(fig_path, dpi=600, bbox_inches="tight")
plt.close(fig)
    

# --------------------------------------------------
# Etiology coding
# --------------------------------------------------

# etiology_coding = {
#     0: "Cardiac cause",
#     1: "cerebrovaskulær",
#     2: "cerebrovaskulær",
#     3: "cerebrovaskulær",
#     4: "cerebrovaskulær",
#     5: "TBI",
#     8: "Other",
#     9: "Other",
#     10: "Other",
#     13: "Other",
#     14: "Other",
#     15: "Other",
#     16: "Other",
#     20: "Other",
#     21: "Other",
# }

# group_colors = {
#     "Cardiac cause": "#FFD700",
#     "cerebrovaskulær": "tab:blue",
#     "TBI": "tab:orange",
#     "Other": "tab:purple",
# }

# # --------------------------------------------------
# # Collect data per day
# # --------------------------------------------------

# etiology_data_per_day = defaultdict(dict)

# for day in sorted(raw_values.keys()):

#     # group patients by etiology (same across days)
#     etiology_groups = {
#         "Cardiac cause": [],
#         "cerebrovaskulær": [],
#         "TBI": [],
#         "Other": [],
#     }

#     for patient_id, codes in etiology_metrics.items():
#         if not codes:
#             continue
#         etiology_groups[etiology_coding[int(codes[0])]].append(patient_id)

#     # build data matrices per etiology for this day
#     for eti_state, patient_ids in etiology_groups.items():
#         patient_series = []

#         for patient_id in patient_ids:
#             df = individual_raw_data.get(patient_id)

#             if df is None or day not in df.columns:
#                 continue

#             series = df.loc[:, day]
#             series.name = patient_id
#             patient_series.append(series)

#         if patient_series:
#             etiology_data_per_day[day][eti_state] = pd.concat(
#                 patient_series, axis=1
#             )

# # --------------------------------------------------
# # Plot: grid with 4 columns
# # --------------------------------------------------

# days = sorted(etiology_data_per_day.keys())
# n_cols = 4
# n_days = len(days)
# n_rows = math.ceil(n_days / n_cols)

# fig, axes = plt.subplots(
#     n_rows,
#     n_cols,
#     figsize=(4 * n_cols, 3 * n_rows),
#     sharex=True,
#     sharey=True,
# )

# axes = axes.flatten()

# for ax, day in zip(axes, days):
#     for eti_state, df in etiology_data_per_day[day].items():
#         color = group_colors.get(eti_state, "gray")

#         for patient_id in df.columns:
#             ax.plot(
#                 df.index,
#                 df[patient_id],
#                 color=color,
#                 alpha=0.3,
#                 linewidth=1,
#             )

#     ax.set_title(f"Day {day}")

# # turn off unused axes
# for ax in axes[n_days:]:
#     ax.axis("off")

# # --------------------------------------------------
# # Global labels & legend
# # --------------------------------------------------

# fig.supxlabel("Time (s)")
# fig.supylabel("Pupil size")

# legend_handles = [
#     plt.Line2D([0], [0], color=c, lw=2, label=k)
#     for k, c in group_colors.items()
# ]

# fig.legend(
#     handles=legend_handles,
#     title=f"Etiology group {save_path_time.split(os.sep)[-1]}",
#     loc="upper center",
#     ncol=len(group_colors),
# )

# # Reserve space for legend (top) and x-label (bottom)
# fig.tight_layout(rect=[0.02, 0.01, 0.98, 0.93])
# fig_path = os.path.join(save_path_time, f"etiology_group_pupil_responses_{save_path_time.split(os.sep)[-1]}.pdf")
# plt.savefig(fig_path, dpi=300, bbox_inches="tight")
# plt.close()

#################################################################################################################



# --------------------------------------------------
# Consciousness coding
# --------------------------------------------------

hc_df = HC_left_numeric_data["Ark1"]

consciousness_coding = {
    "C": "Coma",
    "E": "eMCS",
    "M+": "MCS+",
    "M-": "MCS-",
    "U": "UWS",
}

consciousness_colors = {
    "Coma":   "#4D4D4D",   # dark grey — baseline / deepest impairment
    "UWS":    "#0072B2",   # strong blue
    "MCS-":   "#009E73",   # bluish green
    "MCS+":   "#D55E00",   # vermillion
    "eMCS":   "#CC79A7",   # reddish purple
}


# --------------------------------------------------
# Collect data per day
# --------------------------------------------------

consciousness_data_per_day = defaultdict(dict)

for day in sorted(raw_values.keys()):

    consciousness_groups = {
        "Coma": [],
        "UWS": [],
        "MCS+": [],
        "MCS-": [],
        "eMCS": [],
    }

    day_consciousness_metrics = consciousness_metrics.get(day, {})
    for patient_id, codes in day_consciousness_metrics.items():
        if not codes:
            continue
        state = consciousness_coding.get(codes)
        if state is not None:
            consciousness_groups[state].append(patient_id)

    for con_state, patient_ids in consciousness_groups.items():
        patient_series = []

        for patient_id in patient_ids:
            df = individual_raw_data.get(patient_id)

            if df is None or day not in df.columns:
                continue

            series = df.loc[:, day]
            series.name = patient_id
            patient_series.append(series)

        if patient_series:
            consciousness_data_per_day[day][con_state] = pd.concat(
                patient_series, axis=1
            )

from matplotlib.ticker import MultipleLocator

NPI_data_cleaned["SECONDs"] = pd.NA

for day, day_dict in consciousness_data_per_day.items():
    for state, df in day_dict.items():
        patient_ids = df.columns

        mask = (
            (NPI_data_cleaned["redcap_repeat_instance"] == day) &
            (NPI_data_cleaned["record_id"].isin(patient_ids))
        )

        NPI_data_cleaned.loc[mask, "SECONDs"] = state


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(
    NPI_data_cleaned["redcap_repeat_instance"],
    NPI_data_cleaned[f"npi_{save_path_time.split(os.sep)[-1].lower()}_merged"],
    c=NPI_data_cleaned["SECONDs"].map(consciousness_colors),
    s=30,
    alpha=1,
    edgecolor="black",
    linewidth=0.3,
    zorder=3,
)

# Clinical threshold
ax.axhline(
    y=3,
    color="black",
    linestyle="--",
    linewidth=1,
    alpha=1,
    zorder=2,
)

# Axes formatting
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=11)

ax.set_xlabel("Day", fontsize=13, labelpad=12)
ax.set_ylabel("NPI value", fontsize=13)


# --------------------------------------------------
# Figure-level legend (outside plot)
# --------------------------------------------------

legend_handles = [
    plt.Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markerfacecolor=color,
        markeredgecolor="black",
        markersize=6,
        label=group,
    )
    for group, color in consciousness_colors.items()
]

fig.legend(
    handles=legend_handles,
    title="Consciousness group\n",
    fontsize=11,
    title_fontsize=12,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=len(consciousness_colors),
    labelspacing=0.8,
    handletextpad=0.6,
)

# Reserve space for legend
fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.85])

# Save
fig_path = os.path.join(
    save_path_time,
    f"NPI_group_responses_{save_path_time.split(os.sep)[-1]}.jpg"
)
fig.savefig(fig_path, dpi=600, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------
# Plot: grid with 4 columns
# --------------------------------------------------

days = sorted(consciousness_data_per_day.keys())
n_cols = 4
n_days = len(days)
n_rows = math.ceil(n_days / n_cols)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4 * n_cols, 3 * n_rows),
    sharex=True,
    sharey=True,
)

axes = axes.flatten()

for ax, day in zip(axes, days):
    for con_state, df in consciousness_data_per_day[day].items():
        color = consciousness_colors.get(con_state, "gray")
        
        # plot HC data in all subplots
        for hc_id in hc_df.columns:
            ax.plot(
                hc_df.index,
                hc_df[hc_id],
                color="tab:purple",
                alpha=0.4,
                linewidth=1,
                zorder=0,          # push HC behind patients
            )

        for patient_id in df.columns:
            ax.plot(
                df.index,
                df[patient_id],
                color=color,
                alpha=0.3,
                linewidth=1,
            )

    ax.set_title(f"Day {day}")

for ax in axes[n_days:]:
    ax.axis("off")

# --------------------------------------------------
# Global labels & legend
# --------------------------------------------------

fig.supxlabel("Time (s)")
fig.supylabel("Pupil size")

legend_handles = [
    plt.Line2D([0], [0], color=c, lw=2, label=k)
    for k, c in consciousness_colors.items()
]

# Healthy controls (mean)
legend_handles.append(
    plt.Line2D(
        [0], [0],
        color="tab:purple",
        lw=2,
        label="Healthy controls",
    )
)

fig.legend(
    handles=legend_handles,
    title=f"Consciousness group {save_path_time.split(os.sep)[-1]}",
    loc="upper center",
    ncol=len(consciousness_colors),
)

fig.tight_layout(rect=[0.02, 0.01, 0.98, 0.93])

fig_path = os.path.join(
    save_path_time,
    f"consciousness_group_pupil_responses_{save_path_time.split(os.sep)[-1]}.pdf",
)
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()



#################################################################################################################

# --------------------------------------------------
# Sedation coding
# --------------------------------------------------

# sedation_coding = {
#     'P': "Opioder",
#     'R': "Opioder",
#     'M': "Opioder",
#     'F': "Opioder",
#     "T": "Opioder",
#     'O': "Opioder",
#     'nan': "None",
# }

# group_colors = {
#     "Opioder": "tab:purple",
#     "Sedativa": "tab:blue",
#     "None":  "#FFD700",
# }

# # --------------------------------------------------
# # Collect data per day
# # --------------------------------------------------

# sedation_data_per_day = defaultdict(dict)

# for day in sorted(raw_values.keys()):

#     sedation_groups = {
#     "Opioder" :  [],
#     "Sedativa" :  [],
#     "None" :  [],
#     }

#     day_sedation_metrics = sedation_metrics.get(day, {})
#     for patient_id, codes in day_sedation_metrics.items():
#         if not codes:
#             continue
#         state = sedation_coding.get(codes[0])
#         if state is not None:
#             sedation_groups[state].append(patient_id)

#     for sed_state, patient_ids in sedation_groups.items():
#         patient_series = []

#         for patient_id in patient_ids:
#             df = individual_raw_data.get(patient_id)

#             if df is None or day not in df.columns:
#                 continue

#             series = df.loc[:, day]
#             series.name = patient_id
#             patient_series.append(series)

#         if patient_series:
#             sedation_data_per_day[day][sed_state] = pd.concat(
#                 patient_series, axis=1
#             )

# # --------------------------------------------------
# # Plot: grid with 4 columns
# # --------------------------------------------------

# days = sorted(sedation_data_per_day.keys())
# n_cols = 4
# n_days = len(days)
# n_rows = math.ceil(n_days / n_cols)

# fig, axes = plt.subplots(
#     n_rows,
#     n_cols,
#     figsize=(4 * n_cols, 3 * n_rows),
#     sharex=True,
#     sharey=True,
# )

# axes = axes.flatten()

# for ax, day in zip(axes, days):
#     for sed_state, df in sedation_data_per_day[day].items():
#         color = group_colors.get(sed_state, "gray")

#         for patient_id in df.columns:
#             ax.plot(
#                 df.index,
#                 df[patient_id],
#                 color=color,
#                 alpha=0.3,
#                 linewidth=1,
#             )

#     ax.set_title(f"Day {day}")

# for ax in axes[n_days:]:
#     ax.axis("off")

# # --------------------------------------------------
# # Global labels & legend
# # --------------------------------------------------

# fig.supxlabel("Time (s)")
# fig.supylabel("Pupil size")

# legend_handles = [
#     plt.Line2D([0], [0], color=c, lw=2, label=k)
#     for k, c in group_colors.items()
# ]

# fig.legend(
#     handles=legend_handles,
#     title=f"Sedation group {save_path_time.split(os.sep)[-1]}",
#     loc="upper center",
#     ncol=len(group_colors),
# )

# fig.tight_layout(rect=[0.02, 0.01, 0.98, 0.93])

# fig_path = os.path.join(
#     save_path_time,
#     f"sedation_group_pupil_responses_{save_path_time.split(os.sep)[-1]}.pdf",
# )
# plt.savefig(fig_path, dpi=300, bbox_inches="tight")
# plt.close()


#################################################################################################################

# --------------------------------------------------
# Assign consciousness state to NPI_data_cleaned
# --------------------------------------------------

from matplotlib.ticker import MultipleLocator

NPI_data_cleaned["SECONDs"] = pd.NA

for day, day_dict in consciousness_data_per_day.items():
    for state, df in day_dict.items():
        patient_ids = df.columns

        mask = (
            (NPI_data_cleaned["redcap_repeat_instance"] == day) &
            (NPI_data_cleaned["record_id"].isin(patient_ids))
        )

        NPI_data_cleaned.loc[mask, "SECONDs"] = state


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4))

ax.scatter(
    NPI_data_cleaned["redcap_repeat_instance"],
    NPI_data_cleaned[f"npi_{save_path_time.split(os.sep)[-1].lower()}_merged"],
    c=NPI_data_cleaned["SECONDs"].map(consciousness_colors),
    s=30,
    alpha=1,
    edgecolor="black",
    linewidth=0.3,
    zorder=3,
)

# Clinical threshold
ax.axhline(
    y=3,
    color="black",
    linestyle="--",
    linewidth=1,
    alpha=1,
    zorder=2,
)

# Axes formatting
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", labelsize=11)

ax.set_xlabel("Day", fontsize=13, labelpad=12)
ax.set_ylabel("NPI value", fontsize=13)


# --------------------------------------------------
# Figure-level legend (outside plot)
# --------------------------------------------------

legend_handles = [
    plt.Line2D(
        [0], [0],
        marker="o",
        linestyle="",
        markerfacecolor=color,
        markeredgecolor="black",
        markersize=6,
        label=group,
    )
    for group, color in consciousness_colors.items()
]

fig.legend(
    handles=legend_handles,
    title="Consciousness group\n",
    fontsize=11,
    title_fontsize=12,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=len(consciousness_colors),
    labelspacing=0.8,
    handletextpad=0.6,
)

# Reserve space for legend
fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.85])

# Save
fig_path = os.path.join(
    save_path_time,
    f"NPI_group_responses_{save_path_time.split(os.sep)[-1]}.jpg"
)
fig.savefig(fig_path, dpi=600, bbox_inches="tight")
plt.close(fig)
"""