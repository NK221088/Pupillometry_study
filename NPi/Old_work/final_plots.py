import sys
import os

# Add the project root (one folder up) to Python's module search path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Data_loading.read_NPi_data import NPI_data_cleaned
from Data_loading.read_data import HC_left_numeric_data, patient_left_raw_values, patient_left_individual_raw_data, patient_left_consciousness_metrics, patient_left_etiology_metrics, patient_right_etiology_metrics, patient_right_raw_values, patient_right_individual_raw_data, patient_right_consciousness_metrics, patient_left_sedation_metrics, patient_right_sedation_metrics
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from collections import defaultdict
from matplotlib.ticker import MultipleLocator

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

################################################################################################################

###################################
# Etiology coding
###################################

patient_left_etiology_metrics = {key: [value[0]] for key, value in patient_left_etiology_metrics.items()}
patient_right_etiology_metrics = {key: [value[0]] for key, value in patient_right_etiology_metrics.items()}
patient_left_sedation_metrics = {day: {key: [value[0]] for key, value in day_dict.items()} for day, day_dict in patient_left_sedation_metrics.items()}
patient_right_sedation_metrics = {day: {key: [value[0]] for key, value in day_dict.items()} for day, day_dict in patient_right_sedation_metrics.items()}


laterals = ["left", "right"]
save_path_all = {"left": os.getenv("save_path_time_left"), "right": os.getenv("save_path_time_right")}
etiology_metrics_all = {"left": patient_left_etiology_metrics, "right": patient_right_etiology_metrics}
sedation_metrics_all = {"left": patient_left_sedation_metrics, "right": patient_right_sedation_metrics}
consciousness_metrics_all = {"left": patient_left_consciousness_metrics, "right": patient_right_consciousness_metrics}
raw_values_all = {"left": patient_left_raw_values, "right": patient_right_raw_values}
individual_raw_data_all = {"left": patient_left_individual_raw_data, "right": patient_right_individual_raw_data}    

for lateral in laterals:
    save_path = save_path_all[lateral]
    df_etiology = (
    pd.DataFrame.from_dict(
        etiology_metrics_all[lateral],
        orient="index",
        columns=["etiology_code"],
        )
        .reset_index()
        .rename(columns={"index": "patient_id"})
    )

    df_etiology["patient_id"] = df_etiology["patient_id"].astype(str)

    df_etiology.to_csv(
        os.path.join(
            save_path,
            f"etiology_metrics_{save_path.split(os.sep)[-1]}.csv",
        ),
        index=False,
        )

    rows = []

    for day, day_dict in sedation_metrics_all[lateral].items():
        for patient_id, codes in day_dict.items():
            if not codes:
                continue
            rows.append({
                "day": day,
                "patient_id": str(patient_id),
                "sedation_code": codes[0],
            })

    df_sedation = pd.DataFrame(rows)

    df_sedation.to_csv(
        os.path.join(
            save_path,
            f"sedation_metrics_{save_path.split(os.sep)[-1]}.csv",
        ),
        index=False,
    )

    rows = []

    for day, day_dict in consciousness_metrics_all[lateral].items():
        for patient_id, code in day_dict.items():
            if code is None:
                continue
            rows.append({
                "day": day,
                "patient_id": str(patient_id),
                "consciousness_code": code,
            })

    df_consciousness = pd.DataFrame(rows)

    df_consciousness.to_csv(
        os.path.join(
            save_path,
            f"consciousness_metrics_{save_path.split(os.sep)[-1]}.csv",
        ),
        index=False,
    )


    df_raw_long = (
    pd.concat(raw_values_all[lateral], names=["day", "time"])
    .reset_index()
    .melt(
        id_vars=["day", "time"],
        var_name="patient_id",
        value_name="pupil_size",
    )
    )

    df_raw_long.to_csv(
        os.path.join(
            save_path,
            f"raw_values_{save_path.split(os.sep)[-1]}.csv",
        ),
        index=False,
    )

    df_individual_long = (
        pd.concat(individual_raw_data_all[lateral], names=["day", "time"])
        .reset_index()
        .melt(
            id_vars=["day", "time"],
            var_name="patient_id",
            value_name="pupil_size",
        )
    )

    df_individual_long.to_csv(
        os.path.join(
            save_path,
            f"individual_raw_values_{save_path.split(os.sep)[-1]}.csv",
        ),
        index=False,
    )

    
    
    etiology_metrics = etiology_metrics_all[lateral]
    sedation_metrics = sedation_metrics_all[lateral]
    consciousness_metrics = consciousness_metrics_all[lateral]
    raw_values = raw_values_all[lateral]
    individual_raw_data = individual_raw_data_all[lateral]
    
    etiology_coding = {
        0: "Cardiac cause",
        1: "Cerebrovascular",
        2: "Cerebrovascular",
        3: "Cerebrovascular",
        4: "Cerebrovascular",
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

    group_colors = {
        "Cardiac cause": "#4D4D4D",    # dark grey
        "Cerebrovascular": "#0072B2",  # blue
        "TBI": "#D55E00",              # vermillion
        "Other": "#CC79A7",            # magenta
    }

    # --------------------------------------------------
    # Collect data per day
    # --------------------------------------------------

    etiology_data_per_day = defaultdict(dict)

    for day in sorted(raw_values.keys()):

        # group patients by etiology (same across days)
        etiology_groups = {
            "Cardiac cause": [],
            "Cerebrovascular": [],
            "TBI": [],
            "Other": [],
        }

        for patient_id, codes in etiology_metrics.items():
            if not codes:
                continue
            etiology_groups[etiology_coding[int(codes[0])]].append(patient_id)

        # build data matrices per etiology for this day
        for eti_state, patient_ids in etiology_groups.items():
            patient_series = []

            for patient_id in patient_ids:
                df = individual_raw_data.get(patient_id)

                if df is None or day not in df.columns:
                    continue

                series = df.loc[:, day]
                series.name = patient_id
                patient_series.append(series)

            if patient_series:
                etiology_data_per_day[day][eti_state] = pd.concat(
                    patient_series, axis=1
                )

    # --------------------------------------------------
    # Plot: grid with 4 columns
    # --------------------------------------------------

    days = sorted(etiology_data_per_day.keys())
    n_cols = 4
    n_days = len(days)
    n_rows = math.ceil(n_days / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols + 2, 3 * n_rows),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, day in zip(axes, days):
        for eti_state, df in etiology_data_per_day[day].items():
            color = group_colors.get(eti_state, "gray")

            for patient_id in df.columns:
                ax.plot(
                    df.index,
                    df[patient_id],
                    color=color,
                    alpha=0.4,
                    linewidth=1,
                )

        ax.set_title(f"Day {day}")
        ax.set_xticks([])

    # turn off unused axes
    for ax in axes[n_days:]:
        ax.axis("off")

    # --------------------------------------------------
    # Global labels & legend
    # --------------------------------------------------

    fig.supylabel("Pupil size (mm)")

    legend_handles = [
        plt.Line2D([0], [0], color=c, lw=2, label=k)
        for k, c in group_colors.items()
    ]
    
    fig.suptitle(
        f"{save_path.split(os.sep)[-1]} eyes",
        y=0.92,
        fontsize=14,
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(group_colors),
        frameon=False,
    )

    fig.tight_layout(rect=[0.02, 0.01, 0.98, 0.88])

    
    fig_path = os.path.join(save_path, f"etiology_group_pupil_responses_{save_path.split(os.sep)[-1]}.jpg")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()

    ###################################
    # Sedation coding
    ###################################

    sedation_coding = {
        'P': "Sedated",
        'R': "Sedated",
        'M': "Sedated",
        'F': "Sedated",
        "T": "Sedated",
        'O': "Sedated",
        'nan': "Not-sedated",
    }

    group_colors = {
        "Sedated": "tab:purple",
        "Not-sedated": "#E69F00",   # muted amber
    }


    # --------------------------------------------------
    # Collect data per day
    # --------------------------------------------------

    sedation_data_per_day = defaultdict(dict)

    for day in sorted(raw_values.keys()):

        sedation_groups = {
        "Sedated" :  [],
        "Not-sedated" :  [],
        }

        day_sedation_metrics = sedation_metrics.get(day, {})
        for patient_id, codes in day_sedation_metrics.items():
            if not codes:
                continue
            state = sedation_coding.get(codes[0])
            if state is not None:
                sedation_groups[state].append(patient_id)

        for sed_state, patient_ids in sedation_groups.items():
            patient_series = []

            for patient_id in patient_ids:
                df = individual_raw_data.get(patient_id)

                if df is None or day not in df.columns:
                    continue

                series = df.loc[:, day]
                series.name = patient_id
                patient_series.append(series)

            if patient_series:
                sedation_data_per_day[day][sed_state] = pd.concat(
                    patient_series, axis=1
                )

    # --------------------------------------------------
    # Plot: grid with 4 columns
    # --------------------------------------------------

    days = sorted(sedation_data_per_day.keys())
    n_cols = 4
    n_days = len(days)
    n_rows = math.ceil(n_days / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols + 2, 3 * n_rows),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, day in zip(axes, days):
        for sed_state, df in sedation_data_per_day[day].items():
            color = group_colors.get(sed_state, "gray")

            for patient_id in df.columns:
                ax.plot(
                    df.index,
                    df[patient_id],
                    color=color,
                    alpha=0.4,
                    linewidth=1,
                )

        ax.set_title(f"Day {day}")
        ax.set_xticks([])

    for ax in axes[n_days:]:
        ax.axis("off")

    # --------------------------------------------------
    # Global labels & legend
    # --------------------------------------------------

    fig.supylabel("Pupil size (mm)")

    legend_handles = [
        plt.Line2D([0], [0], color=c, lw=2, label=k)
        for k, c in group_colors.items()
    ]

    fig.suptitle(
        f"{save_path.split(os.sep)[-1]} eyes",
        y=0.92,
        fontsize=14,
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(group_colors),
        frameon=False,
    )

    fig.tight_layout(rect=[0.02, 0.01, 0.98, 0.88])

    fig_path = os.path.join(
        save_path,
        f"sedation_group_pupil_responses_{save_path.split(os.sep)[-1]}.jpg",
    )
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()

    ###################################
    # Consciousness coding
    ###################################
    

    hc_df = HC_left_numeric_data["Ark1"]

    consciousness_coding = {
        "C": "Coma",
        "U": "UWS",
        "M-": "MCS-",
        "M+": "MCS+",
        "E": "eMCS",
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
            "MCS-": [],
            "UWS": [],
            "MCS+": [],
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
        figsize=(4 * n_cols + 2, 3 * n_rows),
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
                    alpha=0.1,
                    linewidth=1,
                    zorder=0,          # push HC behind patients
                )

            for patient_id in df.columns:
                ax.plot(
                    df.index,
                    df[patient_id],
                    color=color,
                    alpha=0.9,
                    linewidth=1,
                )

        ax.set_title(f"Day {day}")
        ax.set_xticks([])

    for ax in axes[n_days:]:
        ax.axis("off")

    # --------------------------------------------------
    # Global labels & legend
    # --------------------------------------------------

    fig.supylabel("Pupil size (mm)")

    legend_order = [
        "Coma",
        "UWS",
        "MCS-",
        "MCS+",
        "eMCS",
    ]

    legend_handles = [
        plt.Line2D([0], [0],
                color=consciousness_colors[label],
                lw=2,
                label=label)
        for label in legend_order
    ]

    # Healthy controls last
    legend_handles.append(
        plt.Line2D([0], [0],
                color="tab:purple",
                lw=2,
                label="Healthy controls")
    )

    fig.suptitle(
        f"{save_path.split(os.sep)[-1]} eyes",
        y=0.92,
        fontsize=14,
    )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        bbox_transform=fig.transFigure,
        ncol=len(legend_handles),
        frameon=False,
        handlelength=2.5,
        columnspacing=1.8,
    )


    fig.tight_layout(rect=[0.02, 0.01, 0.98, 0.88])

    fig_path = os.path.join(
        save_path,
        f"consciousness_group_pupil_responses_{save_path.split(os.sep)[-1]}.jpg",
    )
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()