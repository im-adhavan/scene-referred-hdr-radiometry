import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from config import RAW_ROOT, HDR_ROOT, OUTPUT_DIR
from src.exr_utils import read_exr
from src.raw_utils import read_exposure_time, load_raw_linear
from src.metrics import compute_dr, compute_log_spread
from src.hdr_merge import merge_hdr_gpu


# ============================================================
# Utility: Save DataFrame as Proper PNG Table
# ============================================================

def save_dataframe_table(df, title, path, fontsize=12, scale=(1.2, 1.6)):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=20)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(scale[0], scale[1])

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# EV Span
# ============================================================

def compute_ev_span(scene):
    raw_folder = os.path.join(RAW_ROOT, scene)
    nef_files = sorted([
        f for f in os.listdir(raw_folder)
        if f.lower().endswith(".nef")
    ])

    exposures = [
        read_exposure_time(os.path.join(raw_folder, f))
        for f in nef_files
    ]

    return math.log2(max(exposures) / min(exposures))


# ============================================================
# Redundancy Study
# ============================================================

def redundancy_study(scene):
    raw_folder = os.path.join(RAW_ROOT, scene)
    nef_files = sorted([
        f for f in os.listdir(raw_folder)
        if f.lower().endswith(".nef")
    ])

    exposures = []
    images = []

    import torch

    for f in nef_files:
        path = os.path.join(raw_folder, f)
        exposures.append(read_exposure_time(path))
        images.append(torch.from_numpy(load_raw_linear(path)))

    configs = {
        "9_exp": list(range(9)),
        "7_exp": list(range(1, 8)),
        "5_exp": list(range(2, 7)),
        "3_exp": list(range(3, 6))
    }

    results = {}

    for label, idx in configs.items():
        imgs = [images[i] for i in idx]
        exps = [exposures[i] for i in idx]
        hdr = merge_hdr_gpu(imgs, exps)
        results[label] = compute_dr(hdr)

    return results


# ============================================================
# Main Analysis Pipeline
# ============================================================

def run_full_analysis():

    figures_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("\n===== HDR Radiometric Analysis =====")

    # --------------------------------------------------------
    # Compute Scene Metrics
    # --------------------------------------------------------

    exr_files = [f for f in os.listdir(HDR_ROOT)
                 if f.endswith(".exr")]

    scene_results = []

    for file in tqdm(exr_files, desc="Analyzing HDR scenes"):
        scene = file.replace(".exr", "")
        hdr = read_exr(os.path.join(HDR_ROOT, file))

        dr_val = compute_dr(hdr)
        log_spread = compute_log_spread(hdr)
        ev_span = compute_ev_span(scene)
        theoretical_dr = ev_span * math.log10(2)

        scene_results.append(
            [scene, dr_val, log_spread, ev_span, theoretical_dr]
        )

    df = pd.DataFrame(scene_results,
                      columns=["Scene",
                               "DR_log10",
                               "LogSpread",
                               "EV_span",
                               "Theoretical_DR"])

    # --------------------------------------------------------
    # Correlation & Regression
    # --------------------------------------------------------

    r_ev, _ = pearsonr(df["EV_span"], df["DR_log10"])
    r_log, _ = pearsonr(df["LogSpread"], df["DR_log10"])

    X = df["EV_span"].values.reshape(-1, 1)
    y = df["DR_log10"].values

    reg = LinearRegression().fit(X, y)
    r2 = reg.score(X, y)

    corr_df = pd.DataFrame({
        "Metric": [
            "Corr(DR, EV_span)",
            "Corr(DR, LogSpread)",
            "R² (DR ~ EV_span)",
            "Slope",
            "Intercept"
        ],
        "Value": [
            round(r_ev, 4),
            round(r_log, 4),
            round(r2, 4),
            round(reg.coef_[0], 4),
            round(reg.intercept_, 4)
        ]
    })

    save_dataframe_table(
        corr_df,
        "Correlation & Regression Summary",
        os.path.join(figures_dir, "correlation_summary.png")
    )

    # --------------------------------------------------------
    # Redundancy Study
    # --------------------------------------------------------

    selected = df.sort_values("DR_log10").iloc[
        [0, 25, 50, 75, 104]
    ]["Scene"]

    redundancy_data = {}

    for scene in selected:
        redundancy_data[scene] = redundancy_study(scene)

    redundancy_df = pd.DataFrame(redundancy_data).T

    redundancy_df["Delta_3_vs_9"] = (
        redundancy_df["3_exp"] - redundancy_df["9_exp"]
    )

    redundancy_df["Delta_5_vs_9"] = (
        redundancy_df["5_exp"] - redundancy_df["9_exp"]
    )

    save_dataframe_table(
        redundancy_df.round(4),
        "Exposure Redundancy Results",
        os.path.join(figures_dir, "redundancy_table.png"),
        fontsize=11,
        scale=(1.1, 1.5)
    )

    # --------------------------------------------------------
    # Dataset Summary Table
    # --------------------------------------------------------

    summary_df = pd.DataFrame({
        "Metric": [
            "Scenes Analyzed",
            "Mean DR",
            "Min DR",
            "Max DR",
            "Mean Δ (3 vs 9)",
            "Mean Δ (5 vs 9)"
        ],
        "Value": [
            len(df),
            round(df["DR_log10"].mean(), 4),
            round(df["DR_log10"].min(), 4),
            round(df["DR_log10"].max(), 4),
            round(redundancy_df["Delta_3_vs_9"].mean(), 4),
            round(redundancy_df["Delta_5_vs_9"].mean(), 4)
        ]
    })

    save_dataframe_table(
        summary_df,
        "HDR Radiometric Dataset Summary",
        os.path.join(figures_dir, "final_summary.png")
    )

    # --------------------------------------------------------
    # Visual Plots
    # --------------------------------------------------------

    sns.set(style="whitegrid")

    # DR Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["DR_log10"], bins=20, kde=True)
    plt.title("HDR Dynamic Range Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "dr_distribution.png"), dpi=300)
    plt.close()

    # DR vs EV
    plt.figure(figsize=(6, 6))
    plt.scatter(df["EV_span"], df["DR_log10"])
    plt.plot(df["EV_span"], df["Theoretical_DR"], color="red")
    plt.xlabel("EV Span")
    plt.ylabel("Measured DR log10")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "dr_vs_ev.png"), dpi=300)
    plt.close()

    # DR vs LogSpread
    plt.figure(figsize=(6, 6))
    plt.scatter(df["DR_log10"], df["LogSpread"])
    plt.xlabel("DR log10")
    plt.ylabel("Log-Luminance Spread")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "dr_vs_logspread.png"), dpi=300)
    plt.close()

    # Correlation Heatmap
    corr_matrix = df[["DR_log10",
                      "EV_span",
                      "LogSpread"]].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix,
                annot=True,
                cmap="coolwarm",
                vmin=-1,
                vmax=1)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "correlation_matrix.png"), dpi=300)
    plt.close()

    print("\nAll PNG outputs saved to:", figures_dir)