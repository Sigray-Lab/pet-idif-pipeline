#!/usr/bin/env python3
"""Venous blood sample analysis and IDIF calibration.

Loads venous whole-blood and plasma samples, converts to Bq/mL, compares
against the image-derived input function (IDIF), and constructs calibrated
plasma input functions for kinetic modeling.

Part 1: Comparison plots (IDIF vs blood, ratios)
Part 2: Calibrated input function construction
Part 3: Kinetic analysis comparison (Patlak, Logan, 1TCM with raw IDIF vs
         calibrated plasma input)
Part 4: Output tables and comparison figures
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

# Add pipeline directory to path for imports
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir))
from pipeline.s05_kinetics import (
    compute_patlak,
    compute_logan,
    fit_1tcm,
    fit_2tcm,
)


FIGURE_DPI = 300
FIGURE_STYLE = "seaborn-v0_8-whitegrid"
NCI_CC_TO_BQ_ML = 37.0  # 1 nCi/cc = 37 Bq/mL


def load_tac(path: Path) -> pd.DataFrame:
    """Load a TAC TSV, skipping provenance comment lines."""
    return pd.read_csv(path, sep="\t", comment="#")


def load_blood(path: Path) -> pd.DataFrame:
    """Load venous blood sample file and convert to Bq/mL.

    File has columns: 'ABSS sec', 'Cbl disp corr' (nCi/cc), 'Cpl (nCi/cc)'
    """
    df = pd.read_csv(path, sep="\t")
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "ABSS sec": "time_s",
        "Cbl disp corr": "wb_nCi_cc",
        "Cpl (nCi/cc)": "pl_nCi_cc",
    })
    df["time_min"] = df["time_s"] / 60.0
    df["wb_Bq_mL"] = df["wb_nCi_cc"] * NCI_CC_TO_BQ_ML
    df["pl_Bq_mL"] = df["pl_nCi_cc"] * NCI_CC_TO_BQ_ML
    df["plasma_blood_ratio"] = df["pl_Bq_mL"] / df["wb_Bq_mL"]
    return df


def match_idif_to_blood(idif_df: pd.DataFrame, blood_df: pd.DataFrame) -> np.ndarray:
    """For each blood sample time, find the nearest IDIF frame activity."""
    idif_times = idif_df["mid_time_s"].values
    idif_act = idif_df["mean_activity_Bq_per_mL"].values
    matched = np.zeros(len(blood_df))
    for i, bt in enumerate(blood_df["time_s"].values):
        idx = np.argmin(np.abs(idif_times - bt))
        matched[i] = idif_act[idx]
    return matched


def main():
    base = Path(__file__).resolve().parent.parent
    sub = "sub-SUB001_20260225"
    subject_id = "SUB001_20260225"
    out_dir = base / "Outputs"
    qc_dir = base / "QC"

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    blood_path = base / "raw" / subject_id / "blood" / "blood_manual.txt"
    blood = load_blood(blood_path)
    print(f"Blood samples: {len(blood)} time points, {blood['time_s'].iloc[0]:.0f} to {blood['time_s'].iloc[-1]:.0f} s")

    idif = load_tac(out_dir / f"{sub}_idif.tsv")
    brain_wb = load_tac(out_dir / f"{sub}_tac-raw.tsv")
    brain_er = load_tac(out_dir / f"{sub}_tac-raw-eroded.tsv")

    # Radiochem for SUV
    rc = pd.read_csv(base / "raw" / "Radiochem.csv")
    row = rc[rc["id"] == subject_id].iloc[0]
    dose_bq = float(row["injected_MBq"]) * 1e6
    weight_g = float(row["weight_kg"]) * 1000.0
    suv_factor = dose_bq / weight_g

    # Match IDIF values at blood sample times
    blood["idif_Bq_mL"] = match_idif_to_blood(idif, blood)
    blood["idif_wb_ratio"] = blood["idif_Bq_mL"] / blood["wb_Bq_mL"]
    blood["idif_pl_ratio"] = blood["idif_Bq_mL"] / blood["pl_Bq_mL"]

    # Extract arrays
    time_min = idif["mid_time_min"].values
    time_s = idif["mid_time_s"].values
    idif_act = idif["mean_activity_Bq_per_mL"].values
    brain_wb_act = brain_wb["mean_activity_Bq_per_mL"].values
    brain_er_act = brain_er["mean_activity_Bq_per_mL"].values
    durations_s = (brain_wb["end_s"].values - brain_wb["start_s"].values).astype(np.float64)

    # ------------------------------------------------------------------
    # Print blood sample table
    # ------------------------------------------------------------------
    print(f"\n{'Time(s)':>8} {'Time(min)':>9} {'WB(nCi)':>10} {'Pl(nCi)':>10} "
          f"{'WB(Bq/mL)':>10} {'Pl(Bq/mL)':>10} {'IDIF(Bq)':>10} "
          f"{'Pl/WB':>6} {'IDIF/WB':>8} {'IDIF/Pl':>8}")
    print("-" * 110)
    for _, r in blood.iterrows():
        print(f"{r['time_s']:8.0f} {r['time_min']:9.1f} {r['wb_nCi_cc']:10.2f} {r['pl_nCi_cc']:10.2f} "
              f"{r['wb_Bq_mL']:10.1f} {r['pl_Bq_mL']:10.1f} {r['idif_Bq_mL']:10.1f} "
              f"{r['plasma_blood_ratio']:6.3f} {r['idif_wb_ratio']:8.3f} {r['idif_pl_ratio']:8.3f}")

    # ------------------------------------------------------------------
    # Ratio summaries
    # ------------------------------------------------------------------
    late_mask = blood["time_s"] >= 1800  # >= 30 min
    R_idif_wb_late = blood.loc[late_mask, "idif_wb_ratio"].mean()
    R_idif_pl_late = blood.loc[late_mask, "idif_pl_ratio"].mean()
    R_pl_wb_late = blood.loc[late_mask, "plasma_blood_ratio"].mean()

    print(f"\n=== Late-time ratios (>= 30 min, {late_mask.sum()} samples) ===")
    print(f"  IDIF / WB mean:    {R_idif_wb_late:.3f}")
    print(f"  IDIF / Plasma mean: {R_idif_pl_late:.3f}")
    print(f"  Plasma / WB mean:  {R_pl_wb_late:.3f}")

    # ==================================================================
    # PART 1: Figures
    # ==================================================================
    plt.style.use(FIGURE_STYLE)

    # ---- Figure 1: IDIF vs blood samples (linear + log) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, log_y in zip(axes, [False, True]):
        ax.plot(time_min, idif_act, "-", color="crimson", linewidth=1.5, alpha=0.8,
                label="IDIF (carotid)")
        ax.plot(time_min, brain_er_act, "-", color="steelblue", linewidth=1, alpha=0.5,
                label="Brain (eroded, 66 mL)")
        ax.plot(blood["time_min"], blood["wb_Bq_mL"], "o", color="darkorange",
                markersize=7, markeredgecolor="k", markeredgewidth=0.5, label="Venous WB")
        ax.plot(blood["time_min"], blood["pl_Bq_mL"], "s", color="mediumpurple",
                markersize=7, markeredgecolor="k", markeredgewidth=0.5, label="Venous Plasma")
        ax.set_xlabel("Time (min)", fontsize=12)
        ax.set_ylabel("Activity (Bq/mL)", fontsize=12)
        if log_y:
            ax.set_yscale("log")
            ax.set_title("Log scale", fontsize=12)
        else:
            ax.set_title("Linear scale", fontsize=12)
        ax.set_xlim(left=0)
        ax.legend(fontsize=9)

    fig.suptitle(f"{subject_id}: IDIF vs Venous Blood Samples", fontsize=14)
    plt.tight_layout()
    fig.savefig(qc_dir / f"{sub}_blood-vs-idif.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {qc_dir / f'{sub}_blood-vs-idif.png'}")

    # ---- Figure 2: SUV scale comparison ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(time_min, idif_act / suv_factor, "-", color="crimson", linewidth=1.5,
            alpha=0.8, label="IDIF (carotid)")
    ax.plot(time_min, brain_er_act / suv_factor, "-", color="steelblue", linewidth=1,
            alpha=0.5, label="Brain (eroded, 66 mL)")
    ax.plot(blood["time_min"], blood["wb_Bq_mL"] / suv_factor, "o", color="darkorange",
            markersize=7, markeredgecolor="k", markeredgewidth=0.5, label="Venous WB")
    ax.plot(blood["time_min"], blood["pl_Bq_mL"] / suv_factor, "s", color="mediumpurple",
            markersize=7, markeredgecolor="k", markeredgewidth=0.5, label="Venous Plasma")
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("SUV", fontsize=12)
    ax.set_title(f"{subject_id}: IDIF vs Venous Blood (SUV)", fontsize=14)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(qc_dir / f"{sub}_blood-vs-idif-suv.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {qc_dir / f'{sub}_blood-vs-idif-suv.png'}")

    # ---- Figure 3: Plasma / Whole Blood ratio ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(blood["time_min"], blood["plasma_blood_ratio"], "o-", color="teal",
            markersize=7, markeredgecolor="k", markeredgewidth=0.5, linewidth=1.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=R_pl_wb_late, color="teal", linestyle=":", alpha=0.5,
               label=f"Late mean (>= 30 min) = {R_pl_wb_late:.3f}")
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Plasma / Whole Blood ratio", fontsize=12)
    ax.set_title(f"{subject_id}: Plasma to Whole Blood Ratio", fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.9)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(qc_dir / f"{sub}_plasma-blood-ratio.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {qc_dir / f'{sub}_plasma-blood-ratio.png'}")

    # ---- Figure 4: IDIF / Whole Blood ratio (NEW) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(blood["time_min"], blood["idif_wb_ratio"], "o-", color="crimson",
            markersize=7, markeredgecolor="k", markeredgewidth=0.5, linewidth=1.5)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Unity")
    ax.axhline(y=R_idif_wb_late, color="crimson", linestyle=":", alpha=0.5,
               label=f"Late mean (>= 30 min) = {R_idif_wb_late:.3f}")

    # Also show IDIF/Plasma ratio
    ax.plot(blood["time_min"], blood["idif_pl_ratio"], "s--", color="mediumpurple",
            markersize=6, markeredgecolor="k", markeredgewidth=0.5, linewidth=1,
            alpha=0.8, label=f"IDIF/Plasma (late mean = {R_idif_pl_late:.3f})")

    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Ratio", fontsize=12)
    ax.set_title(f"{subject_id}: IDIF / Venous Blood Ratios", fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    fig.savefig(qc_dir / f"{sub}_idif-wholeblood-ratio.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {qc_dir / f'{sub}_idif-wholeblood-ratio.png'}")

    # ==================================================================
    # PART 2: Calibrated input functions
    # ==================================================================
    print("\n" + "=" * 60)
    print("PART 2: Calibrated Input Functions")
    print("=" * 60)

    # --- A. IDIF scaled to whole blood (simple scalar) ---
    # At late times, IDIF reads ~19% high vs venous WB
    # Scale down: Cwb(t) = IDIF(t) / R_idif_wb_late
    Cwb = idif_act / R_idif_wb_late
    print(f"\nA. IDIF scaled to whole blood:")
    print(f"   Scale factor: 1/{R_idif_wb_late:.3f} = {1.0/R_idif_wb_late:.3f}")

    # --- B. IDIF scaled to plasma (time-varying) ---
    # Strategy: scale IDIF to WB first, then multiply by interpolated plasma/blood ratio
    # This accounts for the changing plasma/blood ratio over time (1.09 early -> 1.26 late)

    # Interpolate plasma/blood ratio using PCHIP (monotone cubic) for smoothness
    # Extend to cover full time range: constant extrapolation at boundaries
    blood_t_s = blood["time_s"].values
    blood_pl_wb = blood["plasma_blood_ratio"].values

    # For times before first blood sample (60s), use the earliest ratio
    # For times after last blood sample (9000s), use the latest ratio
    interp_fn = PchipInterpolator(blood_t_s, blood_pl_wb, extrapolate=False)

    pl_wb_interpolated = np.zeros_like(time_s)
    for i, ts in enumerate(time_s):
        if ts <= blood_t_s[0]:
            pl_wb_interpolated[i] = blood_pl_wb[0]
        elif ts >= blood_t_s[-1]:
            pl_wb_interpolated[i] = blood_pl_wb[-1]
        else:
            pl_wb_interpolated[i] = interp_fn(ts)

    # Cpl(t) = Cwb(t) * f_pl(t) = [IDIF(t) / R_idif_wb] * pl_wb_interpolated(t)
    Cpl = Cwb * pl_wb_interpolated

    print(f"\nB. IDIF scaled to plasma (time-varying Pl/WB ratio):")
    print(f"   Plasma/WB ratio range: {pl_wb_interpolated.min():.3f} to {pl_wb_interpolated.max():.3f}")
    print(f"   At t=1 min: Pl/WB = {pl_wb_interpolated[time_min >= 1][0] if np.any(time_min >= 1) else 'N/A':.3f}")
    print(f"   At t=30 min: Pl/WB = {pl_wb_interpolated[time_min >= 30][0] if np.any(time_min >= 30) else 'N/A':.3f}")

    # --- C. IDIF scaled to plasma (simple scalar, for comparison) ---
    Cpl_simple = idif_act / R_idif_pl_late
    print(f"\nC. IDIF scaled to plasma (simple scalar):")
    print(f"   Scale factor: 1/{R_idif_pl_late:.3f} = {1.0/R_idif_pl_late:.3f}")

    # Print comparison at selected times
    print(f"\n{'Time(min)':>9} {'Raw IDIF':>10} {'Scaled WB':>10} {'Scaled Pl':>10} "
          f"{'Simple Pl':>10} {'Brain(Er)':>10}")
    print("-" * 70)
    for t_show in [1, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 135]:
        idx = np.argmin(np.abs(time_min - t_show))
        if abs(time_min[idx] - t_show) < 3:
            print(f"{time_min[idx]:9.1f} {idif_act[idx]:10.1f} {Cwb[idx]:10.1f} "
                  f"{Cpl[idx]:10.1f} {Cpl_simple[idx]:10.1f} {brain_er_act[idx]:10.1f}")

    # ---- Figure 5: Calibrated input functions (NEW) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, log_y in zip(axes, [False, True]):
        ax.plot(time_min, idif_act, "-", color="crimson", linewidth=2, label="Raw IDIF")
        ax.plot(time_min, Cwb, "--", color="darkorange", linewidth=1.5,
                label=f"IDIF scaled to WB (x{1.0/R_idif_wb_late:.3f})")
        ax.plot(time_min, Cpl, "-.", color="mediumpurple", linewidth=1.5,
                label="IDIF scaled to Plasma (time-varying)")
        ax.plot(time_min, Cpl_simple, ":", color="royalblue", linewidth=1.5, alpha=0.7,
                label=f"IDIF scaled to Plasma (simple x{1.0/R_idif_pl_late:.3f})")

        # Overlay blood sample points
        ax.plot(blood["time_min"], blood["wb_Bq_mL"], "o", color="darkorange",
                markersize=5, markeredgecolor="k", markeredgewidth=0.5, alpha=0.6)
        ax.plot(blood["time_min"], blood["pl_Bq_mL"], "s", color="mediumpurple",
                markersize=5, markeredgecolor="k", markeredgewidth=0.5, alpha=0.6)

        ax.set_xlabel("Time (min)", fontsize=12)
        ax.set_ylabel("Activity (Bq/mL)", fontsize=12)
        if log_y:
            ax.set_yscale("log")
            ax.set_title("Log scale", fontsize=12)
        else:
            ax.set_title("Linear scale", fontsize=12)
        ax.set_xlim(left=0)
        ax.legend(fontsize=8)

    fig.suptitle(f"{subject_id}: Calibrated Input Functions", fontsize=14)
    plt.tight_layout()
    fig.savefig(qc_dir / f"{sub}_calibrated-input-functions.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {qc_dir / f'{sub}_calibrated-input-functions.png'}")

    # ==================================================================
    # PART 3: Kinetic analysis comparison
    # ==================================================================
    print("\n" + "=" * 60)
    print("PART 3: Kinetic Analysis Comparison")
    print("=" * 60)

    # Use eroded brain TAC as tissue signal
    Ct = brain_er_act.astype(np.float64)
    t_s_arr = time_s.astype(np.float64)

    # Kinetic parameters (matching s05_kinetics defaults)
    patlak_t_star_s = 30.0 * 60.0   # 30 min
    patlak_t_end_s = 150.0 * 60.0   # 150 min (full scan)
    logan_t_star_s = 30.0 * 60.0
    logan_t_end_s = 150.0 * 60.0
    tcm_t_end_s = 150.0 * 60.0
    Vb = 0.05
    dt_s = 1.0

    input_functions = {
        "Raw IDIF": idif_act.astype(np.float64),
        "IDIF->WB": Cwb.astype(np.float64),
        "IDIF->Plasma (TV)": Cpl.astype(np.float64),
        "IDIF->Plasma (scalar)": Cpl_simple.astype(np.float64),
    }

    results = {}

    for label, Cp in input_functions.items():
        print(f"\n--- {label} ---")

        # Patlak
        patlak = compute_patlak(Ct, Cp, t_s_arr, patlak_t_star_s, patlak_t_end_s)
        print(f"  Patlak: Ki = {patlak['Ki']:.6f} mL/mL/min, V0 = {patlak['V0']:.4f}, R2 = {patlak['R2']:.6f}")

        # Logan
        logan = compute_logan(Ct, Cp, t_s_arr, logan_t_star_s, logan_t_end_s)
        print(f"  Logan:  VT = {logan['VT']:.4f} mL/mL, R2 = {logan['R2']:.6f}")

        # 1TCM
        tcm1 = fit_1tcm(Ct, Cp, t_s_arr, tcm_t_end_s, Vb, dt_s, durations_s=durations_s)
        VT_1tcm = tcm1["VT"]
        print(f"  1TCM:   K1 = {tcm1['K1']:.5f}, k2 = {tcm1['k2']:.5f}, "
              f"VT = {VT_1tcm:.4f}, R2 = {tcm1['R2']:.6f}")

        # 2TCM reversible
        tcm2r = fit_2tcm(Ct, Cp, t_s_arr, tcm_t_end_s, Vb, dt_s,
                         reversible=True, durations_s=durations_s)
        print(f"  2TCM-R: K1 = {tcm2r['K1']:.5f}, k2 = {tcm2r['k2']:.5f}, "
              f"k3 = {tcm2r['k3']:.5f}, k4 = {tcm2r['k4']:.5f}, "
              f"VT = {tcm2r['VT']:.4f}, R2 = {tcm2r['R2']:.6f}")

        # 2TCM irreversible
        tcm2i = fit_2tcm(Ct, Cp, t_s_arr, tcm_t_end_s, Vb, dt_s,
                         reversible=False, durations_s=durations_s)
        print(f"  2TCM-I: K1 = {tcm2i['K1']:.5f}, k2 = {tcm2i['k2']:.5f}, "
              f"k3 = {tcm2i['k3']:.5f}, Ki = {tcm2i['Ki_derived']:.6f}, "
              f"R2 = {tcm2i['R2']:.6f}")

        results[label] = {
            "patlak": patlak,
            "logan": logan,
            "1tcm": tcm1,
            "2tcm_rev": tcm2r,
            "2tcm_irrev": tcm2i,
        }

    # ==================================================================
    # PART 4: Comparison figure and output tables
    # ==================================================================
    print("\n" + "=" * 60)
    print("PART 4: Comparison Outputs")
    print("=" * 60)

    # ---- Figure 6: Kinetics comparison bar chart (NEW) ----
    labels_short = list(input_functions.keys())
    n_inputs = len(labels_short)

    # Collect key metrics
    Ki_vals = [results[l]["patlak"]["Ki"] for l in labels_short]
    V0_vals = [results[l]["patlak"]["V0"] for l in labels_short]
    VT_logan_vals = [results[l]["logan"]["VT"] for l in labels_short]
    K1_1tcm_vals = [results[l]["1tcm"]["K1"] for l in labels_short]
    k2_1tcm_vals = [results[l]["1tcm"]["k2"] for l in labels_short]
    VT_1tcm_vals = [results[l]["1tcm"]["VT"] for l in labels_short]
    VT_2tcm_vals = [results[l]["2tcm_rev"]["VT"] for l in labels_short]
    K1_2tcm_vals = [results[l]["2tcm_rev"]["K1"] for l in labels_short]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = ["crimson", "darkorange", "mediumpurple", "royalblue"]
    x = np.arange(n_inputs)
    bar_width = 0.6

    # Panel 1: Patlak Ki
    ax = axes[0, 0]
    ax.bar(x, [v * 1e3 for v in Ki_vals], bar_width, color=colors)
    ax.set_ylabel("Ki (x10$^{-3}$ mL/mL/min)", fontsize=10)
    ax.set_title("Patlak Ki", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=8, rotation=20, ha="right")

    # Panel 2: Patlak V0
    ax = axes[0, 1]
    ax.bar(x, V0_vals, bar_width, color=colors)
    ax.set_ylabel("V0 (mL/mL)", fontsize=10)
    ax.set_title("Patlak V0 (distribution volume)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=8, rotation=20, ha="right")

    # Panel 3: Logan VT
    ax = axes[0, 2]
    ax.bar(x, VT_logan_vals, bar_width, color=colors)
    ax.set_ylabel("VT (mL/mL)", fontsize=10)
    ax.set_title("Logan VT", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=8, rotation=20, ha="right")

    # Panel 4: 1TCM K1
    ax = axes[1, 0]
    ax.bar(x, K1_1tcm_vals, bar_width, color=colors)
    ax.set_ylabel("K1 (mL/mL/min)", fontsize=10)
    ax.set_title("1TCM K1", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=8, rotation=20, ha="right")

    # Panel 5: 1TCM VT = K1/k2
    ax = axes[1, 1]
    ax.bar(x, VT_1tcm_vals, bar_width, color=colors)
    ax.set_ylabel("VT (mL/mL)", fontsize=10)
    ax.set_title("1TCM VT (= K1/k2)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=8, rotation=20, ha="right")

    # Panel 6: 2TCM-R VT
    ax = axes[1, 2]
    ax.bar(x, VT_2tcm_vals, bar_width, color=colors)
    ax.set_ylabel("VT (mL/mL)", fontsize=10)
    ax.set_title("2TCM Reversible VT", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, fontsize=8, rotation=20, ha="right")

    fig.suptitle(f"{subject_id}: Kinetic Parameters by Input Function", fontsize=14)
    plt.tight_layout()
    fig.savefig(qc_dir / f"{sub}_kinetics-comparison.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {qc_dir / f'{sub}_kinetics-comparison.png'}")

    # ---- Figure 7: 1TCM model fits for each input function ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes_flat = axes.flatten()

    for i, (label, Cp) in enumerate(input_functions.items()):
        ax = axes_flat[i]
        tcm1 = results[label]["1tcm"]
        t_fit = tcm1["t_frames"] / 60.0

        ax.plot(time_min, Ct / suv_factor, "o", color="steelblue", markersize=4,
                alpha=0.7, label="Measured (brain)")
        ax.plot(t_fit, tcm1["Ct_model_at_frames"] / suv_factor, "-", color="red",
                linewidth=1.5, label=f"1TCM fit (R2={tcm1['R2']:.4f})")
        ax.set_xlabel("Time (min)", fontsize=10)
        ax.set_ylabel("SUV", fontsize=10)
        ax.set_title(f"{label}\nK1={tcm1['K1']:.4f}, k2={tcm1['k2']:.4f}, "
                     f"VT={tcm1['VT']:.3f}", fontsize=10)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    fig.suptitle(f"{subject_id}: 1TCM Fits by Input Function", fontsize=14)
    plt.tight_layout()
    fig.savefig(qc_dir / f"{sub}_1tcm-fits-comparison.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {qc_dir / f'{sub}_1tcm-fits-comparison.png'}")

    # ---- Save blood samples TSV ----
    blood_out = blood[["time_s", "time_min", "wb_nCi_cc", "pl_nCi_cc",
                       "wb_Bq_mL", "pl_Bq_mL", "idif_Bq_mL",
                       "plasma_blood_ratio", "idif_wb_ratio", "idif_pl_ratio"]].copy()
    blood_tsv_path = out_dir / f"{sub}_blood-samples.tsv"
    with open(blood_tsv_path, "w") as f:
        f.write(f"# subject: {subject_id}\n")
        f.write(f"# analysis: venous blood sample comparison with IDIF\n")
        f.write(f"# source: {blood_path.name}\n")
        f.write(f"# conversion: 1 nCi/cc = 37 Bq/mL\n")
        f.write(f"# late_mean_idif_wb_ratio: {R_idif_wb_late:.4f}\n")
        f.write(f"# late_mean_idif_pl_ratio: {R_idif_pl_late:.4f}\n")
        f.write(f"# late_mean_plasma_blood_ratio: {R_pl_wb_late:.4f}\n")
        blood_out.to_csv(f, sep="\t", index=False, float_format="%.4f")
    print(f"\nSaved: {blood_tsv_path}")

    # ---- Save calibrated input functions TSV ----
    input_fn_df = pd.DataFrame({
        "frame": idif["frame"].values,
        "mid_time_s": time_s,
        "mid_time_min": time_min,
        "raw_idif_Bq_mL": idif_act,
        "scaled_wb_Bq_mL": Cwb,
        "scaled_plasma_tv_Bq_mL": Cpl,
        "scaled_plasma_scalar_Bq_mL": Cpl_simple,
        "plasma_blood_ratio_interp": pl_wb_interpolated,
    })
    input_fn_path = out_dir / f"{sub}_input-functions.tsv"
    with open(input_fn_path, "w") as f:
        f.write(f"# subject: {subject_id}\n")
        f.write(f"# analysis: calibrated input functions from blood samples\n")
        f.write(f"# raw_idif: uncorrected IDIF from carotid\n")
        f.write(f"# scaled_wb: IDIF / {R_idif_wb_late:.4f} (late IDIF/WB ratio)\n")
        f.write(f"# scaled_plasma_tv: (IDIF / R_wb) * interpolated Pl/WB ratio (time-varying)\n")
        f.write(f"# scaled_plasma_scalar: IDIF / {R_idif_pl_late:.4f} (late IDIF/Pl ratio)\n")
        f.write(f"# R_idif_wb_late: {R_idif_wb_late:.4f}\n")
        f.write(f"# R_idif_pl_late: {R_idif_pl_late:.4f}\n")
        input_fn_df.to_csv(f, sep="\t", index=False, float_format="%.4f")
    print(f"Saved: {input_fn_path}")

    # ---- Save kinetics comparison TSV ----
    comp_rows = []
    for label in labels_short:
        r = results[label]
        comp_rows.append({
            "input_function": label,
            # Patlak
            "patlak_Ki": r["patlak"]["Ki"],
            "patlak_Ki_se": r["patlak"]["Ki_se"],
            "patlak_V0": r["patlak"]["V0"],
            "patlak_R2": r["patlak"]["R2"],
            # Logan
            "logan_VT": r["logan"]["VT"],
            "logan_VT_se": r["logan"]["VT_se"],
            "logan_R2": r["logan"]["R2"],
            # 1TCM
            "tcm1_K1": r["1tcm"]["K1"],
            "tcm1_k2": r["1tcm"]["k2"],
            "tcm1_VT": r["1tcm"]["VT"],
            "tcm1_R2": r["1tcm"]["R2"],
            "tcm1_AIC": r["1tcm"]["AIC"],
            # 2TCM reversible
            "tcm2r_K1": r["2tcm_rev"]["K1"],
            "tcm2r_k2": r["2tcm_rev"]["k2"],
            "tcm2r_k3": r["2tcm_rev"]["k3"],
            "tcm2r_k4": r["2tcm_rev"]["k4"],
            "tcm2r_VT": r["2tcm_rev"]["VT"],
            "tcm2r_R2": r["2tcm_rev"]["R2"],
            "tcm2r_AIC": r["2tcm_rev"]["AIC"],
            # 2TCM irreversible
            "tcm2i_K1": r["2tcm_irrev"]["K1"],
            "tcm2i_k2": r["2tcm_irrev"]["k2"],
            "tcm2i_k3": r["2tcm_irrev"]["k3"],
            "tcm2i_Ki": r["2tcm_irrev"]["Ki_derived"],
            "tcm2i_R2": r["2tcm_irrev"]["R2"],
            "tcm2i_AIC": r["2tcm_irrev"]["AIC"],
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_path = out_dir / f"{sub}_kinetics-comparison.tsv"
    with open(comp_path, "w") as f:
        f.write(f"# subject: {subject_id}\n")
        f.write(f"# analysis: kinetic parameter comparison across input functions\n")
        f.write(f"# tissue: brain eroded\n")
        f.write(f"# patlak_t_star_min: 30\n")
        f.write(f"# logan_t_star_min: 30\n")
        f.write(f"# tcm_Vb: {Vb}\n")
        f.write(f"# tcm_dt_s: {dt_s}\n")
        comp_df.to_csv(f, sep="\t", index=False, float_format="%.6f")
    print(f"Saved: {comp_path}")

    # ---- Summary table to console ----
    print("\n" + "=" * 80)
    print("KINETICS COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Input Function':<25} {'Patlak Ki':>12} {'Logan VT':>10} "
          f"{'1TCM K1':>10} {'1TCM VT':>10} {'2TCM-R VT':>10}")
    print("-" * 80)
    for label in labels_short:
        r = results[label]
        print(f"{label:<25} {r['patlak']['Ki']:12.6f} {r['logan']['VT']:10.4f} "
              f"{r['1tcm']['K1']:10.5f} {r['1tcm']['VT']:10.4f} "
              f"{r['2tcm_rev']['VT']:10.4f}")

    # Percent change relative to raw IDIF
    print(f"\n{'Percent change vs Raw IDIF:'}")
    print("-" * 80)
    ref = results["Raw IDIF"]
    for label in labels_short[1:]:
        r = results[label]
        def pct(new, old):
            if old == 0 or np.isnan(old) or np.isnan(new):
                return "   N/A"
            return f"{(new - old) / old * 100:+8.1f}%"

        print(f"{label:<25} {pct(r['patlak']['Ki'], ref['patlak']['Ki']):>12} "
              f"{pct(r['logan']['VT'], ref['logan']['VT']):>10} "
              f"{pct(r['1tcm']['K1'], ref['1tcm']['K1']):>10} "
              f"{pct(r['1tcm']['VT'], ref['1tcm']['VT']):>10} "
              f"{pct(r['2tcm_rev']['VT'], ref['2tcm_rev']['VT']):>10}")

    print("\nDone.")


if __name__ == "__main__":
    main()
