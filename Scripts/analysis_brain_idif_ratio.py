#!/usr/bin/env python3
"""Compute brain:IDIF ratio vs time to assess equilibrium behavior.

Diagnostic interpretation:
  - Ratio stabilizes early (~20-30 min): reversible partitioning (1TCM equilibrium)
  - Ratio keeps rising: accumulation (irreversible trapping or metabolite buildup)
  - Ratio drifts down: washout faster than blood clearance

Uses both the whole-brain mask (81 mL) and eroded mask (66 mL) TACs.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DPI = 300
FIGURE_STYLE = "seaborn-v0_8-whitegrid"


def load_tac(path: Path) -> pd.DataFrame:
    """Load a TAC TSV, skipping provenance comment lines."""
    return pd.read_csv(path, sep="\t", comment="#")


def main():
    base = Path(__file__).resolve().parent.parent
    sub = "sub-SUB001_20260225"
    out_dir = base / "Outputs"
    qc_dir = base / "QC"

    # Load TACs
    idif = load_tac(out_dir / f"{sub}_idif.tsv")
    brain_wb = load_tac(out_dir / f"{sub}_tac-raw.tsv")
    brain_er = load_tac(out_dir / f"{sub}_tac-raw-eroded.tsv")

    # Load radiochem for SUV
    rc = pd.read_csv(base / "raw" / "Radiochem.csv")
    row = rc[rc["id"] == sub[4:]].iloc[0]
    dose_bq = float(row["injected_MBq"]) * 1e6
    weight_g = float(row["weight_kg"]) * 1000.0
    suv_factor = dose_bq / weight_g

    time_min = idif["mid_time_min"].values
    idif_act = idif["mean_activity_Bq_per_mL"].values
    wb_act = brain_wb["mean_activity_Bq_per_mL"].values
    er_act = brain_er["mean_activity_Bq_per_mL"].values

    # Skip frames where IDIF is zero or near-zero (frame 0, and be safe with frame 1)
    valid = (idif_act > 100) & (time_min > 0.3)

    t = time_min[valid]
    ratio_wb = wb_act[valid] / idif_act[valid]
    ratio_er = er_act[valid] / idif_act[valid]

    # SUV versions for reference
    idif_suv = idif_act / suv_factor
    wb_suv = wb_act / suv_factor
    er_suv = er_act / suv_factor

    # ---- Print ratio table ----
    print(f"{'Time (min)':>10} {'IDIF Bq/mL':>12} {'Brain(WB)':>12} {'Brain(Er)':>12} "
          f"{'Ratio(WB)':>10} {'Ratio(Er)':>10}")
    print("-" * 78)
    for i in range(len(t)):
        fi = np.where(valid)[0][i]
        print(f"{t[i]:10.1f} {idif_act[fi]:12.1f} {wb_act[fi]:12.1f} {er_act[fi]:12.1f} "
              f"{ratio_wb[i]:10.3f} {ratio_er[i]:10.3f}")

    # ---- Summary stats ----
    # Early phase: 2-10 min
    early = (t >= 2) & (t <= 10)
    # Mid phase: 15-30 min
    mid = (t >= 15) & (t <= 30)
    # Late phase: 30-60 min (plateau)
    late = (t >= 30) & (t <= 60)
    # Very late: 60-150 min
    vlate = (t >= 60) & (t <= 150)

    print("\n=== Ratio Summary (eroded mask) ===")
    for label, mask in [("2-10 min", early), ("15-30 min", mid),
                         ("30-60 min", late), ("60-150 min", vlate)]:
        if np.any(mask):
            vals = ratio_er[mask]
            print(f"  {label:12s}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
                  f"range=[{np.min(vals):.3f}, {np.max(vals):.3f}]")

    # ---- Drift analysis ----
    # Linear regression on ratio vs time for t > 15 min (post-equilibrium)
    post_eq = t >= 15
    if np.sum(post_eq) >= 3:
        from numpy.polynomial import polynomial as P
        coeffs = P.polyfit(t[post_eq], ratio_er[post_eq], 1)  # [intercept, slope]
        slope = coeffs[1]
        intercept = coeffs[0]
        ratio_at_15 = intercept + slope * 15
        ratio_at_135 = intercept + slope * 135
        pct_change = (ratio_at_135 - ratio_at_15) / ratio_at_15 * 100

        print(f"\n=== Drift Analysis (eroded, t > 15 min) ===")
        print(f"  Linear fit: ratio = {intercept:.4f} + {slope:.6f} * t")
        print(f"  Slope: {slope:.6f} per min ({slope * 60:.4f} per hour)")
        print(f"  Ratio at 15 min: {ratio_at_15:.3f}")
        print(f"  Ratio at 135 min: {ratio_at_135:.3f}")
        print(f"  Change over 15-135 min: {pct_change:+.1f}%")
        if abs(pct_change) < 10:
            print(f"  --> STABLE: ratio changes < 10% over 2 hours, consistent with reversible equilibrium")
        elif pct_change > 10:
            print(f"  --> RISING: ratio increases > 10%, could indicate accumulation/trapping/metabolite")
        else:
            print(f"  --> FALLING: ratio decreases > 10%, could indicate washout")

    # ---- Figure 1: Brain:IDIF ratio vs time ----
    plt.style.use(FIGURE_STYLE)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2, 1]})

    # Top panel: Both TACs (SUV)
    ax = axes[0]
    ax.plot(time_min, wb_suv, "o-", color="steelblue", markersize=3, linewidth=1.5,
            label="Brain (whole, 81 mL)")
    ax.plot(time_min, er_suv, "s-", color="teal", markersize=3, linewidth=1.5,
            label="Brain (eroded, 66 mL)")
    ax.plot(time_min, idif_suv, "^-", color="crimson", markersize=3, linewidth=1.5,
            label="IDIF (carotid)")
    ax.set_ylabel("SUV", fontsize=12)
    ax.set_title(f"{sub}: Brain TAC and IDIF (SUV)", fontsize=14)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10, loc="upper right")
    # Inset for first 5 min
    ax_ins = ax.inset_axes([0.35, 0.35, 0.35, 0.55])
    ax_ins.plot(time_min, idif_suv, "^-", color="crimson", markersize=2, linewidth=1)
    ax_ins.plot(time_min, wb_suv, "o-", color="steelblue", markersize=2, linewidth=1)
    ax_ins.plot(time_min, er_suv, "s-", color="teal", markersize=2, linewidth=1)
    ax_ins.set_xlim(0, 5)
    ax_ins.set_ylim(0, max(idif_suv[:15]) * 1.1)
    ax_ins.set_xlabel("min", fontsize=8)
    ax_ins.set_ylabel("SUV", fontsize=8)
    ax_ins.set_title("First 5 min", fontsize=9)
    ax_ins.tick_params(labelsize=7)

    # Bottom panel: Ratio vs time
    ax = axes[1]
    ax.plot(t, ratio_wb, "o-", color="steelblue", markersize=4, linewidth=1.5,
            label="Whole brain / IDIF")
    ax.plot(t, ratio_er, "s-", color="teal", markersize=4, linewidth=1.5,
            label="Eroded brain / IDIF")

    # Overlay linear fit
    if np.sum(post_eq) >= 3:
        t_fit = np.linspace(15, 140, 100)
        ax.plot(t_fit, intercept + slope * t_fit, "--", color="gray", linewidth=1,
                label=f"Linear fit (slope={slope:+.5f}/min, {pct_change:+.1f}% drift)")

    ax.axhline(y=np.mean(ratio_er[late]), color="teal", linestyle=":", alpha=0.5,
               label=f"Plateau mean = {np.mean(ratio_er[late]):.3f}")
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Brain / IDIF ratio", fontsize=12)
    ax.set_title("Brain:IDIF Concentration Ratio vs Time", fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    fig_path = qc_dir / f"{sub}_brain-idif-ratio.png"
    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_path}")

    # ---- Figure 2: Ratio on log-time axis to see early kinetics ----
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogx(t, ratio_er, "s-", color="teal", markersize=5, linewidth=1.5,
                label="Eroded brain / IDIF")
    ax.semilogx(t, ratio_wb, "o-", color="steelblue", markersize=4, linewidth=1.5,
                alpha=0.6, label="Whole brain / IDIF")
    ax.axhline(y=np.mean(ratio_er[late]), color="teal", linestyle=":", alpha=0.5)
    ax.axvspan(30, 60, alpha=0.08, color="teal", label="Plateau window (30-60 min)")
    ax.set_xlabel("Time (min, log scale)", fontsize=12)
    ax.set_ylabel("Brain / IDIF ratio", fontsize=12)
    ax.set_title(f"{sub}: Brain:IDIF Ratio (log time axis)", fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig_path2 = qc_dir / f"{sub}_brain-idif-ratio-logtime.png"
    fig.savefig(fig_path2, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path2}")

    # ---- Save ratio TSV ----
    ratio_df = pd.DataFrame({
        "frame": idif["frame"].values[valid],
        "mid_time_min": t,
        "idif_Bq_per_mL": idif_act[valid],
        "brain_whole_Bq_per_mL": wb_act[valid],
        "brain_eroded_Bq_per_mL": er_act[valid],
        "ratio_whole": ratio_wb,
        "ratio_eroded": ratio_er,
    })
    tsv_path = out_dir / f"{sub}_brain-idif-ratio.tsv"
    with open(tsv_path, "w") as f:
        f.write(f"# subject: {sub[4:]}\n")
        f.write(f"# analysis: brain:IDIF concentration ratio vs time\n")
        f.write(f"# plateau_ratio_30_60min: {np.mean(ratio_er[late]):.4f}\n")
        if np.sum(post_eq) >= 3:
            f.write(f"# drift_slope_per_min: {slope:.6f}\n")
            f.write(f"# drift_pct_15_to_135min: {pct_change:+.1f}\n")
        ratio_df.to_csv(f, sep="\t", index=False, float_format="%.4f")
    print(f"Saved: {tsv_path}")


if __name__ == "__main__":
    main()
