"""Step 2: SUV time-activity curve."""
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.cache import check_outputs_current, log_skip, write_provenance_header
from pipeline.config import PipelineConfig

log = logging.getLogger("s02_suv_tac")


def _load_radiochem(csv_path: Path, subject_id: str, cfg: PipelineConfig) -> dict:
    """Load radiochemistry data for the given subject."""
    df = pd.read_csv(csv_path)
    row = df[df["id"] == subject_id]
    if row.empty:
        raise ValueError(f"Subject {subject_id} not found in {csv_path}")
    row = row.iloc[0]

    dose_mbq = float(row["injected_MBq"])
    weight_kg = float(row["weight_kg"])

    # Sanity checks
    dmin, dmax = cfg.INJECTED_DOSE_RANGE_MBQ
    if not (dmin <= dose_mbq <= dmax):
        log.warning("Injected dose %.2f MBq outside expected range (%.0f-%.0f)",
                     dose_mbq, dmin, dmax)

    wmin, wmax = cfg.BODY_WEIGHT_RANGE_KG
    if not (wmin <= weight_kg <= wmax):
        log.warning("Body weight %.2f kg outside expected range (%.0f-%.0f)",
                     weight_kg, wmin, wmax)

    dose_bq = dose_mbq * 1e6
    weight_g = weight_kg * 1000.0

    log.info("Radiochem: dose=%.2f MBq (%.0f Bq), weight=%.1f kg (%.0f g)",
             dose_mbq, dose_bq, weight_kg, weight_g)

    return {
        "dose_mbq": dose_mbq,
        "dose_bq": dose_bq,
        "weight_kg": weight_kg,
        "weight_g": weight_g,
    }


def run(subject_id: str, cfg: PipelineConfig, force: bool = False) -> dict:
    """
    Compute SUV TAC from raw TAC and radiochemistry data.
    SUV = activity_Bq_per_mL / (dose_Bq / weight_g)
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("STEP s02: SUV TAC")
    log.info("Subject: %s", subject_id)

    sub = f"sub-{subject_id}"

    # Inputs
    tac_raw_path = cfg.outputs_dir() / f"{sub}_tac-raw.tsv"
    radiochem_path = cfg.radiochem_path()

    # Outputs
    outputs = {
        "tac_suv": cfg.outputs_dir() / f"{sub}_tac-suv.tsv",
        "fig_suv": cfg.figures_dir() / f"{sub}_tac-suv.png",
    }

    # Cache check
    input_paths = [tac_raw_path, radiochem_path]
    output_paths = list(outputs.values())
    if not force and check_outputs_current(input_paths, output_paths):
        log_skip("s02_suv_tac", output_paths)
        return outputs

    for inp in input_paths:
        if not inp.exists():
            raise FileNotFoundError(f"Required input missing: {inp}")
        log.info("Input: %s", inp)

    # Load raw TAC
    tac_raw = pd.read_csv(tac_raw_path, sep="\t", comment="#")
    log.info("Raw TAC: %d frames", len(tac_raw))

    # Load radiochem
    rc = _load_radiochem(radiochem_path, subject_id, cfg)

    # Compute SUV
    suv_factor = rc["dose_bq"] / rc["weight_g"]
    log.info("SUV normalization factor: %.2f (Bq/mL per SUV unit)", suv_factor)

    tac_suv = tac_raw.copy()
    tac_suv["suv"] = tac_suv["mean_activity_Bq_per_mL"] / suv_factor
    tac_suv["suv_std"] = tac_suv["std_activity"] / suv_factor

    # Summary
    peak_idx = int(np.argmax(tac_suv["suv"].values))
    peak_suv = tac_suv["suv"].iloc[peak_idx]
    peak_time = tac_suv["mid_time_min"].iloc[peak_idx]

    # Plateau: 30-60 min
    plateau_mask = (tac_suv["mid_time_min"] >= 30) & (tac_suv["mid_time_min"] <= 60)
    plateau_suv = float(tac_suv.loc[plateau_mask, "suv"].mean()) if plateau_mask.any() else np.nan

    log.info("Peak SUV: %.3f at %.1f min", peak_suv, peak_time)
    log.info("Plateau SUV (30-60 min): %.3f", plateau_suv)

    # Save TSV
    outputs["tac_suv"].parent.mkdir(parents=True, exist_ok=True)
    with open(outputs["tac_suv"], "w") as fout:
        write_provenance_header(
            fout, subject_id, "s02_suv_tac.py", cfg.PIPELINE_VERSION,
            inputs=[str(p) for p in input_paths],
            parameters={
                "injected_dose_MBq": rc["dose_mbq"],
                "body_weight_kg": rc["weight_kg"],
                "suv_factor": round(suv_factor, 2),
            },
        )
        tac_suv.to_csv(fout, sep="\t", index=False, float_format="%.6f")
    log.info("Wrote: %s", outputs["tac_suv"])

    # Plot
    plt.style.use(cfg.FIGURE_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tac_suv["mid_time_min"], tac_suv["suv"],
            "o-", color="darkorange", markersize=4, linewidth=1.5)
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("SUV", fontsize=12)
    ax.set_title(f"{subject_id}: SUV TAC (dose={rc['dose_mbq']:.1f} MBq, weight={rc['weight_kg']:.1f} kg)",
                 fontsize=13)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Add plateau line
    if not np.isnan(plateau_suv):
        ax.axhline(plateau_suv, color="gray", linestyle="--", alpha=0.6,
                    label=f"Plateau (30-60 min): {plateau_suv:.3f}")
        ax.legend(fontsize=10)

    plt.tight_layout()
    outputs["fig_suv"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["fig_suv"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", outputs["fig_suv"])

    elapsed = time.time() - t0
    log.info("DONE s02: %.1f s elapsed", elapsed)
    return outputs


if __name__ == "__main__":
    import argparse
    from pipeline.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Step 2: SUV TAC")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(base_dir=Path(args.base_dir))
    setup_logging(args.subject, cfg.logs_dir(), verbose=args.verbose)
    run(args.subject, cfg, force=args.force)
