"""Step 3: %ID (percent injected dose) TAC and summary table."""
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

log = logging.getLogger("s03_percent_id")


def run(subject_id: str, cfg: PipelineConfig, force: bool = False) -> dict:
    """
    Compute %ID TAC and summary statistics.

    Uses:
    - Brain mask TAC (from s01): provides mean concentration (Bq/mL)
    - Brain volume from MR-derived mask (from s00b/s00c, via s01 TSV header)

    %ID = (mean_Bq_per_mL * brain_volume_mL) / injected_dose_Bq * 100
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("STEP s03: Percent injected dose (%ID)")
    log.info("Subject: %s", subject_id)

    sub = f"sub-{subject_id}"

    # Inputs
    tac_raw_path = cfg.outputs_dir() / f"{sub}_tac-raw.tsv"
    radiochem_path = cfg.radiochem_path()

    # Outputs
    outputs = {
        "tac_pctid": cfg.outputs_dir() / f"{sub}_tac-pctID.tsv",
        "summary": cfg.outputs_dir() / f"{sub}_summary.tsv",
        "fig_pctid": cfg.figures_dir() / f"{sub}_tac-pctID.png",
    }

    # Cache check
    input_paths = [tac_raw_path, radiochem_path]
    output_paths = list(outputs.values())
    if not force and check_outputs_current(input_paths, output_paths):
        log_skip("s03_percent_id", output_paths)
        return outputs

    for inp in input_paths:
        if not inp.exists():
            raise FileNotFoundError(f"Required input missing: {inp}")
        log.info("Input: %s", inp)

    # Load raw TAC (mean concentration from crude brain mask)
    tac_raw = pd.read_csv(tac_raw_path, sep="\t", comment="#")
    log.info("Raw TAC: %d frames loaded", len(tac_raw))

    # Load radiochem
    rc_df = pd.read_csv(radiochem_path)
    row = rc_df[rc_df["id"] == subject_id]
    if row.empty:
        raise ValueError(f"Subject {subject_id} not found in {radiochem_path}")
    row = row.iloc[0]
    dose_mbq = float(row["injected_MBq"])
    weight_kg = float(row["weight_kg"])
    dose_bq = dose_mbq * 1e6
    weight_g = weight_kg * 1000.0

    log.info("Injected dose: %.2f MBq (%.0f Bq)", dose_mbq, dose_bq)
    log.info("Body weight: %.1f kg (%.0f g)", weight_kg, weight_g)

    # Extract brain volume from s01 raw TAC provenance header
    brain_vol_ml = None
    with open(tac_raw_path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if "roi_volume_mL" in line:
                brain_vol_ml = float(line.split(":")[-1].strip())
                break
    if brain_vol_ml is None:
        raise ValueError(
            f"Could not find roi_volume_mL in provenance header of {tac_raw_path}. "
            "Ensure s01 ran with a valid brain mask."
        )
    log.info("Brain volume (from mask): %.1f mL", brain_vol_ml)

    # Compute %ID
    # %ID = (mean_Bq_per_mL * brain_volume_mL) / dose_Bq * 100
    mean_activity = tac_raw["mean_activity_Bq_per_mL"].values
    pct_id = (mean_activity * brain_vol_ml) / dose_bq * 100.0

    # Also compute SUV for summary
    suv_factor = dose_bq / weight_g
    suv = mean_activity / suv_factor

    # Build output dataframe
    tac_pctid = pd.DataFrame({
        "frame": tac_raw["frame"].values,
        "mid_time_min": tac_raw["mid_time_min"].values,
        "mean_activity_Bq_per_mL": mean_activity,
        "pct_ID": pct_id,
    })

    # Save %ID TSV
    outputs["tac_pctid"].parent.mkdir(parents=True, exist_ok=True)
    with open(outputs["tac_pctid"], "w") as fout:
        write_provenance_header(
            fout, subject_id, "s03_percent_id.py", cfg.PIPELINE_VERSION,
            inputs=[str(p) for p in input_paths],
            parameters={
                "injected_dose_MBq": dose_mbq,
                "brain_volume_mL": brain_vol_ml,
            },
            extra_lines=[
                f"brain_volume_source: MR-derived mask ({brain_vol_ml:.1f} mL)",
                f"mean_concentration_source: brain mask (s01 raw TAC)",
            ],
        )
        tac_pctid.to_csv(fout, sep="\t", index=False, float_format="%.6f")
    log.info("Wrote: %s", outputs["tac_pctid"])

    # --- Summary statistics ---
    peak_idx = int(np.argmax(pct_id))
    peak_pctid = pct_id[peak_idx]
    peak_time = tac_raw["mid_time_min"].iloc[peak_idx]

    # Plateau: 30-60 min
    t_min = tac_raw["mid_time_min"].values
    plateau_mask = (t_min >= 30) & (t_min <= 60)
    plateau_pctid = float(np.mean(pct_id[plateau_mask])) if np.any(plateau_mask) else np.nan
    plateau_suv = float(np.mean(suv[plateau_mask])) if np.any(plateau_mask) else np.nan

    last_pctid = pct_id[-1]

    log.info("Peak %%ID: %.4f%% at %.1f min", peak_pctid, peak_time)
    log.info("Plateau %%ID (30-60 min): %.4f%%", plateau_pctid)
    log.info("Last frame %%ID: %.4f%%", last_pctid)
    log.info("Plateau SUV (30-60 min): %.4f", plateau_suv)

    summary = pd.DataFrame([{
        "subject": subject_id,
        "peak_pct_ID": round(peak_pctid, 6),
        "peak_time_min": round(peak_time, 2),
        "plateau_pct_ID_30_60min": round(plateau_pctid, 6),
        "last_frame_pct_ID": round(last_pctid, 6),
        "roi_volume_mL": brain_vol_ml,
        "injected_dose_MBq": dose_mbq,
        "body_weight_kg": weight_kg,
        "suv_plateau": round(plateau_suv, 4),
    }])

    with open(outputs["summary"], "w") as fout:
        write_provenance_header(
            fout, subject_id, "s03_percent_id.py", cfg.PIPELINE_VERSION,
            inputs=[str(p) for p in input_paths],
            parameters={
                "brain_volume_mL": brain_vol_ml,
                "volume_source": f"MR-derived mask ({brain_vol_ml:.1f} mL)",
            },
        )
        summary.to_csv(fout, sep="\t", index=False, float_format="%.6f")
    log.info("Wrote summary: %s", outputs["summary"])

    # --- Plot ---
    plt.style.use(cfg.FIGURE_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tac_pctid["mid_time_min"], tac_pctid["pct_ID"],
            "o-", color="forestgreen", markersize=4, linewidth=1.5)
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("%ID", fontsize=12)
    ax.set_title(
        f"{subject_id}: %ID TAC "
        f"(dose={dose_mbq:.1f} MBq, brain vol={brain_vol_ml:.1f} mL)",
        fontsize=13,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if not np.isnan(plateau_pctid):
        ax.axhline(plateau_pctid, color="gray", linestyle="--", alpha=0.6,
                    label=f"Plateau (30-60 min): {plateau_pctid:.4f}%")
        ax.legend(fontsize=10)

    plt.tight_layout()
    outputs["fig_pctid"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["fig_pctid"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", outputs["fig_pctid"])

    elapsed = time.time() - t0
    log.info("DONE s03: %.1f s elapsed", elapsed)
    return outputs


if __name__ == "__main__":
    import argparse
    from pipeline.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Step 3: %ID TAC")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(base_dir=Path(args.base_dir))
    setup_logging(args.subject, cfg.logs_dir(), verbose=args.verbose)
    run(args.subject, cfg, force=args.force)
