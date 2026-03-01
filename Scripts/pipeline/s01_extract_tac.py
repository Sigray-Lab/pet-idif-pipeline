"""Step 1: Extract raw time-activity curve from brain mask."""
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from pipeline.cache import check_outputs_current, log_skip, write_provenance_header
from pipeline.config import PipelineConfig

log = logging.getLogger("s01_extract_tac")


def run(subject_id: str, cfg: PipelineConfig, force: bool = False) -> dict:
    """
    Extract mean activity (Bq/mL) per frame within the conservative brain mask.
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("STEP s01: Extract raw TAC")
    log.info("Subject: %s", subject_id)

    derived = cfg.derived_dir(subject_id)
    sub = f"sub-{subject_id}"

    # Inputs
    pet_path = derived / "PET1" / f"{sub}_pet.nii.gz"
    # Prefer MR-derived whole-brain mask (from s00c) over manual mask
    mask_whole = derived / "masks" / f"{sub}_space-PET_mask-brain-whole.nii.gz"
    mask_manual = derived / "masks" / f"{sub}_space-PET_mask-brain.nii.gz"
    if mask_whole.exists():
        mask_path = mask_whole
        log.info("Using MR-derived whole-brain mask: %s", mask_path)
    else:
        mask_path = mask_manual
        log.info("Using manual brain mask (MR-derived not available): %s", mask_path)
    frames_path = derived / "PET1" / f"{sub}_frames.tsv"

    # Outputs
    outputs = {
        "tac_raw": cfg.outputs_dir() / f"{sub}_tac-raw.tsv",
        "fig_raw": cfg.figures_dir() / f"{sub}_tac-raw.png",
    }

    # Cache check
    input_paths = [pet_path, mask_path, frames_path]
    output_paths = list(outputs.values())
    if not force and check_outputs_current(input_paths, output_paths):
        log_skip("s01_extract_tac", output_paths)
        return outputs

    for inp in input_paths:
        if not inp.exists():
            raise FileNotFoundError(f"Required input missing: {inp}")
        log.info("Input: %s", inp)

    # Load PET
    pet_img = nib.load(pet_path)
    pet_data = pet_img.get_fdata()
    vox_size = tuple(np.round(np.abs(np.diag(pet_img.affine)[:3]), 4))
    log.info("PET shape: %s, voxel: %s mm, dtype: %s",
             pet_data.shape, vox_size, pet_img.get_data_dtype())

    n_frames = pet_data.shape[3]
    if n_frames != cfg.EXPECTED_N_FRAMES:
        log.warning("PET has %d frames, expected %d", n_frames, cfg.EXPECTED_N_FRAMES)

    # Load mask
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata() > 0
    n_voxels = int(np.sum(mask_data))
    vox_vol_mm3 = float(np.abs(np.linalg.det(mask_img.affine[:3, :3])))
    vol_ml = n_voxels * vox_vol_mm3 / 1000.0
    log.info("Brain mask: %d voxels, volume = %.2f mL", n_voxels, vol_ml)

    if mask_data.shape != pet_data.shape[:3]:
        raise ValueError(
            f"Mask shape {mask_data.shape} does not match PET {pet_data.shape[:3]}"
        )

    # Load frames
    frames = pd.read_csv(frames_path, sep="\t", comment="#")
    if len(frames) != n_frames:
        raise ValueError(
            f"Frame count mismatch: frames.tsv has {len(frames)}, PET has {n_frames}"
        )

    # Extract TAC
    log.info("Extracting mean activity per frame...")
    means = []
    stds = []
    mins = []
    maxs = []

    for f_idx in range(n_frames):
        frame_data = pet_data[:, :, :, f_idx]
        roi_vals = frame_data[mask_data]
        mean_val = float(np.mean(roi_vals))
        std_val = float(np.std(roi_vals))
        min_val = float(np.min(roi_vals))
        max_val = float(np.max(roi_vals))
        means.append(mean_val)
        stds.append(std_val)
        mins.append(min_val)
        maxs.append(max_val)

        n_zero = int(np.sum(roi_vals == 0))
        if n_zero > 0:
            pct_zero = 100.0 * n_zero / n_voxels
            log.warning("Frame %d has %d zero-valued voxels inside mask (%.1f%%)",
                        f_idx, n_zero, pct_zero)

        log.debug("Frame %2d: mean=%.1f, std=%.1f, min=%.1f, max=%.1f Bq/mL",
                  f_idx, mean_val, std_val, min_val, max_val)

    # Build output dataframe
    tac_df = pd.DataFrame({
        "frame": frames["frame_index"].values,
        "start_s": frames["start_s"].values,
        "end_s": frames["end_s"].values,
        "mid_time_s": frames["mid_time_s"].values,
        "mid_time_min": frames["mid_time_min"].values,
        "mean_activity_Bq_per_mL": means,
        "std_activity": stds,
        "min_activity": mins,
        "max_activity": maxs,
        "n_voxels": [n_voxels] * n_frames,
    })

    # Save TSV
    outputs["tac_raw"].parent.mkdir(parents=True, exist_ok=True)
    with open(outputs["tac_raw"], "w") as fout:
        write_provenance_header(
            fout, subject_id, "s01_extract_tac.py", cfg.PIPELINE_VERSION,
            inputs=[str(p) for p in input_paths],
            parameters={},
            extra_lines=[f"roi_volume_mL: {vol_ml:.2f}", f"n_voxels: {n_voxels}"],
        )
        tac_df.to_csv(fout, sep="\t", index=False, float_format="%.4f")
    log.info("Wrote: %s (%d frames)", outputs["tac_raw"], n_frames)

    # Summary stats
    peak_idx = int(np.argmax(means))
    log.info("Peak activity: %.1f Bq/mL at frame %d (%.1f min)",
             means[peak_idx], peak_idx, frames["mid_time_min"].iloc[peak_idx])

    # Plot
    plt.style.use(cfg.FIGURE_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tac_df["mid_time_min"], tac_df["mean_activity_Bq_per_mL"],
            "o-", color="steelblue", markersize=4, linewidth=1.5)
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Mean Activity (Bq/mL)", fontsize=12)
    ax.set_title(f"{subject_id}: Raw Brain TAC", fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    outputs["fig_raw"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["fig_raw"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", outputs["fig_raw"])

    elapsed = time.time() - t0
    log.info("DONE s01: %.1f s elapsed", elapsed)
    return outputs


if __name__ == "__main__":
    import argparse
    from pipeline.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Step 1: Extract raw TAC")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(base_dir=Path(args.base_dir))
    setup_logging(args.subject, cfg.logs_dir(), verbose=args.verbose)
    run(args.subject, cfg, force=args.force)
