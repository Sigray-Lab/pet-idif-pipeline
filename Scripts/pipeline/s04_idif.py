"""Step 4: Image-derived input function (IDIF) from carotid/jugular vessels."""
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from pipeline.cache import check_outputs_current, log_skip, write_provenance_header
from pipeline.config import PipelineConfig

log = logging.getLogger("s04_idif")


def _find_arterial_peak(pet_data: np.ndarray, neck_mask: np.ndarray,
                        n_search_frames: int) -> tuple:
    """
    Find the arterial bolus peak frame.
    For each of the first N frames, compute the top-1% voxel mean within the neck mask.
    Returns (peak_frame_idx, top1_values_per_frame).
    """
    top1_vals = []
    neck_vox_count = int(np.sum(neck_mask))

    for f_idx in range(min(n_search_frames, pet_data.shape[3])):
        frame = pet_data[:, :, :, f_idx]
        vals = frame[neck_mask]
        # Top 1% threshold
        thresh = np.percentile(vals, 99)
        top_vals = vals[vals >= thresh]
        mean_top1 = float(np.mean(top_vals)) if len(top_vals) > 0 else 0.0
        top1_vals.append(mean_top1)
        log.debug("Frame %2d: top-1%% mean = %.1f Bq/mL (%d voxels)",
                  f_idx, mean_top1, len(top_vals))

    peak_idx = int(np.argmax(top1_vals))
    log.info("Arterial peak: frame %d, top-1%% mean = %.1f Bq/mL",
             peak_idx, top1_vals[peak_idx])

    return peak_idx, top1_vals


def _create_vascular_mask(pet_data: np.ndarray, neck_mask: np.ndarray,
                          frames_df: pd.DataFrame,
                          cfg: PipelineConfig) -> tuple:
    """
    Create vascular segmentation mask from first-pass summed image.

    Per-slice approach: for each z-slice that has neck mask coverage,
    select at least IDIF_MIN_VOXELS_PER_SLICE hottest voxels. This ensures
    vascular coverage along the entire z-axis of the search region.

    Returns (mask_3d, summed_image_3d).
    """
    # Find frames in the sum window
    starts = frames_df["start_s"].values
    ends = frames_df["end_s"].values
    in_window = []
    for i in range(len(starts)):
        if ends[i] > cfg.IDIF_SUM_WINDOW_START_S and starts[i] < cfg.IDIF_SUM_WINDOW_END_S:
            in_window.append(i)

    log.info("First-pass sum window: %.0f-%.0f s, frames %s",
             cfg.IDIF_SUM_WINDOW_START_S, cfg.IDIF_SUM_WINDOW_END_S, in_window)

    # Sum selected frames
    summed = np.sum(pet_data[:, :, :, in_window], axis=3)

    # Mask to neck region
    summed_masked = summed * neck_mask

    # Per-slice voxel selection: for each z-slice with neck mask coverage,
    # take the top N hottest voxels (at least IDIF_MIN_VOXELS_PER_SLICE)
    min_per_slice = cfg.IDIF_MIN_VOXELS_PER_SLICE
    mask_3d = np.zeros_like(neck_mask, dtype=np.uint8)
    nz = summed_masked.shape[2]

    total_voxels = 0
    slices_with_mask = 0

    for z in range(nz):
        slice_neck = neck_mask[:, :, z]
        n_neck_slice = int(np.sum(slice_neck))
        if n_neck_slice == 0:
            continue

        slices_with_mask += 1
        slice_vals = summed_masked[:, :, z]
        vals_in_neck = slice_vals[slice_neck > 0]

        # Select the hottest voxels in this slice
        n_select = max(min_per_slice, min_per_slice)
        n_select = min(n_select, n_neck_slice)  # cannot exceed available voxels

        # Find threshold to get at least n_select voxels
        sorted_vals = np.sort(vals_in_neck)[::-1]  # descending
        thresh = sorted_vals[n_select - 1]

        # Select voxels at or above the threshold in this slice
        slice_mask = (slice_vals >= thresh) & (slice_neck > 0)
        n_selected = int(np.sum(slice_mask))
        mask_3d[:, :, z] = slice_mask.astype(np.uint8)
        total_voxels += n_selected

        log.debug("z=%3d: neck=%d, selected=%d, thresh=%.0f",
                  z, n_neck_slice, n_selected, thresh)

    log.info("Per-slice selection: %d z-slices with neck mask, %d total voxels "
             "(min %d per slice)", slices_with_mask, total_voxels, min_per_slice)

    # Apply connected-component filtering to remove isolated noise voxels
    struct_26 = ndimage.generate_binary_structure(3, 3)
    labeled, n_labels = ndimage.label(mask_3d, structure=struct_26)
    sizes = ndimage.sum(mask_3d, labeled, range(1, n_labels + 1))

    filtered = np.zeros_like(mask_3d)
    n_retained = 0
    for label_idx in range(1, n_labels + 1):
        if sizes[label_idx - 1] >= cfg.IDIF_MIN_CLUSTER_SIZE:
            filtered[labeled == label_idx] = 1
            n_retained += 1

    n_filtered = int(np.sum(filtered))
    log.info("After CC filtering (min cluster=%d): %d clusters retained, %d voxels "
             "(removed %d isolated voxels)",
             cfg.IDIF_MIN_CLUSTER_SIZE, n_retained, n_filtered,
             total_voxels - n_filtered)

    return filtered, summed


def _qc_mask_figure(summed: np.ndarray, mask: np.ndarray, out_path: Path,
                     subject_id: str, cfg: PipelineConfig) -> None:
    """QC: IDIF mask overlay on summed first-pass image."""
    plt.style.use(cfg.FIGURE_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    com = ndimage.center_of_mass(mask)
    slices = [int(com[0]), int(com[1]), int(com[2])]

    views = ["Sagittal", "Coronal", "Axial"]
    for i, (title, si) in enumerate(zip(views, slices)):
        if i == 0:
            s_sl = summed[si, :, :]
            m_sl = mask[si, :, :]
        elif i == 1:
            s_sl = summed[:, si, :]
            m_sl = mask[:, si, :]
        else:
            s_sl = summed[:, :, si]
            m_sl = mask[:, :, si]

        axes[i].imshow(s_sl.T, cmap="gray", origin="lower", aspect="equal")
        overlay = np.ma.masked_where(m_sl.T == 0, m_sl.T)
        axes[i].imshow(overlay, cmap="spring", alpha=0.7, origin="lower", aspect="equal")
        axes[i].set_title(title, fontsize=12)
        axes[i].axis("off")

    fig.suptitle(f"{subject_id}: IDIF Mask on First-Pass Sum", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved mask QC: %s", out_path)


def run(subject_id: str, cfg: PipelineConfig, force: bool = False) -> dict:
    """
    Run step 4: Image-derived input function.
    4a: Arterial peak detection
    4b: Vascular mask creation
    4c: IDIF TAC extraction
    4d: QC metrics and figures
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("STEP s04: Image-derived input function (IDIF)")
    log.info("Subject: %s", subject_id)

    derived = cfg.derived_dir(subject_id)
    sub = f"sub-{subject_id}"

    # Inputs
    pet_path = derived / "PET1" / f"{sub}_pet.nii.gz"
    neck_mask_path = derived / "masks" / f"{sub}_space-PET_mask-neck-loose.nii.gz"
    frames_path = derived / "PET1" / f"{sub}_frames.tsv"
    brain_tac_path = cfg.outputs_dir() / f"{sub}_tac-raw.tsv"
    radiochem_path = cfg.radiochem_path()

    # Outputs
    outputs = {
        "idif_mask": derived / "masks" / f"{sub}_space-PET_mask-idif.nii.gz",
        "summed_fp": derived / "PET1" / f"{sub}_idif-summed-firstpass.nii.gz",
        "idif_tsv": cfg.outputs_dir() / f"{sub}_idif.tsv",
        "fig_idif": cfg.figures_dir() / f"{sub}_idif.png",
        "fig_idif_log": cfg.figures_dir() / f"{sub}_idif-log.png",
        "fig_mask_qc": cfg.figures_dir() / f"{sub}_idif-mask-qc.png",
        "fig_combined": cfg.figures_dir() / f"{sub}_tac-combined.png",
        "fig_combined_suv": cfg.figures_dir() / f"{sub}_tac-combined-suv.png",
        "fig_combined_suv_log": cfg.figures_dir() / f"{sub}_tac-combined-suv-log.png",
    }

    # Cache check
    input_paths = [pet_path, neck_mask_path, frames_path]
    output_paths = list(outputs.values())
    if not force and check_outputs_current(input_paths, output_paths):
        log_skip("s04_idif", output_paths)
        return outputs

    for inp in input_paths:
        if not inp.exists():
            raise FileNotFoundError(f"Required input missing: {inp}")
        log.info("Input: %s", inp)

    # Load PET
    pet_img = nib.load(pet_path)
    pet_data = pet_img.get_fdata()
    log.info("PET shape: %s", pet_data.shape)

    # Load neck mask
    neck_img = nib.load(neck_mask_path)
    neck_mask = neck_img.get_fdata() > 0
    n_neck = int(np.sum(neck_mask))
    log.info("Neck mask: %d voxels", n_neck)

    # Load frames
    frames = pd.read_csv(frames_path, sep="\t", comment="#")
    log.info("Frames: %d entries", len(frames))

    # === 4a: Arterial peak detection ===
    log.info("--- 4a: Arterial peak detection ---")
    peak_frame, top1_values = _find_arterial_peak(
        pet_data, neck_mask, cfg.IDIF_PEAK_SEARCH_FRAMES,
    )
    peak_time_s = frames["mid_time_s"].iloc[peak_frame]
    log.info("Arterial peak at frame %d (%.1f s = %.2f min)",
             peak_frame, peak_time_s, peak_time_s / 60)

    # === 4b: Create vascular mask ===
    log.info("--- 4b: Vascular mask creation ---")
    idif_mask, summed_fp = _create_vascular_mask(pet_data, neck_mask, frames, cfg)
    n_idif = int(np.sum(idif_mask))
    log.info("IDIF mask: %d voxels", n_idif)

    # Save IDIF mask
    mask_nii = nib.Nifti1Image(idif_mask.astype(np.float32), pet_img.affine, pet_img.header)
    outputs["idif_mask"].parent.mkdir(parents=True, exist_ok=True)
    nib.save(mask_nii, outputs["idif_mask"])
    log.info("Saved IDIF mask: %s", outputs["idif_mask"])

    # Save summed first-pass
    sum_nii = nib.Nifti1Image(summed_fp.astype(np.float32), pet_img.affine, pet_img.header)
    nib.save(sum_nii, outputs["summed_fp"])
    log.info("Saved summed first-pass: %s", outputs["summed_fp"])

    # === 4c: Extract IDIF TAC ===
    log.info("--- 4c: IDIF TAC extraction ---")

    # Check for manual mask override
    manual_mask_path = derived / "masks" / f"{sub}_space-PET_mask-idif-manual.nii.gz"
    if manual_mask_path.exists():
        log.info("Using MANUAL IDIF mask: %s", manual_mask_path)
        manual_img = nib.load(manual_mask_path)
        idif_mask_for_tac = manual_img.get_fdata() > 0
        n_manual = int(np.sum(idif_mask_for_tac))
        log.info("Manual mask: %d voxels", n_manual)
    else:
        log.info("Using auto-generated IDIF mask (%d voxels)", n_idif)
        idif_mask_for_tac = idif_mask > 0

    n_frames = pet_data.shape[3]
    idif_means = []
    idif_stds = []

    for f_idx in range(n_frames):
        vals = pet_data[:, :, :, f_idx][idif_mask_for_tac]
        idif_means.append(float(np.mean(vals)))
        idif_stds.append(float(np.std(vals)))
        log.debug("IDIF frame %2d: mean=%.1f, std=%.1f Bq/mL",
                  f_idx, idif_means[-1], idif_stds[-1])

    # Build IDIF dataframe
    idif_df = pd.DataFrame({
        "frame": frames["frame_index"].values,
        "start_s": frames["start_s"].values,
        "end_s": frames["end_s"].values,
        "mid_time_s": frames["mid_time_s"].values,
        "mid_time_min": frames["mid_time_min"].values,
        "mean_activity_Bq_per_mL": idif_means,
        "std_activity": idif_stds,
        "n_voxels": [int(np.sum(idif_mask_for_tac))] * n_frames,
    })

    # === 4d: QC metrics ===
    log.info("--- 4d: QC metrics ---")
    idif_peak = max(idif_means)
    idif_peak_idx = int(np.argmax(idif_means))
    idif_peak_time = frames["mid_time_min"].iloc[idif_peak_idx]

    # Load brain TAC for comparison
    brain_tac = None
    brain_peak = np.nan
    brain_peak_time = np.nan
    ratio = np.nan
    if brain_tac_path.exists():
        brain_tac = pd.read_csv(brain_tac_path, sep="\t", comment="#")
        brain_peak = float(brain_tac["mean_activity_Bq_per_mL"].max())
        brain_peak_idx = int(brain_tac["mean_activity_Bq_per_mL"].idxmax())
        brain_peak_time = float(brain_tac["mid_time_min"].iloc[brain_peak_idx])
        ratio = idif_peak / brain_peak if brain_peak > 0 else np.nan

    log.info("IDIF peak: %.1f Bq/mL at %.2f min", idif_peak, idif_peak_time)
    log.info("Brain peak: %.1f Bq/mL at %.2f min", brain_peak, brain_peak_time)
    log.info("IDIF/brain peak ratio: %.2f", ratio)

    if ratio < 2.0:
        log.warning(
            "IDIF/brain peak ratio (%.2f) is < 2.0. "
            "The IDIF mask may contain tissue contamination. Manual QC recommended.",
            ratio,
        )

    # Save IDIF TSV
    outputs["idif_tsv"].parent.mkdir(parents=True, exist_ok=True)
    with open(outputs["idif_tsv"], "w") as fout:
        write_provenance_header(
            fout, subject_id, "s04_idif.py", cfg.PIPELINE_VERSION,
            inputs=[str(p) for p in input_paths],
            parameters={
                "n_idif_voxels": int(np.sum(idif_mask_for_tac)),
                "peak_search_frames": cfg.IDIF_PEAK_SEARCH_FRAMES,
                "sum_window_s": f"{cfg.IDIF_SUM_WINDOW_START_S}-{cfg.IDIF_SUM_WINDOW_END_S}",
                "min_voxels_per_slice": cfg.IDIF_MIN_VOXELS_PER_SLICE,
                "min_cluster_size": cfg.IDIF_MIN_CLUSTER_SIZE,
            },
            extra_lines=[
                f"mask_type: {'manual' if manual_mask_path.exists() else 'auto'}",
                f"idif_peak_Bq_per_mL: {idif_peak:.1f}",
                f"idif_peak_time_min: {idif_peak_time:.2f}",
                f"brain_peak_Bq_per_mL: {brain_peak:.1f}",
                f"idif_brain_ratio: {ratio:.2f}",
            ],
        )
        idif_df.to_csv(fout, sep="\t", index=False, float_format="%.4f")
    log.info("Wrote: %s", outputs["idif_tsv"])

    # --- Figures ---
    plt.style.use(cfg.FIGURE_STYLE)

    # IDIF plot (linear)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idif_df["mid_time_min"], idif_df["mean_activity_Bq_per_mL"],
            "o-", color="crimson", markersize=4, linewidth=1.5, label="IDIF")
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Activity (Bq/mL)", fontsize=12)
    ax.set_title(f"{subject_id}: IDIF TAC", fontsize=14)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)
    plt.tight_layout()
    outputs["fig_idif"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["fig_idif"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", outputs["fig_idif"])

    # IDIF plot (log y)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(idif_df["mid_time_min"], idif_df["mean_activity_Bq_per_mL"],
            "o-", color="crimson", markersize=4, linewidth=1.5, label="IDIF")
    ax.set_yscale("log")
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Activity (Bq/mL, log scale)", fontsize=12)
    ax.set_title(f"{subject_id}: IDIF TAC (log scale)", fontsize=14)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(outputs["fig_idif_log"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", outputs["fig_idif_log"])

    # Mask QC
    _qc_mask_figure(summed_fp, idif_mask, outputs["fig_mask_qc"], subject_id, cfg)

    # Combined plot (brain + IDIF) in Bq/mL
    if brain_tac is not None:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(brain_tac["mid_time_min"], brain_tac["mean_activity_Bq_per_mL"],
                "o-", color="steelblue", markersize=3, linewidth=1.5, label="Brain")
        ax.plot(idif_df["mid_time_min"], idif_df["mean_activity_Bq_per_mL"],
                "s-", color="crimson", markersize=3, linewidth=1.5, label="IDIF")
        ax.set_xlabel("Time (min)", fontsize=12)
        ax.set_ylabel("Activity (Bq/mL)", fontsize=12)
        ax.set_title(f"{subject_id}: Brain TAC + IDIF", fontsize=14)
        ax.set_xlim(left=0)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig.savefig(outputs["fig_combined"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved: %s", outputs["fig_combined"])
    else:
        log.warning("Brain TAC not found, skipping combined plot")

    # Combined plot (brain + IDIF) in SUV scale
    if brain_tac is not None and radiochem_path.exists():
        rc_df = pd.read_csv(radiochem_path)
        rc_row = rc_df[rc_df["id"] == subject_id]
        if not rc_row.empty:
            rc_row = rc_row.iloc[0]
            dose_bq = float(rc_row["injected_MBq"]) * 1e6
            weight_g = float(rc_row["weight_kg"]) * 1000.0
            suv_factor = dose_bq / weight_g

            brain_suv = brain_tac["mean_activity_Bq_per_mL"].values / suv_factor
            idif_suv = idif_df["mean_activity_Bq_per_mL"].values / suv_factor

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(brain_tac["mid_time_min"], brain_suv,
                    "o-", color="steelblue", markersize=3, linewidth=1.5, label="Brain")
            ax.plot(idif_df["mid_time_min"], idif_suv,
                    "s-", color="crimson", markersize=3, linewidth=1.5, label="IDIF")
            ax.set_xlabel("Time (min)", fontsize=12)
            ax.set_ylabel("SUV", fontsize=12)
            ax.set_title(f"{subject_id}: Brain TAC + IDIF (SUV)", fontsize=14)
            ax.set_xlim(left=0)
            ax.legend(fontsize=10)
            plt.tight_layout()
            fig.savefig(outputs["fig_combined_suv"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved: %s", outputs["fig_combined_suv"])

            # SUV combined plot with log y-axis
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(brain_tac["mid_time_min"], brain_suv,
                    "o-", color="steelblue", markersize=3, linewidth=1.5, label="Brain")
            ax.plot(idif_df["mid_time_min"], idif_suv,
                    "s-", color="crimson", markersize=3, linewidth=1.5, label="IDIF")
            ax.set_yscale("log")
            ax.set_xlabel("Time (min)", fontsize=12)
            ax.set_ylabel("SUV (log scale)", fontsize=12)
            ax.set_title(f"{subject_id}: Brain TAC + IDIF (SUV, log scale)", fontsize=14)
            ax.set_xlim(left=0)
            ax.legend(fontsize=10)
            plt.tight_layout()
            fig.savefig(outputs["fig_combined_suv_log"], dpi=cfg.FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved: %s", outputs["fig_combined_suv_log"])
        else:
            log.warning("Subject not found in Radiochem.csv, skipping SUV combined plot")
    else:
        log.warning("Brain TAC or Radiochem.csv not found, skipping SUV combined plot")

    elapsed = time.time() - t0
    log.info("DONE s04: %.1f s elapsed", elapsed)
    return outputs


if __name__ == "__main__":
    import argparse
    from pipeline.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Step 4: IDIF")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(base_dir=Path(args.base_dir))
    setup_logging(args.subject, cfg.logs_dir(), verbose=args.verbose)
    run(args.subject, cfg, force=args.force)
