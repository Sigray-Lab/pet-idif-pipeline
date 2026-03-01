"""Step 0: DICOM-to-NIfTI conversion, mask copy, frame timing extraction."""
import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from pipeline.cache import check_outputs_current, log_skip, write_provenance_header
from pipeline.config import PipelineConfig

log = logging.getLogger("s00_dcm2nii")

# Expected frame protocol
EXPECTED_DURATIONS = (
    [10] * 6 + [20] * 6 + [30] * 4 + [60] * 10
    + [180] * 5 + [300] * 6 + [600] * 3 + [900] * 2 + [1800] * 1
)


def _run_dcm2niix(dicom_dir: Path, out_dir: Path, prefix: str) -> Path:
    """Run dcm2niix and return path to the produced NIfTI."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "dcm2niix", "-z", "y", "-f", prefix,
        "-o", str(out_dir), str(dicom_dir),
    ]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    log.debug("dcm2niix stdout:\n%s", result.stdout)
    if result.returncode != 0:
        log.error("dcm2niix stderr:\n%s", result.stderr)
        raise RuntimeError(f"dcm2niix failed with return code {result.returncode}")
    nii_path = out_dir / f"{prefix}.nii.gz"
    if not nii_path.exists():
        raise FileNotFoundError(f"Expected NIfTI not found: {nii_path}")
    return nii_path


def _build_frames_tsv(json_path: Path, tsv_path: Path, subject_id: str,
                      cfg: PipelineConfig) -> pd.DataFrame:
    """Parse dcm2niix JSON sidecar and build _frames.tsv."""
    with open(json_path) as f:
        meta = json.load(f)

    starts = meta["FrameTimesStart"]
    durations = meta["FrameDuration"]
    n_frames = len(starts)

    log.info("JSON sidecar: %d frames, total duration %.0f s",
             n_frames, starts[-1] + durations[-1])

    if n_frames != cfg.EXPECTED_N_FRAMES:
        log.warning("Frame count mismatch: got %d, expected %d",
                     n_frames, cfg.EXPECTED_N_FRAMES)

    # Verify against expected protocol
    if durations != EXPECTED_DURATIONS:
        log.warning("Frame durations do not match expected protocol")
        for i, (got, exp) in enumerate(zip(durations, EXPECTED_DURATIONS)):
            if got != exp:
                log.warning("  Frame %d: got %d s, expected %d s", i, got, exp)
    else:
        log.info("Frame durations match expected protocol (43 frames, 150 min)")

    ends = [s + d for s, d in zip(starts, durations)]
    mids_s = [s + d / 2.0 for s, d in zip(starts, durations)]
    mids_min = [m / 60.0 for m in mids_s]

    df = pd.DataFrame({
        "frame_index": range(n_frames),
        "start_s": starts,
        "end_s": ends,
        "duration_s": durations,
        "mid_time_s": mids_s,
        "mid_time_min": mids_min,
    })

    with open(tsv_path, "w") as fout:
        write_provenance_header(
            fout, subject_id, "s00_dcm2nii.py", cfg.PIPELINE_VERSION,
            inputs=[str(json_path)],
            parameters={"n_frames": n_frames},
        )
        df.to_csv(fout, sep="\t", index=False, float_format="%.4f")

    log.info("Wrote frames TSV: %s (%d frames)", tsv_path, n_frames)
    return df


def _copy_mask(src: Path, dst: Path, pet_img: nib.Nifti1Image) -> None:
    """Copy a mask and validate dimensions match PET."""
    if not src.exists():
        raise FileNotFoundError(f"Mask not found: {src}")

    mask_img = nib.load(src)
    pet_shape_3d = pet_img.shape[:3]
    mask_shape = mask_img.shape[:3]

    if mask_shape != pet_shape_3d:
        raise ValueError(
            f"Mask shape {mask_shape} does not match PET {pet_shape_3d}: {src}"
        )

    # Check affine similarity
    aff_diff = np.abs(mask_img.affine - pet_img.affine[:mask_img.affine.shape[0], :])
    if np.max(aff_diff) > 0.1:
        log.warning("Affine mismatch between mask and PET (max diff %.3f mm): %s",
                     np.max(aff_diff), src.name)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    n_vox = int(np.sum(mask_img.get_fdata() > 0))
    vox_vol = np.abs(np.linalg.det(mask_img.affine[:3, :3]))
    vol_ml = n_vox * vox_vol / 1000.0
    log.info("Copied mask: %s -> %s (%d voxels, %.1f mL)",
             src.name, dst.name, n_vox, vol_ml)


def _process_mr(raw_mr_path: Path, out_path: Path) -> None:
    """Load MR NIfTI, squeeze singleton 4th dim if present, save as .nii.gz."""
    img = nib.load(raw_mr_path)
    data = img.get_fdata()
    log.info("MR input shape: %s, dtype: %s, voxel: %s mm",
             data.shape, img.get_data_dtype(), np.abs(np.diag(img.affine)[:3]))

    if data.ndim == 4 and data.shape[3] == 1:
        log.info("Squeezing singleton 4th dimension from MR")
        data = data[:, :, :, 0]
        img = nib.Nifti1Image(data, img.affine, img.header)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, out_path)
    log.info("Wrote MR: %s, shape: %s", out_path, data.shape)


def run(subject_id: str, cfg: PipelineConfig, force: bool = False) -> dict:
    """
    Run step 0: DICOM to NIfTI conversion, mask copy, frame timing.
    Returns dict of output paths.
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("STEP s00: DICOM-to-NIfTI conversion")
    log.info("Subject: %s", subject_id)

    raw = cfg.raw_dir(subject_id)
    derived = cfg.derived_dir(subject_id)
    sub = f"sub-{subject_id}"

    # Define all outputs
    outputs = {
        "pet_nii": derived / "PET1" / f"{sub}_pet.nii.gz",
        "pet_json": derived / "PET1" / f"{sub}_pet.json",
        "frames_tsv": derived / "PET1" / f"{sub}_frames.tsv",
        "ct_nii": derived / "CT1" / f"{sub}_ct.nii.gz",
        "mr_nii": derived / "MR" / f"{sub}_mr.nii.gz",
        "mask_brain": derived / "masks" / f"{sub}_space-PET_mask-brain.nii.gz",
        "mask_neck": derived / "masks" / f"{sub}_space-PET_mask-neck-loose.nii.gz",
    }

    # Define inputs
    inputs = [
        raw / "PET1",
        raw / "CT1",
        raw / "MR" / "T1w.nii",
        raw / "CT_Scan_3_resliced_to_PET_mask.nii.gz",
        raw / "CT_Scan_3_resliced_to_PET_loose_neck_mask_FINAL.nii.gz",
    ]

    # Cache check (use PET nii as proxy for input mtime since dirs are tricky)
    output_paths = list(outputs.values())
    if not force and all(Path(o).exists() for o in output_paths):
        log_skip("s00_dcm2nii", output_paths)
        return outputs

    log.info("Inputs:")
    for inp in inputs:
        exists = inp.exists()
        log.info("  %s [%s]", inp, "OK" if exists else "MISSING")
        if not exists:
            raise FileNotFoundError(f"Required input missing: {inp}")

    # --- PET conversion ---
    log.info("--- Converting PET DICOM ---")
    pet_out_dir = derived / "PET1"
    pet_nii = _run_dcm2niix(raw / "PET1", pet_out_dir, f"{sub}_pet")
    pet_img = nib.load(pet_nii)
    log.info("PET shape: %s, voxel: %s mm, dtype: %s",
             pet_img.shape,
             tuple(np.round(np.abs(np.diag(pet_img.affine)[:3]), 2)),
             pet_img.get_data_dtype())

    if len(pet_img.shape) != 4:
        raise ValueError(f"PET is not 4D: shape={pet_img.shape}")
    if pet_img.shape[3] != cfg.EXPECTED_N_FRAMES:
        log.warning("PET has %d frames, expected %d",
                     pet_img.shape[3], cfg.EXPECTED_N_FRAMES)

    # JSON sidecar (dcm2niix creates it alongside the NIfTI)
    dcm2niix_json = pet_out_dir / f"{sub}_pet.json"
    if not dcm2niix_json.exists():
        raise FileNotFoundError(f"JSON sidecar not found: {dcm2niix_json}")

    # Build frames TSV
    _build_frames_tsv(dcm2niix_json, outputs["frames_tsv"], subject_id, cfg)

    # --- CT conversion ---
    log.info("--- Converting CT DICOM ---")
    ct_out_dir = derived / "CT1"
    _run_dcm2niix(raw / "CT1", ct_out_dir, f"{sub}_ct")
    ct_img = nib.load(outputs["ct_nii"])
    log.info("CT shape: %s, voxel: %s mm",
             ct_img.shape,
             tuple(np.round(np.abs(np.diag(ct_img.affine)[:3]), 2)))

    # --- MR processing ---
    log.info("--- Processing MR NIfTI ---")
    mr_raw = raw / "MR" / "T1w.nii"
    _process_mr(mr_raw, outputs["mr_nii"])

    # --- Copy masks ---
    log.info("--- Copying and validating masks ---")
    _copy_mask(
        raw / "CT_Scan_3_resliced_to_PET_mask.nii.gz",
        outputs["mask_brain"],
        pet_img,
    )
    _copy_mask(
        raw / "CT_Scan_3_resliced_to_PET_loose_neck_mask_FINAL.nii.gz",
        outputs["mask_neck"],
        pet_img,
    )

    elapsed = time.time() - t0
    log.info("DONE s00: %.1f s elapsed", elapsed)
    log.info("Outputs: %s", ", ".join(str(v) for v in outputs.values()))
    return outputs


if __name__ == "__main__":
    import argparse
    from pipeline.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Step 0: DICOM to NIfTI")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(base_dir=Path(args.base_dir))
    setup_logging(args.subject, cfg.logs_dir(), verbose=args.verbose)
    run(args.subject, cfg, force=args.force)
