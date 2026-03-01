"""Step 0c: CT cropping, MR-to-CT rigid coregistration, and mask resampling to PET space."""
import logging
import time
from pathlib import Path

import ants
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

from pipeline.cache import check_outputs_current, log_skip
from pipeline.config import PipelineConfig

log = logging.getLogger("s00c_coregister")


def _crop_ct(ct_ants, cfg: PipelineConfig):
    """Crop CT to head region using configured voxel indices.

    The full CT (644x644x631) includes the entire body. Cropping to the head
    region improves registration quality and speed.

    Returns a new ANTs image with updated origin to preserve world coordinates.
    """
    lr_start, lr_end = cfg.CT_CROP_LR
    ap_start, ap_end = cfg.CT_CROP_AP
    z_start, z_end = cfg.CT_CROP_Z

    ct_data = ct_ants.numpy()
    log.info("CT before crop: shape=%s", ct_data.shape)
    log.info("Crop bounds: LR=[%d:%d], AP=[%d:%d], Z=[%d:%d]",
             lr_start, lr_end, ap_start, ap_end, z_start, z_end)

    # Crop the array
    cropped = ct_data[lr_start:lr_end, ap_start:ap_end, z_start:z_end]
    log.info("CT after crop: shape=%s", cropped.shape)

    # Compute new origin: shift by crop offset in world coordinates
    spacing = np.array(ct_ants.spacing)
    direction = np.array(ct_ants.direction)
    origin = np.array(ct_ants.origin)
    offset_voxels = np.array([lr_start, ap_start, z_start], dtype=float)
    new_origin = origin + direction @ (offset_voxels * spacing)

    ct_cropped = ants.from_numpy(
        cropped,
        origin=new_origin.tolist(),
        spacing=ct_ants.spacing,
        direction=ct_ants.direction,
    )
    log.info("CT cropped origin: %s (was %s)", ct_cropped.origin, ct_ants.origin)
    return ct_cropped


def _qc_registration(ct_ants, mr_in_ct, out_path: Path,
                      subject_id: str, cfg: PipelineConfig) -> None:
    """QC figure: MR warped into CT space, overlaid on CT."""
    plt.style.use(cfg.FIGURE_STYLE)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    ct_data = ct_ants.numpy()
    mr_data = mr_in_ct.numpy()

    # Find center of brain (use MR intensity to find center)
    mr_thresh = mr_data > np.percentile(mr_data[mr_data > 0], 20) if np.any(mr_data > 0) else mr_data > 0
    if np.any(mr_thresh):
        com = ndimage.center_of_mass(mr_thresh)
    else:
        com = [s // 2 for s in ct_data.shape]
    slices = [int(com[0]), int(com[1]), int(com[2])]

    views = ["Sagittal", "Coronal", "Axial"]
    for i, (title, si) in enumerate(zip(views, slices)):
        if i == 0:
            ct_sl = ct_data[si, :, :]
            mr_sl = mr_data[si, :, :]
        elif i == 1:
            ct_sl = ct_data[:, si, :]
            mr_sl = mr_data[:, si, :]
        else:
            ct_sl = ct_data[:, :, si]
            mr_sl = mr_data[:, :, si]

        # Top row: CT only
        axes[0, i].imshow(ct_sl.T, cmap="gray", origin="lower", aspect="equal",
                          vmin=-100, vmax=300)
        axes[0, i].set_title(f"{title} (CT)", fontsize=12)
        axes[0, i].axis("off")

        # Bottom row: CT with MR overlay
        axes[1, i].imshow(ct_sl.T, cmap="gray", origin="lower", aspect="equal",
                          vmin=-100, vmax=300)
        mr_overlay = np.ma.masked_where(mr_sl.T < np.percentile(mr_sl[mr_sl > 0], 10)
                                         if np.any(mr_sl > 0) else True,
                                         mr_sl.T)
        axes[1, i].imshow(mr_overlay, cmap="hot", alpha=0.35, origin="lower", aspect="equal")
        axes[1, i].set_title(f"{title} (CT + MR overlay)", fontsize=12)
        axes[1, i].axis("off")

    fig.suptitle(f"{subject_id}: MR-to-CT Coregistration QC", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved registration QC: %s", out_path)


def _qc_mask_on_pet(pet_path: Path, mask_pet_path: Path, out_path: Path,
                     subject_id: str, cfg: PipelineConfig) -> None:
    """QC figure: MR-derived brain mask overlaid on summed PET."""
    plt.style.use(cfg.FIGURE_STYLE)

    pet_img = nib.load(pet_path)
    pet_data = pet_img.get_fdata()
    # Sum a few mid-frames for a good anatomical image
    n_frames = pet_data.shape[3]
    mid_start = max(0, n_frames // 3)
    mid_end = min(n_frames, 2 * n_frames // 3)
    pet_sum = np.sum(pet_data[:, :, :, mid_start:mid_end], axis=3)

    mask_img = nib.load(mask_pet_path)
    mask_data = mask_img.get_fdata()

    com = ndimage.center_of_mass(mask_data > 0)
    slices = [int(com[0]), int(com[1]), int(com[2])]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = ["Sagittal", "Coronal", "Axial"]

    for i, (title, si) in enumerate(zip(views, slices)):
        if i == 0:
            pet_sl = pet_sum[si, :, :]
            mask_sl = mask_data[si, :, :]
        elif i == 1:
            pet_sl = pet_sum[:, si, :]
            mask_sl = mask_data[:, si, :]
        else:
            pet_sl = pet_sum[:, :, si]
            mask_sl = mask_data[:, :, si]

        axes[i].imshow(pet_sl.T, cmap="gray", origin="lower", aspect="equal")
        mask_overlay = np.ma.masked_where(mask_sl.T == 0, mask_sl.T)
        axes[i].imshow(mask_overlay, cmap="Reds", alpha=0.35, origin="lower", aspect="equal")
        axes[i].set_title(f"{title}", fontsize=12)
        axes[i].axis("off")

    fig.suptitle(f"{subject_id}: Whole-Brain Mask (MR-derived) on PET",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved mask-on-PET QC: %s", out_path)


def run(subject_id: str, cfg: PipelineConfig, force: bool = False) -> dict:
    """
    Run step 0c: Register MR to CT space, resample brain mask to PET space.
    CT and PET share the same coordinate system (same PET/CT scanner, same session),
    so warping MR into CT space brings it into PET space without interpolating PET signal.
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("STEP s00c: MR-to-CT coregistration")
    log.info("Subject: %s", subject_id)

    derived = cfg.derived_dir(subject_id)
    sub = f"sub-{subject_id}"

    # Inputs
    ct_path = derived / "CT1" / f"{sub}_ct.nii.gz"
    mr_n4_path = derived / "MR" / f"{sub}_mr_n4.nii.gz"
    mask_brain_mr = derived / "masks" / f"{sub}_space-MR_mask-brain-whole.nii.gz"
    pet_path = derived / "PET1" / f"{sub}_pet.nii.gz"

    # Outputs
    outputs = {
        "ct_cropped": derived / "CT1" / f"{sub}_ct-cropped.nii.gz",
        "mr_in_ct": derived / "MR" / f"{sub}_mr-in-ct.nii.gz",
        "transform": derived / "MR" / f"{sub}_mr_to_ct_0GenericAffine.mat",
        "mask_brain_pet": derived / "masks" / f"{sub}_space-PET_mask-brain-whole.nii.gz",
        "qc_coreg": cfg.figures_dir() / f"{sub}_coregistration-qc.png",
        "qc_mask_pet": cfg.figures_dir() / f"{sub}_mask-brain-whole-on-pet.png",
    }

    # Cache check
    output_paths = list(outputs.values())
    input_paths = [ct_path, mr_n4_path, mask_brain_mr, pet_path]
    if not force and check_outputs_current(input_paths, output_paths):
        log_skip("s00c_coregister", output_paths)
        return outputs

    for inp in input_paths:
        if not inp.exists():
            raise FileNotFoundError(f"Required input missing: {inp}")
        log.info("Input: %s", inp)

    # Load images with ANTsPy
    ct_ants = ants.image_read(str(ct_path))
    mr_ants = ants.image_read(str(mr_n4_path))
    mask_mr_ants = ants.image_read(str(mask_brain_mr))

    log.info("CT (full): shape=%s, spacing=%s", ct_ants.shape, ct_ants.spacing)
    log.info("MR: shape=%s, spacing=%s", mr_ants.shape, mr_ants.spacing)
    log.info("Brain mask (MR space): shape=%s", mask_mr_ants.shape)

    # --- Crop CT to head region ---
    # Reuse from s00b if available, otherwise crop fresh
    ct_cropped_from_s00b = derived / "CT1" / f"{sub}_ct-cropped.nii.gz"
    if ct_cropped_from_s00b.exists():
        log.info("Reusing cropped CT from s00b: %s", ct_cropped_from_s00b)
        ct_cropped = ants.image_read(str(ct_cropped_from_s00b))
    else:
        ct_cropped = _crop_ct(ct_ants, cfg)
    outputs["ct_cropped"].parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(ct_cropped, str(outputs["ct_cropped"]))
    log.info("Cropped CT: shape=%s", ct_cropped.shape)

    # --- Register MR -> cropped CT (rigid, 6 DOF) ---
    # Reuse transform from s00b if available
    transform_from_s00b = derived / "MR" / f"{sub}_mr_to_ct_0GenericAffine.mat"
    transform_path = str(outputs["transform"])

    if transform_from_s00b.exists():
        log.info("Reusing MR-to-CT transform from s00b: %s", transform_from_s00b)
        # Only copy if source and destination differ (s00b writes to same path)
        if transform_from_s00b.resolve() != outputs["transform"].resolve():
            import shutil
            shutil.copy2(str(transform_from_s00b), transform_path)
        # Warp MR to CT space using saved transform for QC
        mr_in_ct = ants.apply_transforms(
            fixed=ct_cropped,
            moving=mr_ants,
            transformlist=[transform_path],
        )
    else:
        log.info("Registering MR -> cropped CT (rigid)...")
        reg = ants.registration(
            fixed=ct_cropped,
            moving=mr_ants,
            type_of_transform="Rigid",
        )
        log.info("Registration complete")
        import shutil
        shutil.copy2(reg["fwdtransforms"][0], transform_path)
        mr_in_ct = reg["warpedmovout"]

    outputs["mr_in_ct"].parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(mr_in_ct, str(outputs["mr_in_ct"]))
    log.info("Saved MR in CT space: %s", outputs["mr_in_ct"])
    log.info("Saved transform: %s", outputs["transform"])

    # --- Apply transform to brain mask (MR -> full CT space) ---
    # Use the FULL CT as the reference (not cropped) to avoid clipping the brain
    # mask at crop boundaries. The registration used the cropped CT for quality,
    # but the transform applies in world coordinates so it works with any reference.
    log.info("Warping brain mask from MR to full CT space...")
    mask_in_ct = ants.apply_transforms(
        fixed=ct_ants,
        moving=mask_mr_ants,
        transformlist=[transform_path],
        interpolator="genericLabel",
    )
    mask_ct_data = mask_in_ct.numpy()
    mask_ct_data = (mask_ct_data > 0.5).astype(np.float32)
    mask_in_ct = ct_ants.new_image_like(mask_ct_data)

    n_vox_ct = int(np.sum(mask_ct_data > 0))
    vox_vol_ct = float(np.prod(ct_ants.spacing))
    vol_ct_ml = n_vox_ct * vox_vol_ct / 1000.0
    log.info("Brain mask in full CT space: %d voxels, %.1f mL", n_vox_ct, vol_ct_ml)

    # --- Resample mask from CT resolution to PET resolution ---
    log.info("Resampling brain mask from CT to PET resolution...")
    pet_ants = ants.image_read(str(pet_path))
    # For 4D PET, extract first frame as reference for geometry
    pet_data_4d = pet_ants.numpy()
    pet_ref = ants.from_numpy(
        pet_data_4d[:, :, :, 0],
        origin=pet_ants.origin[:3],
        spacing=pet_ants.spacing[:3],
        direction=pet_ants.direction[:3, :3],
    )

    # Resample mask to PET grid using nearest-neighbor (preserves binary mask)
    mask_in_pet = ants.resample_image_to_target(
        mask_in_ct, pet_ref, interp_type="nearestNeighbor",
    )

    mask_pet_data = mask_in_pet.numpy()
    mask_pet_data = (mask_pet_data > 0.5).astype(np.float32)
    mask_in_pet = pet_ref.new_image_like(mask_pet_data)

    n_vox_pet = int(np.sum(mask_pet_data > 0))
    vox_vol_pet = float(np.prod(pet_ants.spacing[:3]))
    vol_pet_ml = n_vox_pet * vox_vol_pet / 1000.0
    log.info("Brain mask in PET space: %d voxels, %.1f mL", n_vox_pet, vol_pet_ml)

    # Sanity check volume
    vol_min, vol_max = cfg.MR_BRAIN_VOLUME_RANGE_ML
    if vol_pet_ml < vol_min or vol_pet_ml > vol_max:
        log.warning(
            "Brain mask volume in PET space (%.1f mL) outside expected range (%.0f-%.0f mL)",
            vol_pet_ml, vol_min, vol_max,
        )

    # Save mask in PET space
    outputs["mask_brain_pet"].parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(mask_in_pet, str(outputs["mask_brain_pet"]))
    log.info("Saved whole-brain mask (PET space): %s", outputs["mask_brain_pet"])

    # --- QC figures ---
    _qc_registration(ct_cropped, mr_in_ct, outputs["qc_coreg"], subject_id, cfg)
    _qc_mask_on_pet(pet_path, outputs["mask_brain_pet"], outputs["qc_mask_pet"],
                     subject_id, cfg)

    elapsed = time.time() - t0
    log.info("DONE s00c: %.1f s elapsed", elapsed)
    return outputs


if __name__ == "__main__":
    import argparse
    from pipeline.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Step 0c: MR-to-CT coregistration")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(base_dir=Path(args.base_dir))
    setup_logging(args.subject, cfg.logs_dir(), verbose=args.verbose)
    run(args.subject, cfg, force=args.force)
