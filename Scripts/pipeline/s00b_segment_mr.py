"""Step 0b: MR brain segmentation using CT-guided intracranial cavity extraction.

Algorithm (CT-guided):
1. N4 bias field correction of MR
2. Crop CT to head region (using config crop bounds)
3. Rigid registration: MR -> cropped CT
4. Resample CT to 1mm isotropic for morphological operations
5. Threshold CT at bone_thresh HU for bone mask
6. Two-stage morphological closing to seal skull gaps
7. Inferior cap to seal foramen magnum
8. Fill holes to identify intracranial cavity (ICC)
9. Largest connected component, cleanup
10. Transform ICC back to MR space via inverse rigid transform
11. QC: volume check + overlay figure

The CT bone boundary provides an unambiguous skull delineation that MR-only
methods (intensity thresholding, atlas registration, SynthStrip) cannot match
for challenging anatomies.
"""
import logging
import shutil
import time
from pathlib import Path

import ants
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from pipeline.cache import check_outputs_current, log_skip
from pipeline.config import PipelineConfig

log = logging.getLogger("s00b_segment_mr")


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    labeled, n_labels = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 3))
    if n_labels == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n_labels + 1))
    largest = np.argmax(sizes) + 1
    return (labeled == largest).astype(np.uint8)


def _crop_ct(ct_ants, cfg: PipelineConfig):
    """Crop CT to head region using configured voxel indices.

    Returns a new ANTs image with updated origin to preserve world coordinates.
    """
    lr_start, lr_end = cfg.CT_CROP_LR
    ap_start, ap_end = cfg.CT_CROP_AP
    z_start, z_end = cfg.CT_CROP_Z

    ct_data = ct_ants.numpy()
    log.info("CT before crop: shape=%s", ct_data.shape)

    cropped = ct_data[lr_start:lr_end, ap_start:ap_end, z_start:z_end]
    log.info("CT after crop: shape=%s", cropped.shape)

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
    return ct_cropped


def _extract_icc_from_ct(ct_1mm, cfg: PipelineConfig) -> np.ndarray:
    """Extract the intracranial cavity from a 1mm-isotropic CT image.

    Uses bone thresholding, morphological closing to seal the skull,
    an inferior cap to block the foramen magnum, and hole-filling.

    Returns a binary mask (uint8) in the same grid as ct_1mm.
    """
    ct_data = ct_1mm.numpy()
    struct6 = ndimage.generate_binary_structure(3, 1)

    # Threshold for bone
    bone_thresh = cfg.CT_BONE_THRESHOLD_HU
    log.info("Bone threshold: %d HU", bone_thresh)
    bone = (ct_data > bone_thresh).astype(np.uint8)
    log.info("Bone voxels: %d (%.1f%% of volume)",
             np.sum(bone), 100.0 * np.sum(bone) / bone.size)

    # Two-stage morphological closing to seal skull gaps
    # Stage 1: seal small gaps (sutures, thin bone)
    # Stage 2: seal larger gaps (orbital fissures, etc.)
    # Two-stage is critical: single-stage 15-iter closing fills the entire cavity
    bone_closed = ndimage.binary_closing(bone, structure=struct6,
                                          iterations=cfg.CT_CLOSE_ITER_1).astype(np.uint8)
    log.info("After closing stage 1 (%d iter): %d voxels",
             cfg.CT_CLOSE_ITER_1, np.sum(bone_closed))
    bone_closed = ndimage.binary_closing(bone_closed, structure=struct6,
                                          iterations=cfg.CT_CLOSE_ITER_2).astype(np.uint8)
    log.info("After closing stage 2 (%d+%d iter): %d voxels",
             cfg.CT_CLOSE_ITER_1, cfg.CT_CLOSE_ITER_2, np.sum(bone_closed))

    # Find S-I axis and add inferior cap
    direction = np.array(ct_1mm.direction)
    si_axis = np.argmax(np.abs(direction[:, 2]))
    si_sign = np.sign(direction[si_axis, 2])
    log.info("S-I axis: dim %d, sign=%.0f", si_axis, si_sign)

    bone_coords = np.argwhere(bone_closed > 0)
    bone_center = bone_coords.mean(axis=0)

    inferior_cutoff_vox = int(bone_center[si_axis] - si_sign * cfg.CT_INFERIOR_CAP_MM)
    inferior_cutoff_vox = np.clip(inferior_cutoff_vox, 0, ct_data.shape[si_axis] - 1)
    log.info("Inferior cap: slice %d along axis %d (centroid=%.0f, offset=%dmm)",
             inferior_cutoff_vox, si_axis, bone_center[si_axis], cfg.CT_INFERIOR_CAP_MM)

    if si_sign > 0:
        slc = [slice(None)] * 3
        slc[si_axis] = slice(0, inferior_cutoff_vox)
        bone_closed[tuple(slc)] = 1
    else:
        slc = [slice(None)] * 3
        slc[si_axis] = slice(inferior_cutoff_vox, None)
        bone_closed[tuple(slc)] = 1

    # Fill holes to identify intracranial cavity
    log.info("Filling holes to identify intracranial cavity...")
    filled = ndimage.binary_fill_holes(bone_closed).astype(np.uint8)
    icc = ((filled > 0) & (bone_closed == 0)).astype(np.uint8)
    log.info("ICC (raw): %d voxels, %.1f mL", np.sum(icc), np.sum(icc) * 1.0 / 1000.0)

    # Remove inferior cap region from ICC
    margin_vox = 5
    if si_sign > 0:
        slc_remove = [slice(None)] * 3
        slc_remove[si_axis] = slice(0, inferior_cutoff_vox + margin_vox)
        icc[tuple(slc_remove)] = 0
    else:
        slc_remove = [slice(None)] * 3
        slc_remove[si_axis] = slice(max(0, inferior_cutoff_vox - margin_vox), None)
        icc[tuple(slc_remove)] = 0

    log.info("ICC (after inferior clip): %d voxels, %.1f mL",
             np.sum(icc), np.sum(icc) * 1.0 / 1000.0)

    # Largest connected component
    icc = _largest_component(icc)
    vol_ml = np.sum(icc) * 1.0 / 1000.0
    log.info("ICC (largest CC): %d voxels, %.1f mL", np.sum(icc), vol_ml)

    return icc


def _qc_figure(mr_data: np.ndarray, mask_data: np.ndarray,
               out_path: Path, title: str, cfg: PipelineConfig) -> None:
    """Create QC figure: MR slices with brain mask overlay."""
    plt.style.use(cfg.FIGURE_STYLE)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    com = ndimage.center_of_mass(mask_data)
    slices = [int(com[0]), int(com[1]), int(com[2])]

    views = [
        ("Sagittal", mr_data[slices[0], :, :], mask_data[slices[0], :, :]),
        ("Coronal", mr_data[:, slices[1], :], mask_data[:, slices[1], :]),
        ("Axial", mr_data[:, :, slices[2]], mask_data[:, :, slices[2]]),
    ]

    for i, (vname, mr_slice, mask_slice) in enumerate(views):
        axes[0, i].imshow(mr_slice.T, cmap="gray", origin="lower", aspect="equal")
        axes[0, i].set_title(f"{vname} (MR)", fontsize=12)
        axes[0, i].axis("off")

        axes[1, i].imshow(mr_slice.T, cmap="gray", origin="lower", aspect="equal")
        mask_overlay = np.ma.masked_where(mask_slice.T == 0, mask_slice.T)
        axes[1, i].imshow(mask_overlay, cmap="Reds", alpha=0.4, origin="lower", aspect="equal")
        axes[1, i].set_title(f"{vname} (MR + brain mask)", fontsize=12)
        axes[1, i].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved QC figure: %s", out_path)


def run(subject_id: str, cfg: PipelineConfig, force: bool = False) -> dict:
    """
    Run step 0b: N4 bias correction + CT-guided brain extraction.

    Uses CT bone to define the intracranial cavity, then transforms to MR space.
    Also performs the MR-to-CT rigid registration (saved for reuse by s00c).
    """
    t0 = time.time()
    log.info("=" * 60)
    log.info("STEP s00b: MR brain segmentation (CT-guided)")
    log.info("Subject: %s", subject_id)

    derived = cfg.derived_dir(subject_id)
    sub = f"sub-{subject_id}"

    # Inputs
    mr_path = derived / "MR" / f"{sub}_mr.nii.gz"
    ct_path = derived / "CT1" / f"{sub}_ct.nii.gz"

    # Outputs
    outputs = {
        "mr_n4": derived / "MR" / f"{sub}_mr_n4.nii.gz",
        "mask_brain_mr": derived / "masks" / f"{sub}_space-MR_mask-brain-whole.nii.gz",
        "ct_cropped": derived / "CT1" / f"{sub}_ct-cropped.nii.gz",
        "transform": derived / "MR" / f"{sub}_mr_to_ct_0GenericAffine.mat",
        "qc_figure": cfg.figures_dir() / f"{sub}_mr-brain-mask-qc.png",
    }

    # Cache check
    output_paths = list(outputs.values())
    input_paths = [mr_path, ct_path]
    if not force and check_outputs_current(input_paths, output_paths):
        log_skip("s00b_segment_mr", output_paths)
        return outputs

    for inp in input_paths:
        if not inp.exists():
            raise FileNotFoundError(f"Required input missing: {inp}")
        log.info("Input: %s", inp)

    # ---- N4 bias field correction ----
    mr_ants = ants.image_read(str(mr_path))
    log.info("MR loaded: shape=%s, spacing=%s", mr_ants.shape, mr_ants.spacing)

    log.info("Running N4 bias field correction...")
    mr_n4 = ants.n4_bias_field_correction(mr_ants)
    log.info("N4 correction complete")

    outputs["mr_n4"].parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(mr_n4, str(outputs["mr_n4"]))
    log.info("Saved N4-corrected MR: %s", outputs["mr_n4"])

    # ---- Crop CT to head region ----
    ct_ants = ants.image_read(str(ct_path))
    log.info("CT loaded: shape=%s, spacing=%s", ct_ants.shape, ct_ants.spacing)

    ct_cropped = _crop_ct(ct_ants, cfg)
    outputs["ct_cropped"].parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(ct_cropped, str(outputs["ct_cropped"]))
    log.info("Saved cropped CT: %s", outputs["ct_cropped"])

    # ---- Rigid registration: MR -> cropped CT ----
    log.info("Registering MR -> cropped CT (rigid)...")
    reg = ants.registration(
        fixed=ct_cropped,
        moving=mr_n4,
        type_of_transform="Rigid",
    )
    log.info("Registration complete. Transforms: %s", reg["fwdtransforms"])

    # Save the transform for reuse by s00c
    shutil.copy2(reg["fwdtransforms"][0], str(outputs["transform"]))
    log.info("Saved MR-to-CT transform: %s", outputs["transform"])

    # ---- CT-guided ICC extraction ----
    log.info("Resampling CT to 1mm isotropic for morphological operations...")
    ct_1mm = ants.resample_image(ct_cropped, (1.0, 1.0, 1.0),
                                  use_voxels=False, interp_type=0)
    log.info("CT at 1mm: shape=%s", ct_1mm.shape)

    icc_1mm = _extract_icc_from_ct(ct_1mm, cfg)

    # ---- Transform ICC to MR space ----
    log.info("Resampling ICC back to CT native resolution...")
    icc_1mm_ants = ct_1mm.new_image_like(icc_1mm.astype(np.float32))
    icc_native = ants.resample_image_to_target(icc_1mm_ants, ct_cropped,
                                                interp_type="nearestNeighbor")
    icc_native_data = (icc_native.numpy() > 0.5).astype(np.float32)
    icc_native = ct_cropped.new_image_like(icc_native_data)

    log.info("Transforming ICC from CT to MR space (inverse rigid)...")
    icc_in_mr = ants.apply_transforms(
        fixed=mr_n4,
        moving=icc_native,
        transformlist=[str(outputs["transform"])],
        whichtoinvert=[True],
        interpolator="genericLabel",
    )

    mask_data = (icc_in_mr.numpy() > 0.5).astype(np.uint8)
    mask_data = ndimage.binary_fill_holes(mask_data).astype(np.uint8)
    mask_data = _largest_component(mask_data)

    # ---- Volume check ----
    vox_vol = float(np.prod(mr_n4.spacing))
    n_vox = int(np.sum(mask_data > 0))
    vol_ml = n_vox * vox_vol / 1000.0
    log.info("Brain mask (MR space): %d voxels, %.1f mL", n_vox, vol_ml)

    vol_min, vol_max = cfg.MR_BRAIN_VOLUME_RANGE_ML
    if vol_ml < vol_min or vol_ml > vol_max:
        log.warning(
            "Brain mask volume %.1f mL outside expected range (%.0f-%.0f mL). "
            "Manual QC recommended.",
            vol_ml, vol_min, vol_max,
        )

    # ---- Save mask ----
    mask_ants = mr_n4.new_image_like(mask_data.astype(np.float32))
    outputs["mask_brain_mr"].parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(mask_ants, str(outputs["mask_brain_mr"]))
    log.info("Saved brain mask (MR space): %s", outputs["mask_brain_mr"])

    # ---- QC figure ----
    mr_data = mr_n4.numpy()
    _qc_figure(mr_data, mask_data, outputs["qc_figure"],
               f"{subject_id}: CT-Guided Brain Extraction (vol={vol_ml:.1f} mL)", cfg)

    elapsed = time.time() - t0
    log.info("DONE s00b: %.1f s elapsed", elapsed)
    return outputs


if __name__ == "__main__":
    import argparse
    from pipeline.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Step 0b: MR brain segmentation (CT-guided)")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfg = PipelineConfig(base_dir=Path(args.base_dir))
    setup_logging(args.subject, cfg.logs_dir(), verbose=args.verbose)
    run(args.subject, cfg, force=args.force)
