#!/usr/bin/env python3
"""Fill the largest enclosed area on each z-slice of a hand-drawn outline mask.

The input mask contains only the outline (shell) of the brain on each slice.
This script fills the largest enclosed interior region per slice, leaving
smaller enclosed areas (CSF, ventricles) unfilled.

Flagged slices:
  - NO_HOLE: slice has mask voxels but no enclosed area (already solid)
  - GAP: no mask voxels at all (unexpected gap in coverage)
  - SIZE_JUMP: largest filled area changed by >50% from previous slice
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def fill_outline_mask(mask_path: Path, out_path: Path):
    """Fill largest enclosed area per z-slice, flag anomalies."""

    print(f"Loading mask: {mask_path}")
    img = nib.load(mask_path)
    data = img.get_fdata() > 0
    shape = data.shape
    vox_vol_mm3 = float(np.abs(np.linalg.det(img.affine[:3, :3])))
    print(f"  Shape: {shape}, voxel: {vox_vol_mm3:.4f} mm^3")

    # Find z-range with any mask voxels
    z_has_mask = np.any(data, axis=(0, 1))
    z_indices = np.where(z_has_mask)[0]
    z_min, z_max = int(z_indices[0]), int(z_indices[-1])
    print(f"  Mask spans z={z_min} to z={z_max} ({z_max - z_min + 1} slices)")

    # Output: start with a copy of the original mask (outline preserved)
    out = data.copy()

    flags = []
    prev_fill_size = None

    # Process from superior (high z) to inferior (low z)
    for z in range(z_max, z_min - 1, -1):
        sl = data[:, :, z]
        n_mask = int(np.sum(sl))

        if n_mask == 0:
            flags.append((z, "GAP", 0))
            prev_fill_size = None
            continue

        # Fill all holes in this 2D slice
        filled = ndimage.binary_fill_holes(sl)
        holes = filled & ~sl  # interior regions only

        n_hole_voxels = int(np.sum(holes))
        if n_hole_voxels == 0:
            # No enclosed area: slice is already solid (expected at extremes)
            flags.append((z, "NO_HOLE", n_mask))
            prev_fill_size = None
            continue

        # Label connected components of holes
        labeled, n_labels = ndimage.label(holes)

        if n_labels == 1:
            # Only one enclosed area: fill it
            largest_mask = holes
            largest_size = n_hole_voxels
        else:
            # Multiple enclosed areas: keep only the largest
            sizes = ndimage.sum(holes, labeled, range(1, n_labels + 1))
            largest_label = int(np.argmax(sizes)) + 1
            largest_size = int(sizes[largest_label - 1])
            largest_mask = labeled == largest_label

        # Check for size jump from previous slice
        if prev_fill_size is not None and prev_fill_size > 0:
            ratio = largest_size / prev_fill_size
            if ratio > 1.5 or ratio < 0.5:
                flags.append((z, "SIZE_JUMP", largest_size,
                              f"prev={prev_fill_size}, ratio={ratio:.2f}"))

        # Fill the largest enclosed area into the output
        out[:, :, z] |= largest_mask
        prev_fill_size = largest_size

    # Save output
    out_data = out.astype(np.float32)
    out_img = nib.Nifti1Image(out_data, img.affine, img.header)
    out_img.header.set_data_dtype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, out_path)

    # Statistics
    n_orig = int(np.sum(data))
    n_filled = int(np.sum(out))
    vol_orig = n_orig * vox_vol_mm3 / 1000.0
    vol_filled = n_filled * vox_vol_mm3 / 1000.0
    print(f"\nResults:")
    print(f"  Original:  {n_orig:>10d} voxels = {vol_orig:.2f} mL")
    print(f"  Filled:    {n_filled:>10d} voxels = {vol_filled:.2f} mL")
    print(f"  Added:     {n_filled - n_orig:>10d} voxels = {vol_filled - vol_orig:.2f} mL")
    print(f"  Saved: {out_path}")

    # Report flags
    if flags:
        print(f"\nFlagged slices ({len(flags)}):")
        for entry in flags:
            z = entry[0]
            flag_type = entry[1]
            if flag_type == "GAP":
                print(f"  z={z:3d}: GAP (no mask voxels)")
            elif flag_type == "NO_HOLE":
                print(f"  z={z:3d}: NO_HOLE (already solid, {entry[2]} voxels)")
            elif flag_type == "SIZE_JUMP":
                print(f"  z={z:3d}: SIZE_JUMP (fill={entry[2]} voxels, {entry[3]})")
    else:
        print("\nNo flagged slices.")

    return out_path


def main():
    base = Path(__file__).resolve().parent.parent.parent
    subject = "SUB001_20260225"
    mask_dir = base / "DerivedData" / subject / "masks"

    mask_path = mask_dir / f"sub-{subject}_mr-in-ct_WB_mask.nii.gz"
    out_path = mask_dir / f"sub-{subject}_mr-in-ct_WB_mask_filled.nii.gz"

    fill_outline_mask(mask_path, out_path)


if __name__ == "__main__":
    main()
