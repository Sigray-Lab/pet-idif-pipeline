#!/usr/bin/env python3
"""Process manually drawn whole-brain mask: resample, erode, extract TAC, plot.

Steps:
1. Resample filled mask to PET space (overwrites pipeline whole-brain mask)
2. Create 2D per-slice eroded version (6 voxels) at CT resolution, resample to PET
3. Extract brain TAC from eroded mask
4. Plot brain TAC alone and combined with IDIF (SUV, normal + log y-axis)
"""

from pathlib import Path

import ants
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


FIGURE_DPI = 300
FIGURE_STYLE = "seaborn-v0_8-whitegrid"
EROSION_VOXELS = 6


def resample_mask_to_pet(mask_ants, pet_path: Path) -> tuple:
    """Resample a CT-resolution mask to PET space. Returns (ANTs image, n_vox, vol_mL)."""
    pet_ants = ants.image_read(str(pet_path))
    pet_data_4d = pet_ants.numpy()
    pet_ref = ants.from_numpy(
        pet_data_4d[:, :, :, 0],
        origin=pet_ants.origin[:3],
        spacing=pet_ants.spacing[:3],
        direction=pet_ants.direction[:3, :3],
    )

    mask_pet = ants.resample_image_to_target(
        mask_ants, pet_ref, interp_type="nearestNeighbor",
    )
    mask_pet_data = (mask_pet.numpy() > 0.5).astype(np.float32)
    mask_pet = pet_ref.new_image_like(mask_pet_data)

    n_vox = int(np.sum(mask_pet_data > 0))
    vox_vol = float(np.prod(pet_ants.spacing[:3]))
    vol_ml = n_vox * vox_vol / 1000.0
    return mask_pet, n_vox, vol_ml


def erode_mask_2d(data: np.ndarray, iterations: int) -> np.ndarray:
    """Erode a 3D binary mask slice-by-slice in 2D."""
    out = np.zeros_like(data, dtype=bool)
    struct = ndimage.generate_binary_structure(2, 1)  # 4-connected cross
    z_has = np.any(data, axis=(0, 1))
    for z in np.where(z_has)[0]:
        sl = data[:, :, z].astype(bool)
        eroded = ndimage.binary_erosion(sl, structure=struct, iterations=iterations)
        out[:, :, z] = eroded
    return out


def main():
    subject_id = "SUB001_20260225"
    base = Path(__file__).resolve().parent.parent.parent
    sub = f"sub-{subject_id}"
    mask_dir = base / "DerivedData" / subject_id / "masks"
    pet_path = base / "DerivedData" / subject_id / "PET1" / f"{sub}_pet.nii.gz"
    frames_path = base / "DerivedData" / subject_id / "PET1" / f"{sub}_frames.tsv"
    idif_path = base / "Outputs" / f"{sub}_idif.tsv"
    radiochem_path = base / "raw" / "Radiochem.csv"
    out_dir = base / "Outputs"
    qc_dir = base / "QC"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Step 1: Resample filled mask to PET space
    # ----------------------------------------------------------------
    filled_path = mask_dir / f"{sub}_mr-in-ct_WB_mask_filled.nii.gz"
    print(f"=== Step 1: Resample filled mask to PET space ===")
    print(f"Loading: {filled_path}")
    filled_ants = ants.image_read(str(filled_path))

    filled_pet, n_filled, vol_filled = resample_mask_to_pet(filled_ants, pet_path)

    # Save as pipeline whole-brain mask (overwrites s00c output)
    wb_pet_path = mask_dir / f"{sub}_space-PET_mask-brain-whole.nii.gz"
    ants.image_write(filled_pet, str(wb_pet_path))
    print(f"  Filled mask in PET space: {n_filled} voxels, {vol_filled:.1f} mL")
    print(f"  Saved: {wb_pet_path}")

    # ----------------------------------------------------------------
    # Step 2: 2D per-slice erosion at CT resolution, then resample
    # ----------------------------------------------------------------
    print(f"\n=== Step 2: Erode filled mask ({EROSION_VOXELS} voxels per slice) ===")
    filled_data = filled_ants.numpy() > 0.5
    eroded_data = erode_mask_2d(filled_data, EROSION_VOXELS)

    n_eroded_ct = int(np.sum(eroded_data))
    vox_vol_ct = float(np.abs(np.linalg.det(
        np.array(filled_ants.direction).reshape(3, 3) * np.array(filled_ants.spacing)
    )))
    vol_eroded_ct = n_eroded_ct * vox_vol_ct / 1000.0
    print(f"  Eroded mask (CT res): {n_eroded_ct} voxels, {vol_eroded_ct:.1f} mL")

    # Save eroded at CT resolution
    eroded_ct_ants = filled_ants.new_image_like(eroded_data.astype(np.float32))
    eroded_ct_path = mask_dir / f"{sub}_mr-in-ct_WB_mask_filled_eroded{EROSION_VOXELS}.nii.gz"
    ants.image_write(eroded_ct_ants, str(eroded_ct_path))
    print(f"  Saved (CT res): {eroded_ct_path}")

    # Resample eroded mask to PET space
    eroded_pet, n_eroded_pet, vol_eroded_pet = resample_mask_to_pet(eroded_ct_ants, pet_path)
    eroded_pet_path = mask_dir / f"{sub}_space-PET_mask-brain-eroded{EROSION_VOXELS}.nii.gz"
    ants.image_write(eroded_pet, str(eroded_pet_path))
    print(f"  Eroded mask (PET res): {n_eroded_pet} voxels, {vol_eroded_pet:.1f} mL")
    print(f"  Saved (PET res): {eroded_pet_path}")

    # ----------------------------------------------------------------
    # Step 3: Extract brain TAC from eroded mask
    # ----------------------------------------------------------------
    print(f"\n=== Step 3: Extract brain TAC from eroded mask ===")
    pet_img = nib.load(pet_path)
    pet_data = pet_img.get_fdata()
    n_frames = pet_data.shape[3]
    print(f"  PET: {pet_data.shape}")

    eroded_nib = nib.load(eroded_pet_path)
    mask_data = eroded_nib.get_fdata() > 0.5
    n_mask_vox = int(np.sum(mask_data))
    print(f"  Eroded mask (PET): {n_mask_vox} voxels")

    frames_df = pd.read_csv(frames_path, sep="\t", comment="#")

    means, stds, mins, maxs = [], [], [], []
    for f_idx in range(n_frames):
        vals = pet_data[:, :, :, f_idx][mask_data]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        mins.append(float(np.min(vals)))
        maxs.append(float(np.max(vals)))

    tac_df = pd.DataFrame({
        "frame": frames_df["frame_index"].values,
        "start_s": frames_df["start_s"].values,
        "end_s": frames_df["end_s"].values,
        "mid_time_s": frames_df["mid_time_s"].values,
        "mid_time_min": frames_df["mid_time_min"].values,
        "mean_activity_Bq_per_mL": means,
        "std_activity": stds,
        "min_activity": mins,
        "max_activity": maxs,
        "n_voxels": [n_mask_vox] * n_frames,
    })

    tac_path = out_dir / f"{sub}_tac-raw-eroded.tsv"
    with open(tac_path, "w") as f:
        f.write(f"# subject: {subject_id}\n")
        f.write(f"# mask: {eroded_pet_path.name}\n")
        f.write(f"# roi_volume_mL: {vol_eroded_pet:.2f}\n")
        f.write(f"# n_voxels: {n_mask_vox}\n")
        f.write(f"# erosion_voxels: {EROSION_VOXELS}\n")
        f.write(f"# whole_brain_volume_mL: {vol_filled:.2f}\n")
        tac_df.to_csv(f, sep="\t", index=False, float_format="%.4f")
    print(f"  Saved TAC: {tac_path}")

    # ----------------------------------------------------------------
    # Step 4: Compute SUV and create plots
    # ----------------------------------------------------------------
    print(f"\n=== Step 4: Plots ===")

    # Load radiochem
    rc_df = pd.read_csv(radiochem_path)
    rc_row = rc_df[rc_df["id"] == subject_id].iloc[0]
    dose_bq = float(rc_row["injected_MBq"]) * 1e6
    weight_g = float(rc_row["weight_kg"]) * 1000.0
    suv_factor = dose_bq / weight_g
    print(f"  SUV factor: {suv_factor:.2f}")

    # Load IDIF TAC
    idif_df = pd.read_csv(idif_path, sep="\t", comment="#")

    # Compute SUVs
    brain_suv = np.array(means) / suv_factor
    idif_suv = idif_df["mean_activity_Bq_per_mL"].values / suv_factor
    mid_time = tac_df["mid_time_min"].values
    idif_time = idif_df["mid_time_min"].values

    plt.style.use(FIGURE_STYLE)

    # Plot 1: Brain TAC alone (Bq/mL)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mid_time, means, "o-", color="steelblue", markersize=4, linewidth=1.5)
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Mean Activity (Bq/mL)", fontsize=12)
    ax.set_title(
        f"{subject_id}: Brain TAC (eroded mask, {vol_eroded_pet:.0f} mL)",
        fontsize=14,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    p1 = qc_dir / f"{sub}_tac-raw-eroded.png"
    fig.savefig(p1, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p1}")

    # Plot 2: Brain + IDIF, SUV, linear y
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mid_time, brain_suv,
            "o-", color="steelblue", markersize=3, linewidth=1.5, label="Brain (eroded)")
    ax.plot(idif_time, idif_suv,
            "s-", color="crimson", markersize=3, linewidth=1.5, label="IDIF")
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("SUV", fontsize=12)
    ax.set_title(f"{subject_id}: Brain TAC + IDIF (SUV)", fontsize=14)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p2 = qc_dir / f"{sub}_tac-combined-suv-eroded.png"
    fig.savefig(p2, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p2}")

    # Plot 3: Brain + IDIF, SUV, log y
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mid_time, brain_suv,
            "o-", color="steelblue", markersize=3, linewidth=1.5, label="Brain (eroded)")
    ax.plot(idif_time, idif_suv,
            "s-", color="crimson", markersize=3, linewidth=1.5, label="IDIF")
    ax.set_yscale("log")
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("SUV (log scale)", fontsize=12)
    ax.set_title(f"{subject_id}: Brain TAC + IDIF (SUV, log scale)", fontsize=14)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p3 = qc_dir / f"{sub}_tac-combined-suv-log-eroded.png"
    fig.savefig(p3, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p3}")

    # Summary
    peak_idx = int(np.argmax(means))
    plateau_mask = (mid_time >= 30) & (mid_time <= 60)
    plateau_suv = float(np.mean(brain_suv[plateau_mask]))
    print(f"\n=== Summary ===")
    print(f"  Whole-brain volume (filled mask): {vol_filled:.1f} mL")
    print(f"  Eroded mask volume: {vol_eroded_pet:.1f} mL")
    print(f"  Peak brain activity: {means[peak_idx]:.1f} Bq/mL at {mid_time[peak_idx]:.1f} min")
    print(f"  Plateau SUV (30-60 min): {plateau_suv:.4f}")
    print(f"  Plateau brain activity (30-60 min): {float(np.mean(np.array(means)[plateau_mask])):.1f} Bq/mL")


if __name__ == "__main__":
    main()
