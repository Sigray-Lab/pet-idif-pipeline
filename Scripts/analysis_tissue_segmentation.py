#!/usr/bin/env python3
"""Tissue segmentation (CSF/GM/WM) via 3-class GMM on T1w MR intensities.

Segments the eroded brain mask into three tissue classes using a Gaussian
Mixture Model on N4-corrected T1w MR intensities, resamples tissue masks
to PET space, extracts per-tissue TACs, and generates QC figures.

On T1w MRI: WM = brightest (~700-800), GM = intermediate (~500), CSF = darkest (~250-300).
"""

from pathlib import Path

import ants
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P
from scipy import ndimage
from sklearn.mixture import GaussianMixture


FIGURE_DPI = 300
FIGURE_STYLE = "seaborn-v0_8-whitegrid"
N_COMPONENTS = 3
RANDOM_STATE = 42

# Tissue label encoding in segmentation volume
LABEL_BG = 0
LABEL_CSF = 1
LABEL_GM = 2
LABEL_WM = 3
TISSUE_NAMES = {LABEL_CSF: "CSF", LABEL_GM: "GM", LABEL_WM: "WM"}
TISSUE_COLORS = {LABEL_CSF: "royalblue", LABEL_GM: "forestgreen", LABEL_WM: "firebrick"}


def load_tac(path: Path) -> pd.DataFrame:
    """Load a TAC TSV, skipping provenance comment lines."""
    return pd.read_csv(path, sep="\t", comment="#")


def resample_seg_to_pet(seg_ants, pet_path: Path):
    """Resample integer segmentation volume from CT to PET space (nearest-neighbor)."""
    pet_ants = ants.image_read(str(pet_path))
    pet_data_4d = pet_ants.numpy()
    pet_ref = ants.from_numpy(
        pet_data_4d[:, :, :, 0],
        origin=pet_ants.origin[:3],
        spacing=pet_ants.spacing[:3],
        direction=pet_ants.direction[:3, :3],
    )
    seg_pet = ants.resample_image_to_target(
        seg_ants, pet_ref, interp_type="nearestNeighbor",
    )
    return seg_pet, pet_ants


def extract_tissue_tac(pet_data, mask_data, frames_df):
    """Extract mean TAC from a binary mask across all PET frames."""
    n_frames = pet_data.shape[3]
    n_vox = int(np.sum(mask_data))
    means, stds = [], []
    for f_idx in range(n_frames):
        vals = pet_data[:, :, :, f_idx][mask_data]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return np.array(means), np.array(stds), n_vox


def main():
    subject_id = "SUB001_20260225"
    base = Path(__file__).resolve().parent.parent
    sub = f"sub-{subject_id}"
    mask_dir = base / "DerivedData" / subject_id / "masks"
    mr_path = base / "DerivedData" / subject_id / "MR" / f"{sub}_mr-in-ct.nii.gz"
    eroded_path = mask_dir / f"{sub}_mr-in-ct_WB_mask_filled_eroded6.nii.gz"
    pet_path = base / "DerivedData" / subject_id / "PET1" / f"{sub}_pet.nii.gz"
    frames_path = base / "DerivedData" / subject_id / "PET1" / f"{sub}_frames.tsv"
    idif_path = base / "Outputs" / f"{sub}_idif.tsv"
    radiochem_path = base / "raw" / "Radiochem.csv"
    out_dir = base / "Outputs"
    qc_dir = base / "QC"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Step 1: Load MR and eroded brain mask (CT space, 0.32mm)
    # ==================================================================
    print("=== Step 1: Load MR and eroded brain mask ===")
    mr_img = nib.load(mr_path)
    mr_data = mr_img.get_fdata()
    mask_img = nib.load(eroded_path)
    mask_data = mask_img.get_fdata() > 0.5

    # Verify same grid
    assert mr_data.shape == mask_data.shape, (
        f"Shape mismatch: MR {mr_data.shape} vs mask {mask_data.shape}"
    )
    assert np.allclose(mr_img.affine, mask_img.affine, atol=1e-4), "Affine mismatch"

    n_mask_vox = int(np.sum(mask_data))
    vox_vol_mm3 = float(np.abs(np.linalg.det(mr_img.affine[:3, :3])))
    mask_vol_ml = n_mask_vox * vox_vol_mm3 / 1000.0
    print(f"  MR shape: {mr_data.shape}, voxel size: {mr_img.header.get_zooms()[:3]}")
    print(f"  Eroded mask: {n_mask_vox} voxels, {mask_vol_ml:.1f} mL")

    # Extract MR intensities within mask
    intensities = mr_data[mask_data].astype(np.float64)
    print(f"  MR intensity stats within mask:")
    print(f"    min={np.min(intensities):.1f}  max={np.max(intensities):.1f}")
    print(f"    mean={np.mean(intensities):.1f}  median={np.median(intensities):.1f}")
    print(f"    std={np.std(intensities):.1f}")

    # ==================================================================
    # Step 2: Fit 3-component GMM
    # ==================================================================
    print("\n=== Step 2: Fit 3-component GMM ===")
    gmm = GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=5,
        max_iter=200,
    )
    intensities_2d = intensities.reshape(-1, 1)
    gmm.fit(intensities_2d)

    # Sort components by mean intensity (ascending: CSF, GM, WM)
    order = np.argsort(gmm.means_.ravel())
    labels_raw = gmm.predict(intensities_2d)

    # Remap: sorted_index -> tissue label (1=CSF, 2=GM, 3=WM)
    label_map = np.zeros(N_COMPONENTS, dtype=np.uint8)
    for tissue_idx, comp_idx in enumerate(order):
        label_map[comp_idx] = tissue_idx + 1  # 1=CSF, 2=GM, 3=WM

    tissue_labels = label_map[labels_raw]

    print(f"  BIC: {gmm.bic(intensities_2d):.1f}")
    print(f"  AIC: {gmm.aic(intensities_2d):.1f}")
    print(f"  Converged: {gmm.converged_}, N iterations: {gmm.n_iter_}")

    # Store GMM params sorted by tissue class
    gmm_params = {}
    for tissue_idx, comp_idx in enumerate(order):
        tissue_label = tissue_idx + 1
        name = TISSUE_NAMES[tissue_label]
        mu = float(gmm.means_[comp_idx, 0])
        sigma = float(np.sqrt(gmm.covariances_[comp_idx, 0, 0]))
        weight = float(gmm.weights_[comp_idx])
        n_vox = int(np.sum(tissue_labels == tissue_label))
        vol = n_vox * vox_vol_mm3 / 1000.0
        gmm_params[tissue_label] = {
            "name": name, "mean": mu, "std": sigma, "weight": weight,
            "n_voxels_ct": n_vox, "volume_ct_mL": vol,
        }
        print(f"  {name}: mean={mu:.1f}, std={sigma:.1f}, weight={weight:.3f}, "
              f"{n_vox} voxels ({vol:.1f} mL, {weight*100:.1f}%)")

    # ==================================================================
    # Step 3: Create 3D segmentation volume + morphological cleanup
    # ==================================================================
    print("\n=== Step 3: Build segmentation volume ===")
    seg_vol = np.zeros(mr_data.shape, dtype=np.uint8)
    seg_vol[mask_data] = tissue_labels

    # Light cleanup: remove isolated single voxels per class
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connected
    total_cleaned = 0
    for label_val in [LABEL_CSF, LABEL_GM, LABEL_WM]:
        class_mask = (seg_vol == label_val)
        labeled_arr, n_cc = ndimage.label(class_mask, structure=struct)
        cc_sizes = ndimage.sum(class_mask, labeled_arr, range(1, n_cc + 1))
        # Remove components of size 1
        small_labels = np.where(np.array(cc_sizes) == 1)[0] + 1
        if len(small_labels) > 0:
            remove_mask = np.isin(labeled_arr, small_labels)
            seg_vol[remove_mask] = LABEL_BG
            total_cleaned += int(np.sum(remove_mask))

    # Reassign orphaned voxels (inside brain mask but now seg=0) to nearest class
    orphans = mask_data & (seg_vol == LABEL_BG)
    n_orphans = int(np.sum(orphans))
    if n_orphans > 0:
        # Use GMM probabilities to assign
        orphan_intensities = mr_data[orphans].reshape(-1, 1)
        orphan_labels_raw = gmm.predict(orphan_intensities)
        orphan_tissue = label_map[orphan_labels_raw]
        seg_vol[orphans] = orphan_tissue
    print(f"  Cleaned {total_cleaned} isolated voxels, reassigned {n_orphans} orphans")

    # Verify all mask voxels are assigned
    assert np.all(seg_vol[mask_data] > 0), "Some mask voxels still unassigned"

    # ==================================================================
    # Step 4: Save tissue masks (CT space)
    # ==================================================================
    print("\n=== Step 4: Save tissue masks (CT space) ===")
    seg_nii = nib.Nifti1Image(seg_vol, mr_img.affine, mr_img.header)
    seg_path = mask_dir / f"{sub}_space-CT_seg-tissue.nii.gz"
    nib.save(seg_nii, seg_path)
    print(f"  Saved: {seg_path}")

    for label_val, name in TISSUE_NAMES.items():
        binary = (seg_vol == label_val).astype(np.uint8)
        nii = nib.Nifti1Image(binary, mr_img.affine, mr_img.header)
        p = mask_dir / f"{sub}_space-CT_mask-{name.lower()}.nii.gz"
        nib.save(nii, p)
        n = int(np.sum(binary))
        v = n * vox_vol_mm3 / 1000.0
        print(f"  {name}: {n} voxels, {v:.1f} mL -> {p.name}")

    # ==================================================================
    # Step 5: Resample segmentation to PET space
    # ==================================================================
    print("\n=== Step 5: Resample to PET space ===")
    seg_ants = ants.from_numpy(
        seg_vol.astype(np.float32),
        origin=list(mr_img.affine[:3, 3]),
        spacing=list(mr_img.header.get_zooms()[:3]),
        direction=mr_img.affine[:3, :3] / np.array(mr_img.header.get_zooms()[:3]),
    )

    seg_pet_ants, pet_ants = resample_seg_to_pet(seg_ants, pet_path)
    seg_pet_data = np.round(seg_pet_ants.numpy()).astype(np.uint8)

    pet_vox_vol = float(np.prod(pet_ants.spacing[:3]))
    pet_masks = {}
    for label_val, name in TISSUE_NAMES.items():
        tmask = (seg_pet_data == label_val)
        n_vox = int(np.sum(tmask))
        vol_ml = n_vox * pet_vox_vol / 1000.0
        pet_masks[label_val] = tmask
        gmm_params[label_val]["n_voxels_pet"] = n_vox
        gmm_params[label_val]["volume_pet_mL"] = vol_ml
        print(f"  {name} (PET): {n_vox} voxels, {vol_ml:.1f} mL")

        # Save PET-space mask
        mask_pet_nii = nib.Nifti1Image(
            tmask.astype(np.uint8),
            nib.load(pet_path).affine,
        )
        p = mask_dir / f"{sub}_space-PET_mask-{name.lower()}.nii.gz"
        nib.save(mask_pet_nii, p)

    total_pet = sum(gmm_params[l]["volume_pet_mL"] for l in TISSUE_NAMES)
    print(f"  Total tissue volume (PET): {total_pet:.1f} mL")

    # ==================================================================
    # Step 6: Extract per-tissue TACs
    # ==================================================================
    print("\n=== Step 6: Extract per-tissue TACs ===")
    pet_img = nib.load(pet_path)
    pet_data = pet_img.get_fdata()
    frames_df = pd.read_csv(frames_path, sep="\t", comment="#")
    n_frames = pet_data.shape[3]
    print(f"  PET: {pet_data.shape}, {n_frames} frames")

    # Load radiochem for SUV
    rc = pd.read_csv(radiochem_path)
    rc_row = rc[rc["id"] == subject_id].iloc[0]
    dose_bq = float(rc_row["injected_MBq"]) * 1e6
    weight_g = float(rc_row["weight_kg"]) * 1000.0
    suv_factor = dose_bq / weight_g
    print(f"  SUV factor: {suv_factor:.2f}")

    # Load IDIF
    idif_df = load_tac(base / "Outputs" / f"{sub}_idif.tsv")
    idif_act = idif_df["mean_activity_Bq_per_mL"].values
    idif_time = idif_df["mid_time_min"].values
    idif_suv = idif_act / suv_factor

    tissue_tacs = {}  # label -> (means, stds, n_vox)
    for label_val, name in TISSUE_NAMES.items():
        means, stds, n_vox = extract_tissue_tac(pet_data, pet_masks[label_val], frames_df)
        tissue_tacs[label_val] = (means, stds, n_vox)
        suv_vals = means / suv_factor
        # Plateau 30-60 min
        mid_t = frames_df["mid_time_min"].values
        plateau_mask = (mid_t >= 30) & (mid_t <= 60)
        plateau_suv = float(np.mean(suv_vals[plateau_mask]))
        peak_idx = int(np.argmax(means[3:]) + 3)  # skip first 3 frames (pre-bolus)
        peak_suv = float(suv_vals[peak_idx])
        peak_time = float(mid_t[peak_idx])
        gmm_params[label_val]["peak_suv"] = peak_suv
        gmm_params[label_val]["peak_time_min"] = peak_time
        gmm_params[label_val]["plateau_suv_30_60"] = plateau_suv
        print(f"  {name}: {n_vox} voxels, peak SUV={peak_suv:.4f} at {peak_time:.1f} min, "
              f"plateau SUV (30-60 min)={plateau_suv:.4f}")

    time_min = frames_df["mid_time_min"].values

    # ==================================================================
    # Step 7: Save TAC TSVs
    # ==================================================================
    print("\n=== Step 7: Save TSVs ===")
    for label_val, name in TISSUE_NAMES.items():
        means, stds, n_vox = tissue_tacs[label_val]
        p = gmm_params[label_val]
        tac_df = pd.DataFrame({
            "frame": frames_df["frame_index"].values,
            "start_s": frames_df["start_s"].values,
            "end_s": frames_df["end_s"].values,
            "mid_time_s": frames_df["mid_time_s"].values,
            "mid_time_min": time_min,
            "mean_activity_Bq_per_mL": means,
            "std_activity": stds,
            "n_voxels": [n_vox] * n_frames,
        })
        tsv_path = out_dir / f"{sub}_tac-{name.lower()}.tsv"
        with open(tsv_path, "w") as f:
            f.write(f"# subject: {subject_id}\n")
            f.write(f"# script: analysis_tissue_segmentation.py\n")
            f.write(f"# tissue_class: {name}\n")
            f.write(f"# segmentation_method: GMM_3class\n")
            f.write(f"# segmentation_mask: eroded (66 mL)\n")
            f.write(f"# roi_volume_mL: {p['volume_pet_mL']:.2f}\n")
            f.write(f"# n_voxels: {n_vox}\n")
            f.write(f"# gmm_mean_intensity: {p['mean']:.1f}\n")
            f.write(f"# gmm_std_intensity: {p['std']:.1f}\n")
            f.write(f"# gmm_weight: {p['weight']:.3f}\n")
            tac_df.to_csv(f, sep="\t", index=False, float_format="%.4f")
        print(f"  Saved: {tsv_path}")

    # Summary TSV
    summary_rows = []
    for label_val, name in TISSUE_NAMES.items():
        p = gmm_params[label_val]
        summary_rows.append({
            "tissue": name,
            "n_voxels_ct": p["n_voxels_ct"],
            "volume_ct_mL": round(p["volume_ct_mL"], 1),
            "n_voxels_pet": p["n_voxels_pet"],
            "volume_pet_mL": round(p["volume_pet_mL"], 1),
            "gmm_mean_intensity": round(p["mean"], 1),
            "gmm_std": round(p["std"], 1),
            "gmm_weight": round(p["weight"], 3),
            "peak_suv": round(p["peak_suv"], 4),
            "peak_time_min": round(p["peak_time_min"], 1),
            "plateau_suv_30_60min": round(p["plateau_suv_30_60"], 4),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"{sub}_tissue-summary.tsv"
    with open(summary_path, "w") as f:
        f.write(f"# subject: {subject_id}\n")
        f.write(f"# script: analysis_tissue_segmentation.py\n")
        f.write(f"# segmentation_method: GMM_3class\n")
        f.write(f"# segmentation_mask: eroded ({mask_vol_ml:.0f} mL)\n")
        f.write(f"# suv_factor: {suv_factor:.2f}\n")
        summary_df.to_csv(f, sep="\t", index=False, float_format="%.4f")
    print(f"  Saved: {summary_path}")

    # ==================================================================
    # Step 8: Figures
    # ==================================================================
    print("\n=== Step 8: Figures ===")
    plt.style.use(FIGURE_STYLE)

    # ---- Figure 1: GMM histogram ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(intensities, bins=200, density=True, alpha=0.4, color="gray",
            edgecolor="none", label="Voxel intensities")
    x_range = np.linspace(intensities.min(), intensities.max(), 1000)
    total_fit = np.zeros_like(x_range)
    for tissue_idx, comp_idx in enumerate(order):
        label_val = tissue_idx + 1
        name = TISSUE_NAMES[label_val]
        color = TISSUE_COLORS[label_val]
        mu = gmm.means_[comp_idx, 0]
        sigma = np.sqrt(gmm.covariances_[comp_idx, 0, 0])
        weight = gmm.weights_[comp_idx]
        gaussian = weight * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x_range - mu) / sigma) ** 2
        )
        total_fit += gaussian
        ax.plot(x_range, gaussian, color=color, linewidth=2.5,
                label=f"{name} ($\\mu$={mu:.0f}, $\\sigma$={sigma:.0f}, w={weight:.2f})")
    ax.plot(x_range, total_fit, "k--", linewidth=1.5, alpha=0.7, label="GMM total")
    ax.set_xlabel("T1w MR Intensity (a.u.)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{subject_id}: T1w Intensity Histogram (3-class GMM, eroded mask)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    plt.tight_layout()
    p1 = qc_dir / f"{sub}_tissue-gmm-histogram.png"
    fig.savefig(p1, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p1}")

    # ---- Figure 2: Segmentation QC (2x3 panel on MR) ----
    # Find centroid of mask
    coords = np.argwhere(mask_data)
    ci, cj, ck = coords.mean(axis=0).astype(int)
    print(f"  Mask centroid: i={ci}, j={cj}, k={ck}")

    # Build RGBA overlay from segmentation
    seg_rgba = np.zeros((*seg_vol.shape, 4), dtype=np.float32)
    alpha_val = 0.5
    # CSF = blue, GM = green, WM = red
    seg_rgba[seg_vol == LABEL_CSF] = [0.0, 0.3, 1.0, alpha_val]
    seg_rgba[seg_vol == LABEL_GM] = [0.0, 0.7, 0.0, alpha_val]
    seg_rgba[seg_vol == LABEL_WM] = [0.9, 0.1, 0.1, alpha_val]

    # MR display range
    mr_masked = mr_data[mask_data]
    vmin = np.percentile(mr_masked, 1) * 0.5
    vmax = np.percentile(mr_masked, 99) * 0.7

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="black")
    slices = [
        ("Sagittal", mr_data[ci, :, :].T, seg_rgba[ci, :, :, :].transpose(1, 0, 2)),
        ("Coronal", mr_data[:, cj, :].T, seg_rgba[:, cj, :, :].transpose(1, 0, 2)),
        ("Axial", mr_data[:, :, ck].T, seg_rgba[:, :, ck, :].transpose(1, 0, 2)),
    ]

    for col, (title, mr_sl, seg_sl) in enumerate(slices):
        # Top row: MR only
        ax = axes[0, col]
        ax.imshow(mr_sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, color="white", fontsize=13)
        ax.axis("off")

        # Bottom row: MR + tissue overlay
        ax = axes[1, col]
        ax.imshow(mr_sl, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        ax.imshow(seg_sl, origin="lower", interpolation="nearest")
        ax.axis("off")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="royalblue", alpha=0.7, label=f"CSF ({gmm_params[LABEL_CSF]['volume_ct_mL']:.0f} mL)"),
        Patch(facecolor="forestgreen", alpha=0.7, label=f"GM ({gmm_params[LABEL_GM]['volume_ct_mL']:.0f} mL)"),
        Patch(facecolor="firebrick", alpha=0.7, label=f"WM ({gmm_params[LABEL_WM]['volume_ct_mL']:.0f} mL)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=12,
               facecolor="black", edgecolor="white", labelcolor="white")

    fig.suptitle(f"{subject_id}: GMM Tissue Segmentation (eroded mask, {mask_vol_ml:.0f} mL)",
                 color="white", fontsize=14, y=0.98)
    plt.subplots_adjust(wspace=0.02, hspace=0.05, bottom=0.06)
    p2 = qc_dir / f"{sub}_tissue-segmentation-qc.png"
    fig.savefig(p2, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"  Saved: {p2}")

    # ---- Figure 3: Tissue TACs + IDIF (SUV, linear) ----
    fig, ax = plt.subplots(figsize=(11, 6))
    for label_val, name in TISSUE_NAMES.items():
        means = tissue_tacs[label_val][0]
        suv = means / suv_factor
        vol = gmm_params[label_val]["volume_pet_mL"]
        marker = {"CSF": "^", "GM": "o", "WM": "s"}[name]
        ax.plot(time_min, suv, f"{marker}-", color=TISSUE_COLORS[label_val],
                markersize=3, linewidth=1.5, label=f"{name} ({vol:.0f} mL)")
    ax.plot(idif_time, idif_suv, "d-", color="darkorange", markersize=2,
            linewidth=1, alpha=0.7, label="IDIF")
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("SUV", fontsize=12)
    ax.set_title(f"{subject_id}: Tissue TACs + IDIF (SUV)", fontsize=14)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p3 = qc_dir / f"{sub}_tissue-tac-suv.png"
    fig.savefig(p3, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p3}")

    # ---- Figure 4: Tissue TACs + IDIF (SUV, log y) ----
    fig, ax = plt.subplots(figsize=(11, 6))
    for label_val, name in TISSUE_NAMES.items():
        means = tissue_tacs[label_val][0]
        suv = means / suv_factor
        vol = gmm_params[label_val]["volume_pet_mL"]
        marker = {"CSF": "^", "GM": "o", "WM": "s"}[name]
        ax.plot(time_min, suv, f"{marker}-", color=TISSUE_COLORS[label_val],
                markersize=3, linewidth=1.5, label=f"{name} ({vol:.0f} mL)")
    ax.plot(idif_time, idif_suv, "d-", color="darkorange", markersize=2,
            linewidth=1, alpha=0.7, label="IDIF")
    ax.set_yscale("log")
    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("SUV (log scale)", fontsize=12)
    ax.set_title(f"{subject_id}: Tissue TACs + IDIF (SUV, log scale)", fontsize=14)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p4 = qc_dir / f"{sub}_tissue-tac-suv-log.png"
    fig.savefig(p4, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p4}")

    # ---- Figure 5: Tissue:IDIF ratio vs time ----
    valid = (idif_act > 100) & (time_min > 0.3)
    t = time_min[valid]

    fig, ax = plt.subplots(figsize=(11, 6))
    for label_val, name in TISSUE_NAMES.items():
        means = tissue_tacs[label_val][0]
        ratio = means[valid] / idif_act[valid]
        color = TISSUE_COLORS[label_val]
        marker = {"CSF": "^", "GM": "o", "WM": "s"}[name]
        ax.plot(t, ratio, f"{marker}-", color=color, markersize=4, linewidth=1.5,
                label=f"{name} / IDIF")

        # Drift analysis for this tissue (t > 15 min)
        post_eq = t >= 15
        if np.sum(post_eq) >= 3:
            coeffs = P.polyfit(t[post_eq], ratio[post_eq], 1)
            slope = coeffs[1]
            intercept = coeffs[0]
            ratio_15 = intercept + slope * 15
            ratio_135 = intercept + slope * 135
            drift = (ratio_135 - ratio_15) / ratio_15 * 100
            # Add drift to legend
            ax.plot([], [], " ", label=f"  {name} drift: {drift:+.0f}% (15-135 min)")

    ax.set_xlabel("Time (min)", fontsize=12)
    ax.set_ylabel("Tissue / IDIF ratio", fontsize=12)
    ax.set_title(f"{subject_id}: Per-Tissue Brain:IDIF Ratio vs Time", fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    p5 = qc_dir / f"{sub}_tissue-idif-ratio.png"
    fig.savefig(p5, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p5}")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n=== Summary ===")
    print(f"  {'Tissue':<6} {'Vol CT (mL)':>12} {'Vol PET (mL)':>13} {'GMM mean':>9} "
          f"{'Plateau SUV':>12} {'Drift (%)':>10}")
    print("  " + "-" * 68)
    for label_val, name in TISSUE_NAMES.items():
        p = gmm_params[label_val]
        # Compute drift
        means = tissue_tacs[label_val][0]
        ratio = means[valid] / idif_act[valid]
        post_eq = t >= 15
        coeffs = P.polyfit(t[post_eq], ratio[post_eq], 1)
        slope = coeffs[1]
        intercept = coeffs[0]
        drift = ((intercept + slope * 135) - (intercept + slope * 15)) / (intercept + slope * 15) * 100
        print(f"  {name:<6} {p['volume_ct_mL']:>12.1f} {p['volume_pet_mL']:>13.1f} "
              f"{p['mean']:>9.1f} {p['plateau_suv_30_60']:>12.4f} {drift:>+10.1f}")


if __name__ == "__main__":
    main()
