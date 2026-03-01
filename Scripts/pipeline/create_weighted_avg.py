#!/usr/bin/env python3
"""Create frame-duration-weighted average PET images for specified time windows."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def weighted_average(pet_data, frames_df, window_start_s, window_end_s):
    """Compute duration-weighted average over frames overlapping [start, end).

    Selects frames where end_s > window_start AND start_s < window_end.
    Returns (3D array, list of frame indices, list of durations used).
    """
    starts = frames_df["start_s"].values
    ends = frames_df["end_s"].values
    durations = frames_df["duration_s"].values

    sel = []
    for i in range(len(starts)):
        if ends[i] > window_start_s and starts[i] < window_end_s:
            sel.append(i)

    if not sel:
        raise ValueError(
            f"No frames overlap window {window_start_s}-{window_end_s} s"
        )

    total_weight = 0.0
    accum = np.zeros(pet_data.shape[:3], dtype=np.float64)
    sel_durations = []
    for i in sel:
        w = durations[i]
        accum += pet_data[:, :, :, i].astype(np.float64) * w
        total_weight += w
        sel_durations.append(w)

    avg = accum / total_weight
    return avg.astype(np.float32), sel, sel_durations


def main():
    subject_id = "SUB001_20260225"
    base = Path(__file__).resolve().parent.parent.parent
    sub = f"sub-{subject_id}"

    pet_path = base / "DerivedData" / subject_id / "PET1" / f"{sub}_pet.nii.gz"
    frames_path = base / "DerivedData" / subject_id / "PET1" / f"{sub}_frames.tsv"
    out_dir = base / "DerivedData" / subject_id / "PET1"

    # Load data
    print(f"Loading PET: {pet_path}")
    pet_img = nib.load(pet_path)
    pet_data = pet_img.get_fdata()
    print(f"  Shape: {pet_data.shape}")

    frames_df = pd.read_csv(frames_path, sep="\t", comment="#")
    print(f"  Frames: {len(frames_df)}")

    # Reference header for 3D output (drop 4th dimension)
    ref_affine = pet_img.affine
    ref_header = pet_img.header.copy()
    ref_header.set_data_shape(pet_data.shape[:3])
    ref_header.set_data_dtype(np.float32)

    # Time windows: (label, start_min, end_min)
    windows = [
        ("20-60min",  20, 60),
        ("10-60min",  10, 60),
        ("30-60min",  30, 60),
        ("20-45min",  20, 45),
        ("60-90min",  60, 90),
    ]

    for label, start_min, end_min in windows:
        start_s = start_min * 60
        end_s = end_min * 60

        avg, sel_frames, sel_durs = weighted_average(
            pet_data, frames_df, start_s, end_s
        )

        fname = f"{sub}_pet_avg-{label}.nii.gz"
        out_path = out_dir / fname

        out_img = nib.Nifti1Image(avg, ref_affine, ref_header)
        nib.save(out_img, out_path)

        total_dur = sum(sel_durs)
        mean_val = float(np.mean(avg[avg > 0])) if np.any(avg > 0) else 0.0
        print(
            f"  {label}: frames {sel_frames[0]}-{sel_frames[-1]} "
            f"({frames_df['start_s'].iloc[sel_frames[0]]:.0f}-"
            f"{frames_df['end_s'].iloc[sel_frames[-1]]:.0f} s), "
            f"total weight {total_dur:.0f} s, "
            f"mean {mean_val:.1f} Bq/mL -> {out_path.name}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
