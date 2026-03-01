# Raw Data Directory

This directory contains the raw imaging data and radiochemistry metadata required to run the pipeline. Raw data is not tracked in version control due to size and privacy.

## Directory Structure

```
raw/
├── README.md               # This file
├── Radiochem.csv           # Radiochemistry and subject metadata (all subjects)
└── <SubjectID>/
    ├── CT1/                # Alignment CT (DICOM files)
    ├── PET1/               # Dynamic PET acquisition (DICOM files, all frames)
    ├── MR/                 # Structural MRI (NIfTI, T1w, 0.5 mm isotropic)
    ├── CT_Scan_3_resliced_to_PET_mask.nii.gz              # Brain mask (PET space)
    └── CT_Scan_3_resliced_to_PET_loose_neck_mask_FINAL.nii.gz  # Neck mask (PET space)
```

## Subject Naming Convention

Subject IDs follow the format: `<ID>_<YYYYMMDD>`

Example: `SUB001_20260225`

## Radiochem.csv

Tab- or comma-separated file with per-session radiochemistry metadata. Required columns:

| Column | Description | Example |
|--------|-------------|---------|
| `id` | Subject ID (matches directory name) | `SUB001_20260225` |
| `sub` | Short subject name | `sub001` |
| `ses` | Session number | `1` |
| `injected_MBq` | Injected dose in MBq | `50.0` |
| `weight_kg` | Body weight in kg | `70.0` |

## Per-Subject Data

### CT1/ (Alignment CT)

DICOM files from the alignment CT scan acquired during the PET session. Typically ~600 slices at 0.32 mm resolution covering the full body.

### PET1/ (Dynamic PET)

DICOM files from the dynamic PET acquisition. The expected protocol is 43 frames over 150 minutes (see the frame protocol in the main README). All frames should be in a single directory.

### MR/ (Structural MRI)

A single T1-weighted structural MRI in NIfTI format (.nii or .nii.gz), at approximately 0.5 mm isotropic resolution. This is used for brain extraction and coregistration.

## Required Manual Masks

Two masks must be manually created in PET space and placed in the subject directory. These define regions of interest for the pipeline.

### Brain mask

**Filename**: `CT_Scan_3_resliced_to_PET_mask.nii.gz`

A conservative binary brain mask in PET space. Must have the same dimensions and affine as the 4D PET data. This is used as a fallback if the automated whole-brain mask is not available. It can be drawn on the PET or CT image and resliced to PET space.

### Loose neck mask

**Filename**: `CT_Scan_3_resliced_to_PET_loose_neck_mask_FINAL.nii.gz`

A loose bounding mask covering the carotid and jugular vessels in the neck region, in PET space. This defines the search region for the image-derived input function (IDIF) in step s04. Should generously cover the major neck vessels without including too much surrounding tissue.

Both masks must match the 4D PET volume dimensions and affine exactly. The pipeline will abort with an error if they do not match.
