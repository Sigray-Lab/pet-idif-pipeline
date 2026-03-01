# Dynamic PET Imaging Pipeline

Dynamic PET imaging pipeline for brain radiotracer studies. Converts raw DICOM data to NIfTI/BIDS format, performs MR brain segmentation and CT-MR coregistration, extracts time-activity curves, computes SUV and %ID metrics, derives an image-derived input function (IDIF), and fits kinetic models.

## Requirements

- Python >= 3.9
- [dcm2niix](https://github.com/rordenlab/dcm2niix) (DICOM to NIfTI conversion)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
pet_idif_pipeline/
├── CLAUDE.md                 # Detailed pipeline specification
├── README.md                 # This file
├── requirements.txt          # Pinned Python dependencies
├── raw/                      # Raw data (see raw/README.md)
├── DerivedData/              # Imaging derivatives (see DerivedData/README.md)
├── Scripts/
│   └── pipeline/
│       ├── run_pipeline.py   # Master runner
│       ├── config.py         # All parameters and defaults
│       ├── s00_dcm2nii.py    # DICOM to NIfTI + mask copy
│       ├── s00b_segment_mr.py# CT-guided brain extraction
│       ├── s00c_coregister.py# CT-MR coregistration
│       ├── s01_extract_tac.py# Raw brain TAC extraction
│       ├── s02_suv_tac.py    # SUV TAC
│       ├── s03_percent_id.py # %ID TAC + summary
│       ├── s04_idif.py       # Image-derived input function
│       └── s05_kinetics.py   # Kinetic modeling (Patlak, Logan, 1/2TCM)
├── Outputs/                  # TSV tables (generated)
├── QC/                       # QC figures (generated)
└── Logs/                     # Pipeline logs (generated)
```

## Quick Start

```bash
# Run the full pipeline for a subject
python Scripts/pipeline/run_pipeline.py \
    --subject SUB001_20260225 \
    --base-dir /path/to/pet_idif_pipeline \
    --verbose

# Run specific steps
python Scripts/pipeline/run_pipeline.py \
    --subject SUB001_20260225 \
    --steps s01 s02 s03 --force

# Force rerun from a specific step onward
python Scripts/pipeline/run_pipeline.py \
    --subject SUB001_20260225 \
    --force-from s01
```

## Pipeline Steps

| Step | Script | Description | Key Outputs |
|------|--------|-------------|-------------|
| s00 | `s00_dcm2nii.py` | DICOM to NIfTI conversion, mask copy | 4D PET, CT, MR NIfTIs; frame timing TSV/JSON |
| s00b | `s00b_segment_mr.py` | CT-guided brain extraction (ICC) | Brain mask in MR space, bias-corrected MR |
| s00c | `s00c_coregister.py` | CT-MR coregistration, mask resampling | Brain mask in PET space |
| s01 | `s01_extract_tac.py` | Raw brain TAC from whole-brain mask | `tac-raw.tsv`, TAC plot |
| s02 | `s02_suv_tac.py` | SUV normalization | `tac-suv.tsv`, SUV plot |
| s03 | `s03_percent_id.py` | %ID calculation + summary | `tac-pctID.tsv`, `summary.tsv` |
| s04 | `s04_idif.py` | Image-derived input function | `idif.tsv`, vascular mask, combined plots |
| s05 | `s05_kinetics.py` | Patlak, Logan, 1TCM, 2TCM fitting | `kinetics-results.tsv`, model fit plots |

### Step Dependency Graph

```
s00 --> s00b --> s00c --> s01 --> s02
                          |
                          +--> s03
s00 --> s04
              s01 + s04 --> s05
```

## Frame Protocol

Dynamic PET acquisition: 43 frames over 150 minutes.

| Block | Frames | Duration (s) | Time Window |
|-------|--------|-------------|-------------|
| 1 | 6 | 10 | 0 - 1 min |
| 2 | 6 | 20 | 1 - 3 min |
| 3 | 4 | 30 | 3 - 5 min |
| 4 | 10 | 60 | 5 - 15 min |
| 5 | 5 | 180 | 15 - 30 min |
| 6 | 6 | 300 | 30 - 60 min |
| 7 | 3 | 600 | 60 - 90 min |
| 8 | 2 | 900 | 90 - 120 min |
| 9 | 1 | 1800 | 120 - 150 min |

## Configuration

All tunable parameters are in `Scripts/pipeline/config.py`. Override any parameter from the CLI:

```bash
python Scripts/pipeline/run_pipeline.py \
    --subject SUB001_20260225 \
    --param PATLAK_T_STAR_MIN=20 IDIF_MIN_VOXELS_PER_SLICE=10
```

## Data Preparation

See [`raw/README.md`](raw/README.md) for how to organize raw data and [`DerivedData/README.md`](DerivedData/README.md) for manually-provided masks.

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `fill_outline_mask.py` | Fill hand-drawn brain outline masks (2D per-slice) |
| `create_weighted_avg.py` | Frame-duration-weighted average PET images |
| `process_manual_wb_mask.py` | Resample manual whole-brain mask, erode, extract TAC |

## Caching

Each pipeline step checks whether its outputs already exist and are newer than its inputs. Cached steps are skipped automatically. Use `--force` to rerun specific steps or `--force-all` to rerun everything.

## Reproducibility

- All outputs include provenance headers with input file MD5 hashes, parameter values, and timestamps
- All parameters are centralized in `config.py` with documented defaults
- Dependency versions are pinned in `requirements.txt`
- Given the same raw data and manual masks, every run produces identical numerical outputs
