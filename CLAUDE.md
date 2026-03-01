# Dynamic PET Imaging Pipeline

## Project overview

This is a dynamic PET imaging pipeline for brain radiotracer studies. Each scan session consists of one dynamic PET acquisition, one alignment CT, and one structural MRI. The pipeline converts raw DICOM data to NIfTI/BIDS format, performs MR brain segmentation and CT-MR coregistration, extracts time-activity curves (TACs), computes SUV and %ID metrics, and derives an image-derived input function (IDIF) from the carotid/jugular vasculature.

## Core principles

### Reproducibility

This pipeline must be fully reproducible. Given the same raw data and manual masks, every run must produce identical numerical outputs (bit-for-bit where possible, numerically identical otherwise). To ensure this:

- Never use random seeds without fixing them. If any step uses randomness, set `np.random.seed(42)` or equivalent.
- All thresholds, parameters, and constants must be defined in a single config file or as explicit CLI arguments with documented defaults. No magic numbers buried in code.
- Every output file must include a provenance header (see Output format section below).
- The pipeline must be deterministic: running `run_pipeline.py` twice on the same input must yield the same output.
- Pin all dependency versions in a `requirements.txt`.

### Intermediate caching and force-rerun

Each pipeline step writes its outputs to `DerivedData/` (for imaging derivatives), `Outputs/` (for final tables), or `QC/` (for figures and QC images). Before running, each step checks whether its expected outputs already exist AND are newer than its inputs. If so, the step is skipped with a log message:

```
[2026-02-25 14:32:01] [INFO ] [s01_extract_tac] SKIP: outputs exist and are up-to-date.
                       Use --force to rerun. Output: Outputs/sub-SUB001_tac-raw.tsv
```

Force-rerun behavior:
- `--force`: rerun the specified step(s) even if outputs exist.
- `--force-from <step>`: rerun from the given step onward (e.g., `--force-from s02` reruns steps 2, 3, 4 but not 0, 1).
- `--force-all`: nuke all derived outputs and rerun everything from scratch.

Implementation pattern for each step:

```python
def check_outputs_current(inputs: list[Path], outputs: list[Path]) -> bool:
    """Return True if all outputs exist and are newer than all inputs."""
    if not all(o.exists() for o in outputs):
        return False
    oldest_output = min(o.stat().st_mtime for o in outputs)
    newest_input = max(i.stat().st_mtime for i in inputs)
    return oldest_output > newest_input
```

## Directory structure

```
pet_idif_pipeline/
├── CLAUDE.md                  <- you are here
├── README.md                  <- project overview for GitHub
├── requirements.txt           <- pinned Python dependencies
├── .gitignore
├── raw/
│   ├── Radiochem.csv          <- radiochemistry, injection info, body weight
│   ├── README.md              <- documents raw data structure
│   └── <SubjectID>/
│       ├── CT1/               <- DICOM: alignment CT (631 files)
│       ├── MR/                <- NIfTI: structural MRI (T1w, 0.5mm iso)
│       ├── PET1/              <- DICOM: dynamic PET (5375 files, all frames)
│       ├── blood/             <- venous blood sample data (see Blood data section)
│       ├── CT_Scan_3_resliced_to_PET_mask.nii.gz          <- manual brain mask (PET space)
│       └── CT_Scan_3_resliced_to_PET_loose_neck_mask_FINAL.nii.gz <- manual loose neck mask (PET space)
├── DerivedData/
│   ├── README.md              <- documents derived data and mask workflow
│   └── <SubjectID>/
│       ├── CT1/               <- NIfTI CT (full + cropped)
│       ├── MR/                <- NIfTI MR (original + N4 bias-corrected + warped to CT)
│       ├── PET1/              <- NIfTI 4D PET + sidecar + frames.tsv + weighted averages
│       └── masks/             <- all masks (renamed, provenance-tracked)
├── Scripts/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── run_pipeline.py    <- master runner
│   │   ├── config.py          <- all parameters, thresholds, defaults
│   │   ├── logging_setup.py   <- logging configuration
│   │   ├── cache.py           <- output caching / freshness checks
│   │   ├── s00_dcm2nii.py     <- DICOM to NIfTI + mask copy
│   │   ├── s00b_segment_mr.py <- CT-guided brain extraction (ICC from bone thresholding)
│   │   ├── s00c_coregister.py <- CT-MR coregistration + mask resampling to PET
│   │   ├── s01_extract_tac.py <- raw brain TAC extraction (uses whole-brain mask)
│   │   ├── s02_suv_tac.py     <- SUV TAC
│   │   ├── s03_percent_id.py  <- %ID TAC + summary (volume from mask, not assumed)
│   │   ├── s04_idif.py        <- image-derived input function
│   │   ├── s05_kinetics.py    <- Patlak, Logan, 1TCM, 2TCM kinetic modeling
│   │   ├── fill_outline_mask.py         <- fill hand-drawn outline masks (2D per-slice)
│   │   ├── create_weighted_avg.py       <- frame-duration-weighted average PET images
│   │   └── process_manual_wb_mask.py    <- erode manual WB mask, extract TAC, plot
│   ├── analysis_brain_idif_ratio.py     <- brain:IDIF ratio vs time (equilibrium check)
│   ├── analysis_tissue_segmentation.py  <- GMM tissue segmentation (CSF/GM/WM TACs)
│   └── analysis_blood_calibration.py    <- venous blood calibration + kinetic comparison
├── Outputs/                   <- TSV tables only
├── QC/                        <- all figures and QC images (300 DPI PNG)
└── Logs/                      <- one log file per pipeline run
```

Subject naming convention: `<ID>_<YYYYMMDD>`
Example: `SUB001_20260225`

## Pipeline step dependency graph

```
s00 --> s00b --> s00c --> s01 --> s02
  |       |                |
  |       |                +--> s03
  |       |
  |       +-- (s00b produces CT crop + MR-to-CT transform, reused by s00c)
  |
  +--> s04 --------+
                   |
             s01 --+--> s05 (Patlak, Logan, 1TCM, 2TCM)
```

s00b performs CT-guided brain extraction (ICC) and rigid MR-to-CT registration. s00c reuses the cropped CT and transform from s00b, then resamples the brain mask to PET space. s01 uses the whole-brain mask from s00c (or from a manually drawn filled mask if available). s03 reads brain volume from the s01 provenance header. s05 requires both the brain TAC (from s01) and the IDIF (from s04).

## Mask management

All masks originate in `raw/<SubjectID>/` and are copied (never moved) into `DerivedData/<SubjectID>/masks/` with standardized names on first pipeline run (step 0).

| Source | Destination (DerivedData/masks/) | Description |
|---|---|---|
| `CT_Scan_3_resliced_to_PET_mask.nii.gz` | `sub-<id>_space-PET_mask-brain.nii.gz` | Original conservative brain mask (legacy, not used by default) |
| `CT_Scan_3_resliced_to_PET_loose_neck_mask_FINAL.nii.gz` | `sub-<id>_space-PET_mask-neck-loose.nii.gz` | Loose neck bounding region for IDIF voxel search |
| generated by s00b | `sub-<id>_space-MR_mask-brain-whole.nii.gz` | CT-guided ICC brain mask (MR space) |
| generated by s00b | `sub-<id>_space-MR_mask-icc-ct-guided.nii.gz` | ICC mask in MR space (intermediate) |
| generated by s00b | `sub-<id>_space-MR_mask-brain-ct-guided.nii.gz` | Brain mask in MR space from CT-guided method |
| generated by s00b | `sub-<id>_space-CT_mask-icc.nii.gz` | ICC mask at CT resolution |
| generated by s00c or process_manual_wb_mask.py | `sub-<id>_space-PET_mask-brain-whole.nii.gz` | Whole-brain mask in PET space (used by s01) |
| generated by s04 | `sub-<id>_space-PET_mask-idif.nii.gz` | Final IDIF vascular mask (~290 voxels, per-slice) |
| manually drawn, filled by fill_outline_mask.py | `sub-<id>_mr-in-ct_WB_mask.nii.gz` | Raw hand-drawn outline mask (CT space) |
| fill_outline_mask.py | `sub-<id>_mr-in-ct_WB_mask_filled.nii.gz` | Filled manual whole-brain mask (CT space) |
| process_manual_wb_mask.py | `sub-<id>_space-PET_mask-brain-eroded6.nii.gz` | Eroded whole-brain mask for TAC extraction |
| process_manual_wb_mask.py | `sub-<id>_mr-in-ct_WB_mask_filled_eroded6.nii.gz` | Eroded mask at CT resolution |
| analysis_tissue_segmentation.py | `sub-<id>_space-CT_seg-tissue.nii.gz` | 3-class tissue label map (CT space; 1=CSF, 2=GM, 3=WM) |
| analysis_tissue_segmentation.py | `sub-<id>_space-CT_mask-csf.nii.gz` | CSF binary mask (CT space) |
| analysis_tissue_segmentation.py | `sub-<id>_space-CT_mask-gm.nii.gz` | GM binary mask (CT space) |
| analysis_tissue_segmentation.py | `sub-<id>_space-CT_mask-wm.nii.gz` | WM binary mask (CT space) |
| analysis_tissue_segmentation.py | `sub-<id>_space-PET_mask-csf.nii.gz` | CSF mask in PET space |
| analysis_tissue_segmentation.py | `sub-<id>_space-PET_mask-gm.nii.gz` | GM mask in PET space |
| analysis_tissue_segmentation.py | `sub-<id>_space-PET_mask-wm.nii.gz` | WM mask in PET space |

**Brain volume for quantification**: The pipeline uses the whole-brain mask volume from `mask-brain-whole.nii.gz` in PET space. s01 preferentially loads this mask (falling back to the legacy conservative mask if unavailable). s03 reads the brain volume from the s01 provenance header.

**Eroded mask for TAC**: A separate eroded mask (6 voxels per slice at CT resolution, ~1.9mm inward) is available for extracting brain TACs with reduced partial volume effects. This is generated by `process_manual_wb_mask.py` and saved alongside the combined TAC+IDIF plots.

When copying masks, verify that each mask has the same dimensions and affine as the 4D PET. If they differ, abort with a clear error message.

## Logging system

### Requirements

Every pipeline run must produce a comprehensive, human-readable log. Logging is not optional: if logging fails, the pipeline should not proceed.

### Log file

Each invocation of `run_pipeline.py` creates a new log file:
```
Logs/pipeline_<SubjectID>_<YYYYMMDD>_<HHMMSS>.log
```

### Log format

Use Python's `logging` module. All log output goes to both the log file and stdout simultaneously (dual handler). Format:

```
[2026-02-25 14:32:01.123] [INFO ] [s01_extract_tac] Loading 4D PET: DerivedData/SUB001_.../PET1/sub-SUB001_pet.nii.gz
[2026-02-25 14:32:01.456] [INFO ] [s01_extract_tac] PET shape: (180, 180, 125, 43), voxel size: (1.2, 1.2, 1.2) mm
[2026-02-25 14:32:01.789] [INFO ] [s01_extract_tac] Brain mask: N voxels, volume = X.XX mL
[2026-02-25 14:32:02.100] [WARN ] [s01_extract_tac] Frame 0 has 342 zero-valued voxels inside mask (1.6%)
[2026-02-25 14:32:05.500] [INFO ] [s01_extract_tac] DONE: wrote Outputs/sub-SUB001_tac-raw.tsv (43 frames)
```

### What to log

For every step, log the following at minimum:

**On entry:**
- Step name, subject ID, timestamp
- All input file paths and their existence/size/modification time
- All parameter values being used (from config or CLI overrides)
- Whether outputs are cached (and if so, skip or force-rerun)

**During processing:**
- Image dimensions, voxel sizes, data types on load
- Mask statistics: N voxels, volume in mL, bounding box
- Per-frame statistics at DEBUG level: frame index, mean, min, max, std within ROI
- Any warnings: mismatched dimensions, NaN/Inf values, unexpected frame count, zero voxels in mask

**On completion:**
- Output file paths and sizes
- Summary statistics: peak value and time, plateau mean, etc.
- Wall-clock time for the step

**On error:**
- Full traceback
- Input state at time of failure
- Suggestion for resolution if possible

### Log levels

- `DEBUG`: per-frame values, intermediate computations (verbose, off by default)
- `INFO`: step entry/exit, file I/O, summary stats (default)
- `WARNING`: non-fatal issues (dimension mismatches below tolerance, NaN voxels, etc.)
- `ERROR`: fatal issues that stop the step but not necessarily the whole pipeline

Enable DEBUG with `--verbose` or `-v` flag.

### Implementation

`Scripts/pipeline/logging_setup.py`:

```python
def setup_logging(subject_id: str, log_dir: Path, verbose: bool = False) -> logging.Logger:
    """
    Configure root logger with file + stdout handlers.
    Returns the configured logger.
    Call once at pipeline start.
    """
```

Each step script gets its own child logger:
```python
log = logging.getLogger("s01_extract_tac")
```

## Configuration

`Scripts/pipeline/config.py` holds all tunable parameters as a dataclass with defaults:

```python
# Frame protocol (expected)
EXPECTED_N_FRAMES = 43
EXPECTED_TOTAL_DURATION_S = 9000  # 150 min

# IDIF parameters
IDIF_PEAK_SEARCH_FRAMES = 15          # search first N frames for arterial peak
IDIF_SUM_WINDOW_START_S = 30.0        # start of first-pass sum window (frames 3-6)
IDIF_SUM_WINDOW_END_S = 80.0          # end of first-pass sum window
IDIF_MIN_VOXELS_PER_SLICE = 8         # minimum voxels per z-slice in IDIF mask
IDIF_TOP_PERCENTILE = 99.5            # starting percentile for per-slice threshold
IDIF_MIN_CLUSTER_SIZE = 3             # minimum connected-component size

# SUV and %ID sanity checks
INJECTED_DOSE_RANGE_MBQ = (10, 200)   # flag if outside range
BODY_WEIGHT_RANGE_KG = (1, 150)       # flag if outside range

# MR brain extraction (CT-guided)
MR_BRAIN_VOLUME_RANGE_ML = (20, 2000) # sanity check range for ICC/brain mask
CT_BONE_THRESHOLD_HU = 200            # HU threshold for bone in CT
CT_CLOSE_ITER_1 = 5                   # first-stage morphological closing iterations
CT_CLOSE_ITER_2 = 10                  # second-stage closing (two-stage prevents cavity fill)
CT_INFERIOR_CAP_MM = 35               # mm below bone centroid to seal foramen magnum

# CT cropping for coregistration (voxel indices)
CT_CROP_LR = (167, 499)
CT_CROP_AP = (137, 567)
CT_CROP_Z = (147, 480)

# Patlak graphical analysis
PATLAK_T_STAR_MIN = 15.0          # start of linear phase (minutes)
PATLAK_T_END_MIN = 80.0           # end of analysis window (minutes)

# Compartment models (1TCM, 2TCM)
TCM_T_END_MIN = 150.0             # end of fitting window (minutes), full scan
TCM_VB = 0.05                     # fixed blood volume fraction (5%)
TCM_DT_S = 1.0                    # interpolation time step (seconds)

# Plotting
FIGURE_DPI = 300
FIGURE_STYLE = "seaborn-v0_8-whitegrid"
```

Every parameter must be overridable from the CLI via `run_pipeline.py --param KEY=VALUE`.

## Radiochemistry CSV

`raw/Radiochem.csv` contains per-session metadata. Exact columns: `id`, `sub`, `ses`, `injected_MBq`, `weight_kg`.

Example row: `SUB001_20260225,sub001,1,50.0,70.0`

Parse this file to compute SUV and %ID:
- SUV = (tissue_activity_Bq_per_mL) / (injected_dose_Bq / body_weight_g)
- %ID = (mean_activity_Bq_per_mL * brain_volume_mL) / injected_dose_Bq * 100

## Blood data

`raw/<SubjectID>/blood/` contains venous blood sample data from well-counter measurements:

| File | Description |
|---|---|
| `blood_manual.txt` | Tab-separated, 12 rows. Columns: `ABSS sec`, `Cbl disp corr` (nCi/cc), `Cpl (nCi/cc)`. Decay-corrected to injection time. |
| `anc.mat` | MATLAB ancillary structure from KI "obtain_blood" pipeline. Contains raw (uncorrected) well counter data with measurement times, volumes, and decay-corrected concentrations. |
| `parent_fraction.txt` | Empty. No metabolite analysis was performed. |
| `blood_uncorr.txt` | Empty. |
| `plots.fig` | MATLAB figure from the obtain_blood pipeline. |
| `Blood_Plasma-measurments_*.xlsx` | Excel protocol with sampling times, volumes, and well counter rack positions. |
| `Blood_Plasma-measurments_*.pdf` | Scanned handwritten well counter measurement sheet. |

Unit conversion: 1 nCi/cc = 37 Bq/mL (exact).

Key metadata from the .mat file:
- Decay correction reference: injection time
- Isotope: C-11, injection type: bolus
- 12 discrete venous samples from 60s to 9000s post-injection
- Blood counted 1-3 min after draw; plasma counted 5-9 min after draw (centrifugation delay)
- No haematocrit, no parent fraction (metabolite) data, no continuous arterial sampling

## Frame protocol

The expected dynamic PET frame protocol (verify against DICOM headers):

| Block | N frames | Duration (s) | Cumulative time |
|-------|----------|-------------|-----------------|
| 1     | 6        | 10          | 0-1 min         |
| 2     | 6        | 20          | 1-3 min         |
| 3     | 4        | 30          | 3-5 min         |
| 4     | 10       | 60          | 5-15 min        |
| 5     | 5        | 180         | 15-30 min       |
| 6     | 6        | 300         | 30-60 min       |
| 7     | 3        | 600         | 60-90 min       |
| 8     | 2        | 900         | 90-120 min      |
| 9     | 1        | 1800        | 120-150 min     |

Total: 43 frames, 150 minutes (9000 seconds).

When converting DICOM to NIfTI, extract frame timing from DICOM headers (tags 0018,1242 ActualFrameDuration and 0054,1300 FrameReferenceTime, or from the per-frame functional groups sequence). Store frame timing in a BIDS-compliant JSON sidecar (`_pet.json`) with fields `FrameTimesStart` and `FrameDuration` (both arrays, in seconds). Also save a `_frames.tsv` with columns: frame_index, start_s, end_s, duration_s, mid_time_s, mid_time_min.

Always verify extracted frame times against the expected protocol above. Print a warning if they do not match.

## Output format

### TSV files

All tabular outputs use tab-separated values with:
1. A provenance header block (lines prefixed with `#`):
```
# subject: SUB001_20260225
# script: s01_extract_tac.py
# version: 1.0.0
# date: 2026-02-25T14:32:01
# inputs: DerivedData/.../sub-SUB001_pet.nii.gz (md5: abc123...)
#         DerivedData/.../masks/sub-SUB001_space-PET_mask-brain.nii.gz (md5: def456...)
# parameters: {}
```
2. A column header row.
3. Data rows.

### Figures

All figures go to `QC/` (not `Outputs/Figures/`). Format: 300 DPI PNG. Use `plt.style.use('seaborn-v0_8-whitegrid')` or similar clean style. Always include: axis labels with units, subject ID in title, legend if multiple traces. Save with `bbox_inches='tight'`.

## Pipeline steps

Build each step as a separate script that can also be imported and called from `run_pipeline.py`. Each script must:
1. Check if outputs are current (skip if cached, unless `--force`).
2. Log entry with all parameters.
3. Validate inputs before processing.
4. Write outputs with provenance headers.
5. Log completion with summary and wall-clock time.

### Step 0: DICOM to NIfTI conversion

Script: `Scripts/pipeline/s00_dcm2nii.py`

Inputs:
- `raw/<SubjectID>/PET1/` (DICOM directory)
- `raw/<SubjectID>/CT1/` (DICOM directory)
- `raw/<SubjectID>/MR/` (NIfTI T1w MR, 0.5mm iso)

Outputs:
- `DerivedData/<SubjectID>/PET1/sub-<SubjectID>_pet.nii.gz`
- `DerivedData/<SubjectID>/PET1/sub-<SubjectID>_pet.json` (BIDS sidecar with frame timing)
- `DerivedData/<SubjectID>/PET1/sub-<SubjectID>_frames.tsv`
- `DerivedData/<SubjectID>/CT1/sub-<SubjectID>_ct.nii.gz`
- `DerivedData/<SubjectID>/MR/sub-<SubjectID>_mr.nii.gz`

Also:
- Copy and rename masks from `raw/` to `DerivedData/<SubjectID>/masks/` (see Mask management).
- Parse DICOM headers to build the frame timing sidecar.
- Verify frame count in the 4D NIfTI matches expected 43 frames. Warn if not.
- Log the DICOM-extracted frame times and flag any mismatch with expected protocol.

### Step 0b: CT-guided brain extraction

Script: `Scripts/pipeline/s00b_segment_mr.py`

Uses the co-acquired CT to define the intracranial cavity (ICC) via bone thresholding, then maps it back to MR space. This approach is robust because CT bone contrast is unambiguous, unlike MR intensity-based methods which can fail on challenging anatomies.

Algorithm:
1. N4 bias field correction on MR: `ants.n4_bias_field_correction()`
2. Crop CT to head region (configurable voxel indices)
3. Rigid registration (6 DOF) of MR to cropped CT: `ants.registration(type_of_transform='Rigid')`
4. Resample CT to 1mm isotropic for morphological operations
5. Bone thresholding at 200 HU
6. Two-stage morphological closing (5 + 10 iterations; critical: single-stage 15-iter fills entire cavity)
7. Inferior cap to seal foramen magnum (35mm below bone centroid along S-I axis)
8. `binary_fill_holes` to identify enclosed intracranial cavity
9. Use closed bone as boundary: `ICC = filled & ~bone_closed`
10. Clip below inferior cap, extract largest connected component
11. Resample ICC back to CT native resolution, transform to MR space via inverse rigid
12. Result: ICC mask in MR space

The registration transform and cropped CT are saved for reuse by s00c.

Outputs:
- `DerivedData/<SubjectID>/MR/sub-<SubjectID>_mr_n4.nii.gz` (bias-corrected MR)
- `DerivedData/<SubjectID>/CT1/sub-<SubjectID>_ct-cropped.nii.gz` (reused by s00c)
- `DerivedData/<SubjectID>/MR/sub-<SubjectID>_mr_to_ct_0GenericAffine.mat` (reused by s00c)
- `DerivedData/<SubjectID>/masks/sub-<SubjectID>_space-MR_mask-brain-whole.nii.gz`
- `QC/sub-<SubjectID>_mr-brain-mask-qc.png`

**Note**: The automated ICC provides a reasonable starting point but can be superseded by a manually drawn whole-brain mask. Draw outlines in CT space, fill using `fill_outline_mask.py`, and resample to PET space via `process_manual_wb_mask.py`.

### Step 0c: CT-MR coregistration and mask resampling

Script: `Scripts/pipeline/s00c_coregister.py`

Reuses the cropped CT and MR-to-CT rigid transform from s00b (if available; otherwise performs its own registration). Warps the MR brain mask from MR space to PET space via CT.

Algorithm:
1. Reuse cropped CT from s00b (or crop fresh if not available)
2. Reuse MR-to-CT rigid transform from s00b (or register fresh)
3. Apply transform to brain mask using the **full CT** as reference (not cropped, to avoid clipping at FOV boundaries)
4. Resample mask from CT resolution (0.32mm) to PET resolution (1.2mm) using nearest-neighbor interpolation

**Critical**: The registration uses the cropped CT for quality, but `ants.apply_transforms()` for the brain mask uses the full CT as reference. The transform works in world coordinates so any reference grid works.

**Note**: The `mask-brain-whole.nii.gz` in PET space may be overwritten by `process_manual_wb_mask.py` if a manually drawn whole-brain mask is available. The pipeline always uses whatever is at this path.

Outputs:
- `DerivedData/<SubjectID>/CT1/sub-<SubjectID>_ct-cropped.nii.gz`
- `DerivedData/<SubjectID>/MR/sub-<SubjectID>_mr-in-ct.nii.gz` (warped MR, for QC)
- `DerivedData/<SubjectID>/MR/sub-<SubjectID>_mr_to_ct_0GenericAffine.mat` (rigid transform)
- `DerivedData/<SubjectID>/masks/sub-<SubjectID>_space-PET_mask-brain-whole.nii.gz`
- `QC/sub-<SubjectID>_coregistration-qc.png`
- `QC/sub-<SubjectID>_mask-brain-whole-on-pet.png`

### Step 1: Extract raw TAC from brain mask

Script: `Scripts/pipeline/s01_extract_tac.py`

Preferentially uses the whole-brain mask (`mask-brain-whole.nii.gz`) for mean concentration extraction. Falls back to the legacy conservative mask (`mask-brain.nii.gz`) if the whole-brain mask is unavailable. The mask volume is recorded in the provenance header and propagated to s03 for %ID calculation.

Inputs:
- `DerivedData/<SubjectID>/PET1/sub-<SubjectID>_pet.nii.gz`
- `DerivedData/<SubjectID>/masks/sub-<SubjectID>_space-PET_mask-brain-whole.nii.gz` (preferred) or `mask-brain.nii.gz` (fallback)
- `DerivedData/<SubjectID>/PET1/sub-<SubjectID>_frames.tsv`

Outputs:
- `Outputs/sub-<SubjectID>_tac-raw.tsv` (columns: frame, start_s, end_s, mid_time_s, mid_time_min, mean_activity_Bq_per_mL, std_activity, min_activity, max_activity, n_voxels)
- `QC/sub-<SubjectID>_tac-raw.png`

### Step 2: SUV TAC

Script: `Scripts/pipeline/s02_suv_tac.py`

Inputs:
- `Outputs/sub-<SubjectID>_tac-raw.tsv`
- `raw/Radiochem.csv`

Outputs:
- `Outputs/sub-<SubjectID>_tac-suv.tsv`
- `QC/sub-<SubjectID>_tac-suv.png`

SUV = activity_Bq_per_mL / (injected_dose_Bq / body_weight_g).

### Step 3: %ID table

Script: `Scripts/pipeline/s03_percent_id.py`

Uses the brain volume from the mask used in step 1 (read from the s01 provenance header `roi_volume_mL` field).

%ID = (mean_Bq_per_mL * brain_volume_mL) / injected_dose_Bq * 100

Inputs:
- `Outputs/sub-<SubjectID>_tac-raw.tsv`
- `raw/Radiochem.csv`

Outputs:
- `Outputs/sub-<SubjectID>_tac-pctID.tsv` (columns: frame, mid_time_min, mean_activity, pct_ID)
- `Outputs/sub-<SubjectID>_summary.tsv` (single-row summary: peak_%ID, peak_time_min, plateau_%ID_30_60min, last_frame_%ID, roi_volume_mL, injected_dose_MBq, body_weight_kg, suv_plateau)
- `QC/sub-<SubjectID>_tac-pctID.png`

### Step 4: Image-derived input function (IDIF)

Script: `Scripts/pipeline/s04_idif.py`

Extracts a vascular TAC from the carotid/jugular vessels without arterial blood sampling.

#### Inputs

- `DerivedData/<SubjectID>/PET1/sub-<SubjectID>_pet.nii.gz`
- `DerivedData/<SubjectID>/masks/sub-<SubjectID>_space-PET_mask-neck-loose.nii.gz`
- `DerivedData/<SubjectID>/PET1/sub-<SubjectID>_frames.tsv`
- `raw/Radiochem.csv` (for SUV combined plot)

#### Outputs

- `DerivedData/<SubjectID>/masks/sub-<SubjectID>_space-PET_mask-idif.nii.gz` (the final ~290 voxel vascular mask)
- `DerivedData/<SubjectID>/PET1/sub-<SubjectID>_idif-summed-firstpass.nii.gz` (QC: summed first-pass image)
- `Outputs/sub-<SubjectID>_idif.tsv`
- `QC/sub-<SubjectID>_idif.png`
- `QC/sub-<SubjectID>_idif-log.png` (same, log y-axis)
- `QC/sub-<SubjectID>_idif-mask-qc.png` (mask overlay on summed image)
- `QC/sub-<SubjectID>_tac-combined.png` (brain TAC + IDIF, Bq/mL scale)
- `QC/sub-<SubjectID>_tac-combined-suv.png` (brain TAC + IDIF, SUV scale)

#### 4a: Automatic arterial peak detection

- Load the 4D PET.
- Within the loose neck mask, for each of the first ~15 frames (0-5 min), compute the top-1% voxel mean.
- Identify the frame with the highest value: this is the arterial bolus peak. Report the peak time.
- Expected: peak around frame 3 (~35 s).
- Log the per-frame top-1% values for QC.

#### 4b: Create vascular segmentation mask (per-slice approach)

- Sum frames 3-6 (30-80 s, configurable via `IDIF_SUM_WINDOW_START_S` / `IDIF_SUM_WINDOW_END_S`). This creates a "first-pass" summed image where vascular structures are brightest.
- Restrict to voxels inside the loose neck mask.
- **Per-slice selection**: For each z-slice that has neck mask coverage, select the top N hottest voxels (at least `IDIF_MIN_VOXELS_PER_SLICE`, default 8). This ensures vascular coverage along the entire z-axis of the search region.
- Apply 3D connected-component labeling (26-connectivity). Remove clusters smaller than `IDIF_MIN_CLUSTER_SIZE` voxels.
- Log: N slices with mask, total voxels before/after CC filtering, N clusters retained.
- Save the binary mask and the summed first-pass image.
- Create QC figure: 3 panels (axial, coronal, sagittal) showing the summed first-pass image with IDIF mask voxels overlaid.

#### 4c: Extract IDIF TAC

- Apply the vascular mask to all 43 frames.
- For each frame, compute the mean activity within the mask.
- Save TSV and plots as described above.

#### 4d: IDIF QC metrics

Log and include in the TSV provenance header:
- Number of voxels in IDIF mask
- IDIF peak activity and time
- Brain TAC peak activity and time
- Ratio: IDIF_peak / brain_peak (should be >> 1 for valid vascular signal)
- If ratio < 2.0, log a WARNING that the IDIF mask may contain tissue contamination.

#### Manual mask override

If the user has manually drawn or edited an IDIF mask, they can place it at:
`DerivedData/<SubjectID>/masks/sub-<SubjectID>_space-PET_mask-idif-manual.nii.gz`

If this file exists, step 4c should use it instead of the automatically generated mask. Log which mask is being used.

### Step 5: Kinetic modeling

Script: `Scripts/pipeline/s05_kinetics.py`

Performs graphical analysis (Patlak, Logan) and compartment model fitting (1TCM, 2TCM reversible, 2TCM irreversible) using the brain TAC and IDIF.

Inputs:
- `Outputs/sub-<SubjectID>_tac-raw.tsv`
- `Outputs/sub-<SubjectID>_idif.tsv`

Outputs:
- `Outputs/sub-<SubjectID>_patlak.tsv` (Patlak coordinates and fit)
- `Outputs/sub-<SubjectID>_logan.tsv` (Logan coordinates and fit)
- `Outputs/sub-<SubjectID>_kinetics-results.tsv` (all fitted parameters in one table)
- `QC/sub-<SubjectID>_patlak.png`
- `QC/sub-<SubjectID>_logan.png`
- `QC/sub-<SubjectID>_tcm-fit.png` and `_tcm-fit-log.png`
- `QC/sub-<SubjectID>_tcm-residuals.png`

Key functions (reusable by analysis scripts):
- `compute_patlak(Ct, Cp, t_s, t_star_s, t_end_s)`: returns Ki (net influx), V0, R2
- `compute_logan(Ct, Cp, t_s, t_star_s, t_end_s)`: returns VT (distribution volume), R2
- `fit_1tcm(Ct, Cp, t_s, t_end_s, Vb, dt_s)`: returns K1, k2, VT
- `fit_2tcm(Ct, Cp, t_s, t_end_s, Vb, dt_s, reversible)`: returns K1, k2, k3, k4, VT, Ki

All rate constants are in per-minute units. Time inputs are in seconds (converted internally). Multi-start optimization with 3-4 initial guess sets. Frame-duration weighting via sqrt(duration).

## Master pipeline runner

Script: `Scripts/pipeline/run_pipeline.py`

```
usage: run_pipeline.py [-h] --subject SUBJECT [--base-dir BASE_DIR]
                       [--steps STEPS [STEPS ...]]
                       [--force] [--force-from STEP] [--force-all]
                       [--verbose] [--param KEY=VALUE [KEY=VALUE ...]]
```

- `--subject`: subject ID (e.g., SUB001_20260225)
- `--base-dir`: project root (default: parent of Scripts/)
- `--steps`: list of steps to run (default: all). E.g., `--steps s00 s01 s02`
- `--force`: rerun specified steps even if cached
- `--force-from`: rerun from this step onward
- `--force-all`: rerun everything
- `--verbose` / `-v`: enable DEBUG logging
- `--param`: override config parameters, e.g., `--param IDIF_SUM_WINDOW_END_S=60`

On start, log: Python version, package versions (nibabel, numpy, ants, etc.), all CLI arguments, resolved config parameters.

Redirect all stdout/stderr to `Logs/pipeline_<SubjectID>_<YYYYMMDD>_<HHMMSS>.log` while simultaneously printing to terminal.

## Dependencies

Required command-line tools:
- `dcm2niix` (DICOM to NIfTI conversion)
- Python >= 3.9

`requirements.txt`:
```
nibabel>=5.0
pydicom>=2.4
numpy>=1.24
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
antspyx>=0.5
```

## Utility scripts

### fill_outline_mask.py

Fills the largest enclosed area on each z-slice of a hand-drawn outline mask. The user draws brain outlines in a viewer (CT space, 0.32mm resolution); this script fills each slice's interior while preserving smaller unfilled regions (CSF, ventricles). Flags slices with no enclosed area, gaps, or size jumps.

### create_weighted_avg.py

Creates frame-duration-weighted average PET images for specified time windows (e.g., 20-60 min). Selects frames overlapping each window and computes `sum(frame * duration) / sum(durations)` per voxel. Outputs 3D NIfTI files to `DerivedData/<SubjectID>/PET1/`.

### process_manual_wb_mask.py

End-to-end processing of the manually drawn whole-brain mask:
1. Resamples filled mask from CT to PET space (overwrites `mask-brain-whole.nii.gz`)
2. Creates 2D per-slice eroded mask (6 voxels, ~1.9mm at CT resolution)
3. Resamples eroded mask to PET space
4. Extracts brain TAC from eroded mask
5. Plots brain TAC alone and combined with IDIF (SUV, normal + log y-axis)

## Analysis scripts (standalone, not part of the pipeline)

These scripts in `Scripts/` perform post-pipeline exploratory analyses. They import functions from the pipeline modules but are run manually, not via `run_pipeline.py`.

### analysis_brain_idif_ratio.py

Computes brain:IDIF concentration ratio vs time to assess whether the system reaches equilibrium. Uses both whole-brain and eroded masks.

Outputs: `Outputs/sub-..._brain-idif-ratio.tsv`, `QC/sub-..._brain-idif-ratio.png`, `QC/sub-..._brain-idif-ratio-logtime.png`

### analysis_tissue_segmentation.py

MR-based tissue segmentation using 3-class Gaussian Mixture Model on T1w intensities within the eroded brain mask (CT space). Segments into CSF, GM, WM, resamples masks to PET space, extracts per-tissue TACs.

Outputs: tissue masks in `DerivedData/.../masks/`, per-tissue TAC TSVs in `Outputs/`, 5 QC figures in `QC/`

### analysis_blood_calibration.py

Compares venous blood samples (whole blood and plasma) against the IDIF, constructs calibrated plasma input functions, and re-runs kinetic analysis with raw IDIF vs calibrated inputs.

Calibration strategies:
- IDIF scaled to whole blood: scalar correction using late-time IDIF/WB ratio
- IDIF scaled to plasma (time-varying): two-step correction using WB scaling then PCHIP-interpolated Pl/WB ratio
- IDIF scaled to plasma (simple scalar): using late-time IDIF/Pl ratio

Outputs: `Outputs/sub-..._blood-samples.tsv`, `Outputs/sub-..._input-functions.tsv`, `Outputs/sub-..._kinetics-comparison.tsv`, 7 QC figures

## Notes for Claude Code

- Before writing any script, read `raw/Radiochem.csv` to understand its actual column names and format.
- Before converting DICOM, peek at a few DICOM files in `raw/<SubjectID>/PET1/` to understand the header structure (enhanced vs classic DICOM, frame timing tags, units).
- The brain mask and loose neck mask are already in PET space (resliced). Verify by checking that their dimensions match the PET 4D volume.
- PET data are expected to be in Bq/mL (decay-corrected). Verify the DICOM rescale slope/intercept and units tag (0054,1001).
- The IDIF step is exploratory. The mask will need manual QC and possibly refinement. Design the code so the user can easily re-run step 4c with a manually edited mask (see manual mask override above).
- When plotting, use a clean style. Always include axis labels with units.
- Keep frame timing as the single source of truth: load once from `_frames.tsv` and pass to all downstream scripts. Never recompute frame times from DICOM after step 0.
- Never use em-dashes in any code, comments, docstrings, log messages, or output text. Use commas, colons, or semicolons instead.
- Write all scripts so they can be both imported (`from s01_extract_tac import run`) and executed directly (`python s01_extract_tac.py --subject ...`).
- If a step fails, log the error and continue to the next step unless the failure is in a dependency (e.g., step 1 fails, then steps 2 and 3 cannot run). The master runner should handle this dependency chain.
- s01 preferentially uses `mask-brain-whole.nii.gz` for TAC extraction. s03 reads brain volume from s01's provenance header (no assumed volume).
- CT must be cropped before registration (full CT is 644x644x631, entire body). Apply transforms to the full CT as reference to avoid brain clipping.
- s00b produces the cropped CT and MR-to-CT rigid transform. s00c reuses both to avoid redundant computation. If the transform file already exists at the same path, s00c skips the copy (avoids SameFileError).
- The manually drawn whole-brain mask workflow: user draws outlines in CT space, `fill_outline_mask.py` fills interiors, `process_manual_wb_mask.py` resamples to PET space and creates eroded version for TAC.
