"""Pipeline configuration: all tunable parameters with defaults."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class PipelineConfig:
    """All tunable parameters for the dynamic PET imaging pipeline."""

    # --- Directories (resolved at runtime) ---
    base_dir: Path = Path(".")

    # --- Pipeline version ---
    PIPELINE_VERSION: str = "1.0.0"

    # --- Frame protocol (expected) ---
    EXPECTED_N_FRAMES: int = 43
    EXPECTED_TOTAL_DURATION_S: int = 9000  # 150 min

    # --- IDIF parameters ---
    IDIF_PEAK_SEARCH_FRAMES: int = 15
    IDIF_SUM_WINDOW_START_S: float = 30.0   # frames 3-6 (30-80s)
    IDIF_SUM_WINDOW_END_S: float = 80.0
    IDIF_MIN_VOXELS_PER_SLICE: int = 8      # minimum voxels per z-slice
    IDIF_TOP_PERCENTILE: float = 99.5       # starting percentile for per-slice threshold
    IDIF_MIN_CLUSTER_SIZE: int = 3

    # --- Patlak graphical analysis ---
    PATLAK_T_STAR_MIN: float = 15.0        # start of linear phase (minutes)
    PATLAK_T_END_MIN: float = 80.0         # end of analysis window (minutes)

    # --- Compartment models (1TCM, 2TCM) ---
    TCM_T_END_MIN: float = 150.0           # end of fitting window (minutes), full scan
    TCM_VB: float = 0.05                   # fixed blood volume fraction (5%)
    TCM_DT_S: float = 1.0                  # interpolation time step for convolution (seconds)

    # --- SUV and %ID sanity checks ---
    INJECTED_DOSE_RANGE_MBQ: Tuple[float, float] = (10.0, 200.0)
    BODY_WEIGHT_RANGE_KG: Tuple[float, float] = (1.0, 150.0)

    # --- MR brain extraction (CT-guided) ---
    MR_BRAIN_VOLUME_RANGE_ML: Tuple[float, float] = (20.0, 2000.0)
    CT_BONE_THRESHOLD_HU: int = 200       # Hounsfield units for bone
    CT_CLOSE_ITER_1: int = 5              # morphological closing stage 1 (seals sutures)
    CT_CLOSE_ITER_2: int = 10             # morphological closing stage 2 (seals larger gaps)
    CT_INFERIOR_CAP_MM: int = 35          # mm below bone centroid for foramen magnum cap

    # --- CT cropping for coregistration (voxel indices, inclusive start, exclusive end) ---
    # CT shape is (644, 644, 631) with dim0=LR, dim1=AP, dim2=Z
    CT_CROP_LR: Tuple[int, int] = (167, 499)
    CT_CROP_AP: Tuple[int, int] = (137, 567)
    CT_CROP_Z: Tuple[int, int] = (147, 480)

    # --- Plotting ---
    FIGURE_DPI: int = 300
    FIGURE_STYLE: str = "seaborn-v0_8-whitegrid"

    # --- Path helpers ---
    def raw_dir(self, subject_id: str) -> Path:
        return self.base_dir / "raw" / subject_id

    def derived_dir(self, subject_id: str) -> Path:
        return self.base_dir / "DerivedData" / subject_id

    def outputs_dir(self) -> Path:
        return self.base_dir / "Outputs"

    def figures_dir(self) -> Path:
        return self.base_dir / "QC"

    def logs_dir(self) -> Path:
        return self.base_dir / "Logs"

    def radiochem_path(self) -> Path:
        return self.base_dir / "raw" / "Radiochem.csv"

    # --- CLI override ---
    def override_from_cli(self, overrides: list) -> None:
        """Parse KEY=VALUE strings and set attributes with type coercion."""
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Invalid param format: {item!r}. Expected KEY=VALUE.")
            key, val_str = item.split("=", 1)
            if not hasattr(self, key):
                raise ValueError(f"Unknown config parameter: {key!r}")
            current = getattr(self, key)
            if isinstance(current, bool):
                setattr(self, key, val_str.lower() in ("true", "1", "yes"))
            elif isinstance(current, int):
                setattr(self, key, int(val_str))
            elif isinstance(current, float):
                setattr(self, key, float(val_str))
            elif isinstance(current, tuple):
                # Accept "(40, 70)" or "40,70"
                cleaned = val_str.strip("() ")
                parts = [p.strip() for p in cleaned.split(",")]
                if all(isinstance(current[0], float) for _ in parts):
                    setattr(self, key, tuple(float(p) for p in parts))
                else:
                    setattr(self, key, tuple(int(p) for p in parts))
            elif isinstance(current, str):
                setattr(self, key, val_str)

    def as_dict(self) -> dict:
        """Return all non-path config values as a dict for logging."""
        out = {}
        for k, v in self.__dict__.items():
            if k == "base_dir":
                out[k] = str(v)
            elif not callable(v):
                out[k] = v
        return out
