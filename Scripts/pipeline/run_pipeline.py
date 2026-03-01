#!/usr/bin/env python3
"""Master pipeline runner for dynamic PET imaging."""
import argparse
import logging
import platform
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

# Ensure the Scripts directory is on the path so 'pipeline' is importable
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from pipeline.config import PipelineConfig
from pipeline.logging_setup import setup_logging
from pipeline import (
    s00_dcm2nii,
    s00b_segment_mr,
    s00c_coregister,
    s01_extract_tac,
    s02_suv_tac,
    s03_percent_id,
    s04_idif,
    s05_kinetics,
)

log = logging.getLogger("run_pipeline")

# Step definitions with dependency graph
STEPS = OrderedDict([
    ("s00",  {"func": s00_dcm2nii.run,      "deps": []}),
    ("s00b", {"func": s00b_segment_mr.run,   "deps": ["s00"]}),
    ("s00c", {"func": s00c_coregister.run,   "deps": ["s00", "s00b"]}),
    ("s01",  {"func": s01_extract_tac.run,   "deps": ["s00"]}),
    ("s02",  {"func": s02_suv_tac.run,       "deps": ["s01"]}),
    ("s03",  {"func": s03_percent_id.run,    "deps": ["s01"]}),
    ("s04",  {"func": s04_idif.run,          "deps": ["s00"]}),
    ("s05",  {"func": s05_kinetics.run,    "deps": ["s01", "s04"]}),
])

ALL_STEP_NAMES = list(STEPS.keys())


def _log_environment(cfg: PipelineConfig, args: argparse.Namespace) -> None:
    """Log Python version, package versions, and CLI arguments."""
    log.info("=" * 60)
    log.info("Dynamic PET Pipeline v%s", cfg.PIPELINE_VERSION)
    log.info("=" * 60)
    log.info("Python: %s", sys.version.replace("\n", " "))
    log.info("Platform: %s", platform.platform())

    # Package versions
    for pkg_name in ["nibabel", "pydicom", "numpy", "scipy", "pandas", "matplotlib", "ants"]:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, "__version__", "unknown")
            log.info("  %s: %s", pkg_name, ver)
        except ImportError:
            log.warning("  %s: NOT INSTALLED", pkg_name)

    # dcm2niix
    dcm2niix_path = shutil.which("dcm2niix")
    log.info("  dcm2niix: %s", dcm2niix_path or "NOT FOUND")

    log.info("CLI args: %s", vars(args))
    log.info("Config: %s", cfg.as_dict())
    log.info("Base dir: %s", cfg.base_dir.resolve())


def _resolve_steps(args: argparse.Namespace) -> list:
    """Determine which steps to run based on CLI arguments."""
    if args.steps:
        # Validate step names
        for s in args.steps:
            if s not in STEPS:
                raise ValueError(f"Unknown step: {s!r}. Valid: {ALL_STEP_NAMES}")
        return args.steps

    return ALL_STEP_NAMES


def _resolve_force(args: argparse.Namespace, steps_to_run: list) -> set:
    """Determine which steps should be force-rerun."""
    if args.force_all:
        return set(ALL_STEP_NAMES)

    if args.force_from:
        if args.force_from not in STEPS:
            raise ValueError(f"Unknown step for --force-from: {args.force_from!r}")
        idx = ALL_STEP_NAMES.index(args.force_from)
        return set(ALL_STEP_NAMES[idx:])

    if args.force:
        return set(steps_to_run)

    return set()


def _nuke_derived(subject_id: str, cfg: PipelineConfig) -> None:
    """Delete all derived outputs for force-all."""
    derived = cfg.derived_dir(subject_id)
    if derived.exists():
        log.warning("FORCE-ALL: deleting %s", derived)
        shutil.rmtree(derived)

    # Also clean Outputs
    for pattern in [f"sub-{subject_id}_*"]:
        for f in cfg.outputs_dir().glob(pattern):
            log.warning("FORCE-ALL: deleting %s", f)
            f.unlink()
        for f in cfg.figures_dir().glob(pattern):
            log.warning("FORCE-ALL: deleting %s", f)
            f.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Dynamic PET Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subject", required=True,
                        help="Subject ID (e.g., SUB001_20260225)")
    parser.add_argument("--base-dir", default=None,
                        help="Project root directory (default: auto-detect)")
    parser.add_argument("--steps", nargs="+", default=None,
                        help=f"Steps to run (default: all). Options: {ALL_STEP_NAMES}")
    parser.add_argument("--force", action="store_true",
                        help="Force rerun of specified steps")
    parser.add_argument("--force-from", default=None, metavar="STEP",
                        help="Force rerun from this step onward")
    parser.add_argument("--force-all", action="store_true",
                        help="Delete all derived outputs and rerun everything")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--param", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Override config parameters")
    args = parser.parse_args()

    # Resolve base directory
    if args.base_dir:
        base = Path(args.base_dir).resolve()
    else:
        # Auto-detect: go up from Scripts/pipeline/ to project root
        base = SCRIPT_DIR.parent.parent.resolve()

    cfg = PipelineConfig(base_dir=base)

    # Apply CLI param overrides
    if args.param:
        cfg.override_from_cli(args.param)

    # Setup logging
    setup_logging(args.subject, cfg.logs_dir(), verbose=args.verbose)
    _log_environment(cfg, args)

    # Resolve steps and force flags
    steps_to_run = _resolve_steps(args)
    force_steps = _resolve_force(args, steps_to_run)

    log.info("Steps to run: %s", steps_to_run)
    log.info("Force rerun: %s", sorted(force_steps) if force_steps else "none")

    # Nuke if force-all
    if args.force_all:
        _nuke_derived(args.subject, cfg)

    # Create output directories
    cfg.outputs_dir().mkdir(parents=True, exist_ok=True)
    cfg.figures_dir().mkdir(parents=True, exist_ok=True)

    # Track which steps completed successfully
    completed = set()
    failed = set()
    t_total = time.time()

    for step_name in steps_to_run:
        step_info = STEPS[step_name]

        # Check dependencies
        unmet = [d for d in step_info["deps"] if d in failed]
        if unmet:
            log.error(
                "SKIP %s: dependency failed: %s",
                step_name, ", ".join(unmet),
            )
            failed.add(step_name)
            continue

        # Check if dependency was requested but not yet run in this session
        # (it might have cached outputs from a prior run, which is fine)
        missing_deps = [d for d in step_info["deps"]
                        if d not in completed and d not in steps_to_run]
        # That is OK: the step will check its own inputs exist

        force = step_name in force_steps

        try:
            step_info["func"](args.subject, cfg, force=force)
            completed.add(step_name)
        except Exception:
            log.exception("FAILED: step %s", step_name)
            failed.add(step_name)

    # Final summary
    elapsed_total = time.time() - t_total
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("Completed: %s", sorted(completed))
    if failed:
        log.error("Failed: %s", sorted(failed))
    log.info("Total time: %.1f s (%.1f min)", elapsed_total, elapsed_total / 60)
    log.info("=" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
