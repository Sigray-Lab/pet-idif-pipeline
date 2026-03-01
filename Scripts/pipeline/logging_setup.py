"""Logging configuration: dual-handler (file + stdout)."""
import logging
from datetime import datetime
from pathlib import Path


def setup_logging(
    subject_id: str, log_dir: Path, verbose: bool = False
) -> logging.Logger:
    """
    Configure root logger with file + stdout handlers.
    Creates log file: Logs/pipeline_<subject_id>_<YYYYMMDD>_<HHMMSS>.log
    Returns the configured root logger.
    Call once at pipeline start.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{subject_id}_{timestamp}.log"

    fmt = "[%(asctime)s.%(msecs)03d] [%(levelname)-5s] [%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Remove any existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    # File handler: always captures DEBUG
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root.addHandler(fh)

    # Console handler: respects verbose flag
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    root.info("Log file: %s", log_file)
    return root
