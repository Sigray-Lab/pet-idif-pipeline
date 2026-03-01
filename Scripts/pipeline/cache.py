"""Output caching, freshness checks, and provenance header writer."""
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("cache")


def check_outputs_current(inputs: list, outputs: list) -> bool:
    """Return True if all outputs exist and are newer than all inputs."""
    for o in outputs:
        if not Path(o).exists():
            log.debug("Output missing: %s", o)
            return False
    oldest_output = min(Path(o).stat().st_mtime for o in outputs)
    newest_input = max(Path(i).stat().st_mtime for i in inputs)
    return oldest_output > newest_input


def log_skip(step_name: str, outputs: list) -> None:
    """Log that a step is being skipped."""
    log.info(
        "SKIP: outputs exist and are up-to-date. "
        "Use --force to rerun. Output: %s",
        ", ".join(str(o) for o in outputs[:3]),
    )


def md5_file(path: Path) -> str:
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance_header(
    f,
    subject_id: str,
    script_name: str,
    version: str,
    inputs: list,
    parameters: dict = None,
    extra_lines: list = None,
) -> None:
    """Write # -prefixed provenance block to an open file handle."""
    f.write(f"# subject: {subject_id}\n")
    f.write(f"# script: {script_name}\n")
    f.write(f"# version: {version}\n")
    f.write(f"# date: {datetime.now().isoformat()}\n")
    for inp in inputs:
        p = Path(inp)
        if p.exists():
            digest = md5_file(p)
            f.write(f"# input: {p} (md5: {digest})\n")
        else:
            f.write(f"# input: {p} (MISSING)\n")
    if parameters:
        f.write(f"# parameters: {json.dumps(parameters)}\n")
    if extra_lines:
        for line in extra_lines:
            f.write(f"# {line}\n")
