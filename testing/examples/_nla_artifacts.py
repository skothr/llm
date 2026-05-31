"""Shared artifact-path resolver for the NLA arc example scripts.

Two artifact locations:
  * CACHE — testing/.cache/nla_artifacts/ — the gitignored working cache the
    capture/derive scripts write to during local development.
  * DATA  — research/arcs/nla-verbalizer/data/ — the committed git-LFS copy a
    clean clone has (see research/ARC_PROCESS.md § "Raw data is a deliverable").

Reads prefer the cache (so a fresh local re-capture is picked up immediately)
and fall back to the committed copy (so figures re-render and the audit replays
from a clean clone with no manual copy-back). Writes always go to the cache;
promote a new/changed artifact to DATA with `nla_data_manifest.py` when
committing.

Self-locating from this file, so callers work from any CWD. Mirrors the
resolution `nla_audit_findings.py` does at the directory level.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = _REPO_ROOT / "testing" / ".cache" / "nla_artifacts"
DATA = _REPO_ROOT / "research" / "arcs" / "nla-verbalizer" / "data"
FIGURES = (
    _REPO_ROOT / "research" / "arcs" / "nla-verbalizer" / "observations" / "figures"
)


def find_artifact(name: str) -> Path | None:
    """Resolve an input artifact: live cache first, then the committed copy.
    Returns None if neither has it — use for resume / optional reads guarded
    by an `if`."""
    cached = CACHE / name
    if cached.exists():
        return cached
    committed = DATA / name
    if committed.exists():
        return committed
    return None


def read_artifact(name: str) -> Path:
    """Resolve an input artifact for an unconditional load; raise if missing."""
    p = find_artifact(name)
    if p is None:
        raise FileNotFoundError(
            f"{name!r} not found in {CACHE} or {DATA}. Run the capture step, "
            f"or `git lfs pull` to fetch the committed copy."
        )
    return p


def write_artifact(name: str) -> Path:
    """Resolve an output artifact path in the working cache (created if needed).
    Writes never target the committed DATA dir — promote with
    nla_data_manifest.py when committing."""
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name
