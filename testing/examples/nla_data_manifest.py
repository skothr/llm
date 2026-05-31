"""Generate / verify the raw-dataset manifest for the NLA research arc.

The arc's raw `.pt` artifacts live (committed, git-LFS) under
`research/arcs/nla-verbalizer/data/`. This script writes a checksummed
`MANIFEST.json` next to them that records, per file: sha256, size, whether
it is a capture-root (needs the model to regenerate) or a derived artifact
(regenerable from other `.pt` by a committed script), the producing script
and command, its `.pt` inputs, what model it needs, and who consumes it.

Two modes:
    python testing/examples/nla_data_manifest.py            # (re)write MANIFEST.json
    python testing/examples/nla_data_manifest.py --check     # verify, exit 1 on drift

The `--check` mode is the drift detector: it recomputes every sha256 and
compares against the committed manifest, catching silent corruption, a
re-capture that wasn't re-committed, or a missing/extra file. Run it in the
arc's audit step alongside `nla_audit_findings.py`.

Self-locating: DATA_DIR resolves relative to this file, so it runs from any
CWD. See `research/ARC_PROCESS.md` § "Raw data is a deliverable".
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = _REPO_ROOT / "research" / "arcs" / "nla-verbalizer" / "data"
MANIFEST = DATA_DIR / "MANIFEST.json"

# Per-artifact provenance. `requires_model` values: none | qwen-base |
# qwen+av | qwen+av+ar | av | ar. `consumers` lists representative figures /
# downstream artifacts / audit (not an exhaustive figure enumeration —
# pairwise_and_hotdims.pt and vocab_atlas.pt feed most of the atlas figures).
META: dict[str, dict[str, Any]] = {
    # ---- capture-roots: require a model load, expensive (CPU-hours) --------
    "aggregate_faithfulness.pt": {
        "class": "capture-root",
        "producing_script": "testing/examples/nla_aggregate_faithfulness.py",
        "inputs": [],
        "requires_model": "qwen+av+ar",
        "consumers": ["geometric_features.pt", "pairwise_and_hotdims.pt", "AUDIT 1/20"],
    },
    "rabbit_haiku_gen_trajectory.pt": {
        "class": "capture-root",
        "producing_script": "testing/examples/nla_faithfulness.py",
        "inputs": [],
        "requires_model": "qwen+av+ar",
        "consumers": ["fig6", "fig14", "pairwise_and_hotdims.pt", "AUDIT 21"],
    },
    "forced_continuation.pt": {
        "class": "capture-root",
        "producing_script": "testing/examples/nla_forced_continuation.py",
        "inputs": [],
        "requires_model": "qwen+av",
        "consumers": ["fig15", "fig16", "geometric_features.pt", "AUDIT 10"],
    },
    "country_concept_vector.pt": {
        "class": "capture-root",
        "producing_script": "testing/examples/nla_country_concept_vector.py",
        "inputs": [],
        "requires_model": "qwen+av",
        "consumers": ["fig13", "pairwise_and_hotdims.pt", "AUDIT 8"],
    },
    "vocab_atlas.pt": {
        "class": "capture-root",
        "producing_script": "testing/examples/nla_vocab_atlas_capture.py",
        "inputs": [],
        "requires_model": "qwen-base",
        "consumers": ["fig19-fig37 (atlas/discriminant figures)", "AUDIT 12/13"],
    },
    "discriminant_stability.pt": {
        "class": "capture-root",
        "producing_script": "testing/examples/nla_discriminant_stability_capture.py",
        "inputs": [],
        "requires_model": "qwen-base",
        "consumers": ["fig28", "AUDIT 14"],
    },
    "interpolation_flipbook.pt": {
        "class": "capture-root",
        "producing_script": "testing/examples/nla_interpolation_flipbook.py",
        "inputs": [],
        "requires_model": "qwen+av+ar",
        "consumers": [
            "fig17",
            "fig18",
            "fig21",
            "dense_interp_near_pivot.pt",
            "AUDIT 11",
        ],
    },
    "mid_seq_vocab_atlas.pt": {
        "class": "capture-root",
        "producing_script": "testing/examples/nla_mid_seq_vocab_atlas_capture.py",
        "inputs": [],
        "requires_model": "qwen-base",
        "consumers": ["mid_seq_compare.pt", "mid_seq_native_compare.pt", "AUDIT 15/16"],
    },
    # ---- derived: regenerable from other .pt by a committed script ---------
    "geometric_features.pt": {
        "class": "derived",
        "producing_script": "testing/examples/nla_geometric_features.py",
        "inputs": [
            "aggregate_faithfulness.pt",
            "rabbit_haiku_gen_trajectory.pt",
            "forced_continuation.pt",
            "country_concept_vector.pt",
        ],
        "requires_model": "none",
        "consumers": ["fig1-fig5"],
    },
    "pairwise_and_hotdims.pt": {
        "class": "derived",
        "producing_script": "testing/examples/nla_pairwise_and_hotdims.py",
        "inputs": [
            "aggregate_faithfulness.pt",
            "rabbit_haiku_gen_trajectory.pt",
            "forced_continuation.pt",
            "country_concept_vector.pt",
        ],
        "requires_model": "none",
        "consumers": ["fig1-fig4, fig7-fig37 (hub artifact)"],
    },
    "sink_removed_atlas.pt": {
        "class": "derived",
        "producing_script": "testing/examples/nla_sink_removed_atlas.py",
        "inputs": ["pairwise_and_hotdims.pt", "geometric_features.pt"],
        "requires_model": "none",
        "consumers": ["fig7", "fig8", "fig9", "fig10", "fig11"],
    },
    "mid_seq_compare.pt": {
        "class": "derived",
        "producing_script": "testing/examples/nla_mid_seq_vocab_atlas_compare.py",
        "inputs": [
            "vocab_atlas.pt",
            "mid_seq_vocab_atlas.pt",
            "pairwise_and_hotdims.pt",
        ],
        "requires_model": "none",
        "consumers": ["fig31", "fig32"],
    },
    "mid_seq_native_compare.pt": {
        "class": "derived",
        "producing_script": "testing/examples/nla_mid_seq_native_compare.py",
        "inputs": [
            "vocab_atlas.pt",
            "mid_seq_vocab_atlas.pt",
            "pairwise_and_hotdims.pt",
        ],
        "requires_model": "none",
        "consumers": ["fig33", "fig34"],
    },
    "concept_arithmetic_atlas.pt": {
        "class": "derived",
        "producing_script": "testing/examples/nla_concept_arithmetic_atlas.py",
        "inputs": ["vocab_atlas.pt"],
        "requires_model": "av",
        "consumers": ["fig35", "AUDIT 17"],
    },
    "dense_interp_near_pivot.pt": {
        "class": "derived",
        "producing_script": "testing/examples/nla_dense_interp_near_pivot.py",
        "inputs": ["interpolation_flipbook.pt"],
        "requires_model": "av",
        "consumers": ["fig36", "fig37", "plateau_attractor_test.pt", "AUDIT 18"],
    },
    "plateau_attractor_test.pt": {
        "class": "derived",
        "producing_script": "testing/examples/nla_plateau_attractor_test.py",
        "inputs": ["dense_interp_near_pivot.pt"],
        "requires_model": "ar",
        "consumers": ["AUDIT 19 (no figure — round-trip validation only)"],
    },
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_entries() -> list[dict[str, Any]]:
    on_disk = sorted(p.name for p in DATA_DIR.glob("*.pt"))
    known = set(META)
    missing = known - set(on_disk)
    extra = set(on_disk) - known
    if missing:
        raise SystemExit(
            f"ERROR: manifest metadata names files absent on disk: {sorted(missing)}"
        )
    if extra:
        raise SystemExit(
            f"ERROR: data dir has .pt files with no metadata: {sorted(extra)}"
        )
    entries: list[dict[str, Any]] = []
    for name in on_disk:
        p = DATA_DIR / name
        m = META[name]
        entries.append(
            {
                "filename": name,
                "sha256": sha256_of(p),
                "size_bytes": p.stat().st_size,
                "class": m["class"],
                "producing_script": m["producing_script"],
                "producing_command": (
                    "PYTHONPATH=$PWD/testing testing/.venv/bin/python "
                    f"{m['producing_script']}"
                ),
                "inputs": m["inputs"],
                "requires_model": m["requires_model"],
                "consumers": m["consumers"],
            }
        )
    return entries


def write_manifest() -> None:
    entries = build_entries()
    doc = {
        "arc": "nla-verbalizer",
        "description": (
            "Raw NLA capture/derived artifacts (layer-20 Qwen2.5-7B-Instruct h "
            "vectors and round-trip products). Committed via git-LFS so figures "
            "and nla_audit_findings.py reproduce from a clean clone."
        ),
        "trust_note": (
            "Loaded with torch.load(weights_only=False) (pickle). Safe here: "
            "locally-generated tensor dumps, not third-party data. Verify sha256 "
            "with `--check` before loading on an untrusted copy."
        ),
        "total_files": len(entries),
        "total_size_bytes": sum(e["size_bytes"] for e in entries),
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(doc, indent=2) + "\n")
    mb = doc["total_size_bytes"] / 1e6
    print(
        f"wrote {MANIFEST.relative_to(_REPO_ROOT)}  ({doc['total_files']} files, {mb:.1f} MB)"
    )


def check_manifest() -> int:
    if not MANIFEST.exists():
        print(f"FAIL: {MANIFEST} does not exist (run without --check to create it)")
        return 1
    doc = json.loads(MANIFEST.read_text())
    recorded = {e["filename"]: e for e in doc["files"]}
    on_disk = {p.name for p in DATA_DIR.glob("*.pt")}
    problems: list[str] = []
    for name in sorted(recorded.keys() | on_disk):
        if name not in on_disk:
            problems.append(f"missing on disk: {name}")
            continue
        if name not in recorded:
            problems.append(f"on disk but not in manifest: {name}")
            continue
        actual = sha256_of(DATA_DIR / name)
        if actual != recorded[name]["sha256"]:
            problems.append(
                f"sha256 drift: {name}\n    manifest={recorded[name]['sha256']}\n    on-disk ={actual}"
            )
    if problems:
        print("MANIFEST CHECK: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"MANIFEST CHECK: OK  ({len(recorded)} files, all sha256 match)")
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        sys.exit(check_manifest())
    write_manifest()
