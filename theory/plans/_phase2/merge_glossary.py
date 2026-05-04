#!/usr/bin/env python3
"""Append per-area glossary fragments into the master kb/glossary.md.

Each fragment at kb/index/_phase2-additions/<area>-glossary.md is wrapped
under a per-area H3 inside one master H2 section. Idempotent: re-running
with the marker present skips work.

Run from theory/ working directory.
"""
from __future__ import annotations

from pathlib import Path
from datetime import date

THEORY = Path(__file__).resolve().parents[2]
GLOSSARY = THEORY / "kb" / "glossary.md"
ADD_DIR = THEORY / "kb" / "index" / "_phase2-additions"

AREAS = [
    "alignment",
    "architecture",
    "evaluation",
    "inference",
    "interpretability",
    "post-training",
    "reasoning",
    "scaling",
    "training",
]

MARKER = "## Phase 2 area additions"


def main() -> int:
    fragments: dict[str, str] = {}
    for area in AREAS:
        p = ADD_DIR / f"{area}-glossary.md"
        if p.exists():
            fragments[area] = p.read_text().rstrip()
    if not fragments:
        print("No glossary fragments found; nothing to do.")
        return 0

    current = GLOSSARY.read_text()
    if MARKER in current:
        print(f"Glossary already contains marker {MARKER!r}; skipping (idempotent).")
        return 0

    out = [current.rstrip()]
    out.append("")
    out.append("")
    out.append(MARKER)
    out.append("")
    out.append(
        f"_Per-area glossary fragments produced by Phase 2 area subagents on "
        f"{date.today()}; merged here. Each subsection is the verbatim fragment "
        f"from `theory/kb/index/_phase2-additions/<area>-glossary.md`._"
    )
    for area in AREAS:
        if area not in fragments:
            continue
        out.append("")
        out.append(f"### {area}")
        out.append("")
        out.append(fragments[area])
    out.append("")  # trailing newline
    GLOSSARY.write_text("\n".join(out))
    total_terms = sum(f.count("\n- **") for f in fragments.values())
    print(f"Merged {len(fragments)} area fragments (~{total_terms} terms) into glossary.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
