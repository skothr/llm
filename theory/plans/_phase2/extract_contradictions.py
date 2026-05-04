#!/usr/bin/env python3
"""Extract every [CONTRADICTION] paragraph from kb/notes/ into a cross-area
index at kb/index/contradictions.md.

A [CONTRADICTION] paragraph is the inline tag plus the paragraph that
contains it (whitespace-bounded). Useful as a Phase 4 lens: where do
sources disagree, and which areas surface the deepest open questions?

Idempotent: rewrites the index file each run.
"""
from __future__ import annotations
import re
from pathlib import Path
from collections import defaultdict
from datetime import date

THEORY = Path(__file__).resolve().parents[2]
NOTES = THEORY / "kb" / "notes"
OUT = THEORY / "kb" / "index" / "contradictions.md"


def extract_paragraphs(text: str) -> list[str]:
    """Return paragraphs (whitespace-separated blocks) that contain
    [CONTRADICTION]. Strips frontmatter first."""
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3 :]
    paras = re.split(r"\n\s*\n", text)
    out = []
    for p in paras:
        if "[CONTRADICTION]" in p:
            out.append(p.strip())
    return out


def main() -> int:
    by_area: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    total = 0
    for area_dir in sorted(NOTES.iterdir()):
        if not area_dir.is_dir():
            continue
        area = area_dir.name
        for note_path in sorted(area_dir.glob("*.md")):
            paras = extract_paragraphs(note_path.read_text())
            if paras:
                by_area[area].append((note_path.stem, paras))
                total += len(paras)

    lines = [
        "# Contradictions surface",
        "",
        f"_Generated {date.today()} by `theory/plans/_phase2/extract_contradictions.py` from "
        f"every `[CONTRADICTION]` paragraph across `kb/notes/`. {total} contradictions across "
        f"{sum(len(v) for v in by_area.values())} notes in {len(by_area)} areas._",
        "",
        "Each entry below is the verbatim paragraph from the note, preceded by a "
        "`<area>/<topic>` header. Use this as a single lens for: where do primary "
        "sources disagree? Which areas have the deepest live open questions? Which "
        "contradictions appear in multiple areas (suggesting a cross-area research thread)?",
        "",
        "## Density by area",
        "",
        "| Area | Notes w/ contradictions | Total contradictions |",
        "|------|------------------------:|--------------------:|",
    ]
    area_totals = []
    for area in sorted(by_area):
        n_notes = len(by_area[area])
        n_contra = sum(len(p) for _, p in by_area[area])
        area_totals.append((area, n_notes, n_contra))
    for area, n_notes, n_contra in sorted(area_totals, key=lambda x: -x[2]):
        lines.append(f"| {area} | {n_notes} | {n_contra} |")
    lines.append("")
    lines.append("## By note (top contention)")
    lines.append("")
    note_totals = [
        (f"{area}/{topic}", len(p))
        for area, items in by_area.items()
        for topic, p in items
    ]
    for note, n in sorted(note_totals, key=lambda x: -x[1])[:10]:
        lines.append(f"- `{note}` — {n}")
    lines.append("")
    for area in sorted(by_area):
        lines.append(f"## {area}")
        lines.append("")
        for topic, paras in by_area[area]:
            lines.append(f"### {area}/{topic}")
            lines.append("")
            lines.append(f"_Source: `kb/notes/{area}/{topic}.md`_")
            lines.append("")
            for p in paras:
                lines.append(p)
                lines.append("")
        lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT.relative_to(THEORY)} ({total} contradictions across {len(by_area)} areas).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
