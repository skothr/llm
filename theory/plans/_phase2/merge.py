#!/usr/bin/env python3
"""Merge Phase 2 area-subagent additions into the master papers.json.

For each `kb/index/_phase2-additions/<area>.json`:
  - Append new_papers (skipping any keys already in papers.json — log dups).
  - Apply existing_paper_updates: set_excerpts_file, set_local_file,
    add_notes_referenced_by (de-duplicated append).

Known canonicalizations (pre-existing index keys vs. new keys for the same paper):
  nsa-2025  ->  yuan2025
  apollo-scheming-2024  ->  meinke2024-apollo-scheming

After dedup, bump last_updated and write the merged file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

THEORY = Path(__file__).resolve().parents[2]  # .../theory/
INDEX = THEORY / "kb" / "index"
ADDITIONS_DIR = INDEX / "_phase2-additions"
PAPERS_JSON = INDEX / "papers.json"

# Canonicalize aliases: alias -> canonical
ALIASES = {
    "nsa-2025": "yuan2025",
    "apollo-scheming-2024": "meinke2024-apollo-scheming",
}


def canon(key: str) -> str:
    return ALIASES.get(key, key)


def main() -> int:
    data = json.loads(PAPERS_JSON.read_text())
    papers_by_key: dict[str, dict] = {p["key"]: p for p in data["papers"]}
    started_with = len(papers_by_key)

    # Pass 0: collapse aliases into canonical keys (move references and drop alias rows).
    for alias, canonical in ALIASES.items():
        if alias in papers_by_key:
            alias_paper = papers_by_key[alias]
            if canonical in papers_by_key:
                # Merge alias's notes_referenced_by into canonical, drop alias.
                tgt = papers_by_key[canonical]
                tgt_refs = list(tgt.get("notes_referenced_by") or [])
                for r in alias_paper.get("notes_referenced_by") or []:
                    if r not in tgt_refs:
                        tgt_refs.append(r)
                tgt["notes_referenced_by"] = tgt_refs
                # Don't overwrite the canonical's excerpts_file / local_file —
                # canonical wins. But fold in topics.
                for t in alias_paper.get("topics") or []:
                    if t not in (tgt.get("topics") or []):
                        tgt.setdefault("topics", []).append(t)
                print(f"  [DEDUP] folded {alias!r} into {canonical!r}")
            else:
                # Canonical doesn't exist yet — rename alias to canonical.
                alias_paper["key"] = canonical
                papers_by_key[canonical] = alias_paper
                print(f"  [DEDUP] renamed {alias!r} -> {canonical!r}")
            del papers_by_key[alias]

    # Pass 1: ingest new_papers + apply existing_paper_updates from each area.
    new_added = 0
    new_skipped = 0
    updates_applied = 0
    log: list[str] = []

    for jf in sorted(ADDITIONS_DIR.glob("*.json")):
        area_data = json.loads(jf.read_text())
        area = area_data.get("area") or jf.stem
        log.append(f"\n--- {area} ({jf.name}) ---")

        for new in area_data.get("new_papers") or []:
            key = canon(new["key"])
            new["key"] = key
            if key in papers_by_key:
                # Update — fold notes_referenced_by, prefer existing for other fields
                existing = papers_by_key[key]
                refs = list(existing.get("notes_referenced_by") or [])
                for r in new.get("notes_referenced_by") or []:
                    if r not in refs:
                        refs.append(r)
                existing["notes_referenced_by"] = refs
                # If existing has no excerpts file but new does, take new's.
                if not existing.get("excerpts_file") and new.get("excerpts_file"):
                    existing["excerpts_file"] = new["excerpts_file"]
                # Same for local_file
                if not existing.get("local_file") and new.get("local_file"):
                    existing["local_file"] = new["local_file"]
                new_skipped += 1
                log.append(f"  [DUP-NEW] {key!r} already present; merged refs only")
            else:
                # Fresh insert. Normalize fields.
                paper = {
                    "key": key,
                    "title": new.get("title", ""),
                    "authors": new.get("authors", ""),
                    "year": new.get("year", 0),
                    "venue": new.get("venue", ""),
                    "url": new.get("url", ""),
                    "local_file": new.get("local_file"),
                    "excerpts_file": new.get("excerpts_file"),
                    "notes_referenced_by": list(new.get("notes_referenced_by") or []),
                    "topics": list(new.get("topics") or []),
                    "category": new.get("category", "recent"),
                    "summary": new.get("summary", ""),
                }
                papers_by_key[key] = paper
                new_added += 1
                log.append(f"  [NEW] {key}")

        for upd in area_data.get("existing_paper_updates") or []:
            key = canon(upd["key"])
            tgt = papers_by_key.get(key)
            if not tgt:
                log.append(f"  [MISS] update target not found: {upd['key']}")
                continue
            if "set_excerpts_file" in upd:
                tgt["excerpts_file"] = upd["set_excerpts_file"]
            if "set_local_file" in upd:
                tgt["local_file"] = upd["set_local_file"]
            if "add_notes_referenced_by" in upd:
                refs = list(tgt.get("notes_referenced_by") or [])
                for r in upd["add_notes_referenced_by"] or []:
                    r_canon = r  # note refs aren't aliased
                    if r_canon not in refs:
                        refs.append(r_canon)
                tgt["notes_referenced_by"] = refs
            updates_applied += 1
            log.append(f"  [UPD] {key}")

    # Rebuild papers list, preserving stable order: existing-paper order first
    # (in original file order), then new additions in insertion order.
    original_order = [p["key"] for p in data["papers"] if canon(p["key"]) in papers_by_key]
    seen = set()
    final_keys: list[str] = []
    for k in original_order:
        ck = canon(k)
        if ck in papers_by_key and ck not in seen:
            final_keys.append(ck)
            seen.add(ck)
    # Append any new keys that weren't in the original list
    for k in papers_by_key.keys():
        if k not in seen:
            final_keys.append(k)
            seen.add(k)
    data["papers"] = [papers_by_key[k] for k in final_keys]

    data["last_updated"] = "2026-05-04"

    PAPERS_JSON.write_text(json.dumps(data, indent=2) + "\n")

    print("\n".join(log))
    print()
    print(f"Started with: {started_with} papers")
    print(f"Now:          {len(data['papers'])} papers")
    print(f"  added:      {new_added}")
    print(f"  dup-new:    {new_skipped}")
    print(f"  updates:    {updates_applied}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
