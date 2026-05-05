#!/usr/bin/env python3
"""Phase 2 self-review lint.

Checks:
1. papers.json is valid; every paper has required fields.
2. Every excerpts_file in papers.json points to a real file (or null).
3. Every kb/notes/<area>/<topic>.md exists for every leaf in topics.md.
4. Every paper-key cited inline in a note ([key §X] or [key]) corresponds to a key in papers.json.
5. Every kb/excerpts/<key>#anchor citation in a note resolves to a real excerpt file.
6. Notes have YAML frontmatter with required keys.
7. Count [INTUITION], [ANALOGY], [CONTRADICTION] tags per note.

Run from theory/.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict

THEORY = Path(__file__).resolve().parents[2]
PAPERS = THEORY / "kb" / "index" / "papers.json"
TOPICS = THEORY / "kb" / "index" / "topics.md"
NOTES = THEORY / "kb" / "notes"
EXCERPTS = THEORY / "kb" / "excerpts"

errors: list[str] = []
warnings: list[str] = []
stats: dict = defaultdict(int)


def check_papers_json():
    data = json.loads(PAPERS.read_text())
    keys = set()
    for p in data["papers"]:
        if "key" not in p:
            errors.append(f"papers.json: paper missing key: {p.get('title','?')}")
            continue
        if p["key"] in keys:
            errors.append(f"papers.json: duplicate key {p['key']!r}")
        keys.add(p["key"])
        if p.get("excerpts_file"):
            ep = THEORY / p["excerpts_file"]
            if not ep.exists():
                errors.append(f"papers.json: {p['key']!r} excerpts_file points to missing file {p['excerpts_file']}")
        for required in ("title", "year"):
            if not p.get(required):
                warnings.append(f"papers.json: {p['key']!r} missing/empty {required}")
    stats["papers_total"] = len(data["papers"])
    stats["papers_with_excerpts"] = sum(1 for p in data["papers"] if p.get("excerpts_file"))
    stats["papers_with_notes"] = sum(1 for p in data["papers"] if p.get("notes_referenced_by"))
    return data, keys


def check_topics():
    text = TOPICS.read_text()
    by_status = defaultdict(int)
    leaves = []
    for line in text.splitlines():
        m = re.match(
            r"^\|\s*(?P<topic>[a-z0-9-]+)\s*\|\s*(?P<status>\w+)\s*\|\s*`(?P<path>kb/notes/[^`]+)`\s*\|\s*$",
            line,
        )
        if m:
            topic = m.group("topic")
            status = m.group("status")
            path = m.group("path")
            leaves.append((topic, status, path))
            by_status[status] += 1
            note_path = THEORY / path
            if not note_path.exists():
                errors.append(f"topics.md: leaf {topic!r} note path {path!r} does not exist")
    stats["leaves_by_status"] = dict(by_status)
    stats["leaves_total"] = len(leaves)
    return leaves


PAPER_CITE_RE = re.compile(r"\[([a-z0-9-]+(?:\.\d+)?(?:/[a-z0-9-]+)*)\s*§")
# Match excerpt anchors anywhere in citation brackets (hybrid form
# `[paper-key §X; kb/excerpts/key#anchor]` is dominant in practice).
EXCERPT_CITE_RE = re.compile(r"(kb/excerpts/[a-z0-9-]+)#([a-z0-9-]+)")
TAG_RE = re.compile(r"\[(INTUITION|ANALOGY|CONTRADICTION|FORUM-SIGNAL|SPECULATION)\]")


def check_notes(paper_keys: set):
    paper_cite_keys: set[str] = set()
    excerpt_cites: set[str] = set()
    note_count = 0
    drafts = 0
    stubs = 0
    tag_total = defaultdict(int)
    for area_dir in NOTES.iterdir():
        if not area_dir.is_dir():
            continue
        for note_path in area_dir.glob("*.md"):
            note_count += 1
            text = note_path.read_text()
            if not text.startswith("---"):
                errors.append(f"{note_path.relative_to(THEORY)}: missing YAML frontmatter")
                continue
            end = text.find("---", 3)
            if end == -1:
                errors.append(f"{note_path.relative_to(THEORY)}: malformed frontmatter")
                continue
            fm = text[3:end]
            status_m = re.search(r"^status:\s*(\S+)", fm, re.MULTILINE)
            if not status_m:
                warnings.append(f"{note_path.relative_to(THEORY)}: no status in frontmatter")
            else:
                if status_m.group(1) == "draft":
                    drafts += 1
                elif status_m.group(1) == "stub":
                    stubs += 1
            body = text[end+3:]

            # Paper-key citations like [key §X ...]
            for m in PAPER_CITE_RE.finditer(body):
                key = m.group(1)
                paper_cite_keys.add(key)
                if key not in paper_keys:
                    warnings.append(
                        f"{note_path.relative_to(THEORY)}: cites unknown paper-key {key!r}"
                    )

            # kb/excerpts/<key>#anchor citations
            for m in EXCERPT_CITE_RE.finditer(body):
                rel = m.group(1)
                anchor = m.group(2)
                excerpt_cites.add(f"{rel}#{anchor}")
                ep = THEORY / (rel + ".md")
                if not ep.exists():
                    warnings.append(
                        f"{note_path.relative_to(THEORY)}: cites missing excerpt file {rel}.md"
                    )

            # Tags
            for m in TAG_RE.finditer(body):
                tag_total[m.group(1)] += 1

    stats["notes_total"] = note_count
    stats["notes_draft"] = drafts
    stats["notes_stub"] = stubs
    stats["paper_keys_cited"] = len(paper_cite_keys)
    stats["excerpt_anchors_cited"] = len(excerpt_cites)
    stats["tag_counts"] = dict(tag_total)


def main():
    data, paper_keys = check_papers_json()
    leaves = check_topics()
    check_notes(paper_keys)

    print("=" * 60)
    print("PHASE 2 LINT REPORT")
    print("=" * 60)
    print(f"\nPapers in index:    {stats['papers_total']}")
    print(f"  with excerpts:    {stats['papers_with_excerpts']}")
    print(f"  with note refs:   {stats['papers_with_notes']}")
    print(f"\nTopics in topics.md: {stats['leaves_total']}")
    print(f"  by status: {stats['leaves_by_status']}")
    print(f"\nNotes written:       {stats['notes_total']}")
    print(f"  draft:             {stats['notes_draft']}")
    print(f"  stub:              {stats['notes_stub']}")
    print(f"\nDistinct paper-keys cited inline: {stats['paper_keys_cited']}")
    print(f"Distinct excerpt anchors cited:   {stats['excerpt_anchors_cited']}")
    print(f"Tag counts: {stats['tag_counts']}")

    print(f"\n--- ERRORS ({len(errors)}) ---")
    for e in errors[:20]:
        print(f"  ! {e}")
    if len(errors) > 20:
        print(f"  ...and {len(errors) - 20} more.")

    print(f"\n--- WARNINGS ({len(warnings)}) ---")
    for w in warnings[:30]:
        print(f"  · {w}")
    if len(warnings) > 30:
        print(f"  ...and {len(warnings) - 30} more.")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
