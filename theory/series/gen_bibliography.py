#!/usr/bin/env python3
"""Generate references.bib from kb/index/papers.json.

Each paper.key becomes a BibTeX cite-key. Format:

@misc{key,
  author = {Authors as listed},
  title  = {Paper title},
  year   = {YYYY},
  howpublished = {venue},
  eprint = {arXiv id (if URL is arxiv)},
  archivePrefix = {arXiv},
  url    = {url},
  note   = {[KB ref: kb/excerpts/key.md]},
}

Re-run after papers.json changes; idempotent (single-author file).
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from datetime import date

THEORY = Path(__file__).resolve().parents[1]
PAPERS = THEORY / "kb" / "index" / "papers.json"
OUT = THEORY / "series" / "references.bib"


ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})")


def bib_entry(p: dict) -> str:
    key = p["key"]
    title = p.get("title", "Untitled")
    authors = p.get("authors", "Unknown")
    year = p.get("year", "")
    venue = p.get("venue", "")
    url = p.get("url", "")
    excerpt = p.get("excerpts_file")

    # Extract arXiv ID if URL is arxiv-flavored.
    arxiv_id = None
    m = ARXIV_RE.search(url) if url else None
    if m:
        arxiv_id = m.group(1)

    # Pick BibTeX entry type by what we have.
    if "neurips" in venue.lower() or "iclr" in venue.lower() or "icml" in venue.lower() or "acl" in venue.lower() or "emnlp" in venue.lower() or "naacl" in venue.lower():
        entry_type = "inproceedings"
    elif "journal" in venue.lower() or "transactions" in venue.lower() or "computational linguistics" in venue.lower():
        entry_type = "article"
    else:
        entry_type = "misc"

    fields = [
        f"  author = {{{authors}}}",
        f"  title  = {{{title}}}",
    ]
    if year:
        fields.append(f"  year   = {{{year}}}")
    if venue:
        if entry_type == "inproceedings":
            fields.append(f"  booktitle = {{{venue}}}")
        elif entry_type == "article":
            fields.append(f"  journal = {{{venue}}}")
        else:
            fields.append(f"  howpublished = {{{venue}}}")
    if arxiv_id:
        fields.append(f"  eprint = {{{arxiv_id}}}")
        fields.append(f"  archivePrefix = {{arXiv}}")
    if url:
        fields.append(f"  url    = {{{url}}}")
    if excerpt:
        fields.append(f"  note   = {{KB excerpt: {excerpt}}}")

    body = ",\n".join(fields)
    return f"@{entry_type}{{{key},\n{body}\n}}\n"


def main() -> int:
    data = json.loads(PAPERS.read_text())
    papers = sorted(data["papers"], key=lambda p: p["key"])

    out = [
        f"% references.bib — auto-generated from theory/kb/index/papers.json",
        f"% Generator: theory/series/gen_bibliography.py",
        f"% Last regenerated: {date.today()}",
        f"% Papers: {len(papers)}",
        "",
    ]
    for p in papers:
        out.append(bib_entry(p))
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT.relative_to(THEORY)} with {len(papers)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
