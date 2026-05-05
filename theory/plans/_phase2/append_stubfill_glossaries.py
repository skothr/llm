#!/usr/bin/env python3
"""Append stubfill-* glossary fragments into the master kb/glossary.md.

Idempotent on the marker. The architecture stub-fill subagent stalled
before writing its glossary fragment, so only training+post+inference
and interp+eval+alignment fragments are merged.
"""
from __future__ import annotations
from pathlib import Path

THEORY = Path(__file__).resolve().parents[2]
GLOSSARY = THEORY / "kb" / "glossary.md"
ADD = THEORY / "kb" / "index" / "_phase2-additions"
MARKER = "## Phase 2 stub-fill additions"


def body(text: str) -> str:
    idx = text.find("\n## ")
    return text[idx:].lstrip("\n") if idx != -1 else text


def main() -> int:
    text = GLOSSARY.read_text()
    if MARKER in text:
        print(f"glossary already contains {MARKER}; idempotent skip")
        return 0
    train = (ADD / "stubfill-train-glossary.md").read_text()
    iea = (ADD / "stubfill-iea-glossary.md").read_text()
    out = [
        text.rstrip(),
        "",
        "",
        MARKER,
        "",
        "_Glossary fragments produced 2026-05-04 by the two stub-fill "
        "subagents (training+post+inference; interpretability+evaluation+"
        "alignment). The third (architecture) batch stalled before its "
        "glossary fragment was written; arch terms remain inline in "
        "their respective notes._",
        "",
        "### training+post-training+inference",
        "",
        body(train),
        "",
        "### interpretability+evaluation+alignment",
        "",
        body(iea),
        "",
    ]
    GLOSSARY.write_text("\n".join(out))
    n = len(GLOSSARY.read_text().splitlines())
    print(f"appended stubfill glossaries; glossary.md now {n} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
