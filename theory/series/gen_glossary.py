#!/usr/bin/env python3
"""Generate theory/series/glossary-terms.json from theory/kb/glossary.md.

The JSON drives the body-text wrapper script (mark_glossary_terms.py)
and the per-paper glossary section emitter. Format mirrors the existing
gen_bibliography.py / references.bib relationship.

Re-run after glossary.md changes; idempotent.
"""
from __future__ import annotations
import json
import re
from collections.abc import Sequence
from pathlib import Path

THEORY = Path(__file__).resolve().parents[1]
GLOSSARY_MD = THEORY / "kb" / "glossary.md"
OUT = THEORY / "series" / "glossary-terms.json"

BULLET_RE = re.compile(r"^- \*\*(?P<term>.+?)\*\*\s*[—-]\s*(?P<def>.+)$")
PAREN_RE = re.compile(r"^(?P<head>[^(]+?)\s*\((?P<paren>.+?)\)\s*$")
KB_CITE_RE = re.compile(r"`\[([^\]`]+)\]`\s*$")


def _looks_like_alias(s: str) -> bool:
    """A parenthetical is an alias if it starts with a capital letter or is acronym-shaped."""
    if not s:
        return False
    if s[0].isupper():
        return True
    # Acronym shape: all letters uppercase, possibly with hyphens / digits.
    if re.fullmatch(r"[A-Z][A-Z0-9-]*", s):
        return True
    return False


def _split_def_and_cite(defn: str) -> tuple[str, str | None]:
    """Strip a trailing `[paper-key §X]` cite, returning (clean_def, cite_or_None)."""
    m = KB_CITE_RE.search(defn)
    if not m:
        return defn, None
    return defn[: m.start()].rstrip(), m.group(1).strip()


def parse_entries(lines: Sequence[str]) -> list[dict]:
    """Parse the bullet entries from a glossary.md line iterable.

    Returns one record per entry: {primary_form, aliases, full_def, kb_cite}.
    More fields (key, short_def, case_strict) are added by later passes.
    """
    entries: list[dict] = []
    for line in lines:
        m = BULLET_RE.match(line)
        if not m:
            continue
        term = m.group("term").strip()
        defn_raw = m.group("def").strip()
        primary = term
        aliases: list[str] = []
        pm = PAREN_RE.match(term)
        if pm:
            head = pm.group("head").strip()
            paren = pm.group("paren").strip()
            if _looks_like_alias(paren):
                primary = head
                aliases = [paren]
        clean_def, kb_cite = _split_def_and_cite(defn_raw)
        entries.append({
            "primary_form": primary,
            "aliases": aliases,
            "full_def": clean_def,
            "kb_cite": kb_cite,
        })
    return entries


def is_case_strict(primary_form: str) -> bool:
    """True if the primary form has any uppercase letter past position 0.

    This catches acronyms (RoPE, GQA, MoE, FlashAttention) while letting
    sentence-leading capitalizations (Tokenization, Embedding) match
    case-insensitively.
    """
    if len(primary_form) < 2:
        return False
    return any(c.isupper() for c in primary_form[1:])


def derive_key(primary_form: str, used: set[str] | None = None) -> str:
    """Lowercase + slugify the primary form. Append -2/-3 on collision."""
    base = re.sub(r"[^a-z0-9]+", "-", primary_form.lower()).strip("-")
    if used is None or base not in used:
        return base
    n = 2
    while f"{base}-{n}" in used:
        n += 1
    return f"{base}-{n}"


def short_def(full: str, limit: int = 140) -> str:
    """First sentence of `full`, truncated to ≤limit chars."""
    # Split on first ". " (sentence boundary).
    head = re.split(r"(?<=\.)\s", full, maxsplit=1)[0]
    if len(head) <= limit:
        return head
    return head[: limit - 1].rstrip() + "…"


def build_records(entries: list[dict]) -> list[dict]:
    """Augment parser output with key, case_strict, short_def fields."""
    used: set[str] = set()
    records: list[dict] = []
    for e in entries:
        key = derive_key(e["primary_form"], used=used)
        used.add(key)
        records.append({
            "key": key,
            "primary_form": e["primary_form"],
            "aliases": e["aliases"],
            "short_def": short_def(e["full_def"]),
            "full_def": e["full_def"],
            "case_strict": is_case_strict(e["primary_form"]),
            "kb_cite": e.get("kb_cite"),
        })
    return records


def main() -> int:
    if not GLOSSARY_MD.exists():
        print(f"error: {GLOSSARY_MD} not found", flush=True)
        return 1
    text = GLOSSARY_MD.read_text(encoding="utf-8")
    entries = parse_entries(text.splitlines())
    records = build_records(entries)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"version": 1, "terms": records}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(THEORY.parent)} ({len(records)} terms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
