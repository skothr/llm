#!/usr/bin/env python3
"""Wrap body-text occurrences of glossary terms in \\glsterm{key}{surface}
across the 5-paper LaTeX series, and emit per-paper glossary-section.tex
and glossary-tooltips.tex fragments scoped to terms used in each paper.

Reads theory/series/glossary-terms.json (produced by gen_glossary.py).
Idempotent: re-running on already-wrapped sources is safe.

Usage:
    python3 theory/series/mark_glossary_terms.py
    python3 theory/series/mark_glossary_terms.py --dry-run
    python3 theory/series/mark_glossary_terms.py --conflicts
    python3 theory/series/mark_glossary_terms.py --paper 3
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

THEORY = Path(__file__).resolve().parents[1]
SERIES = THEORY / "series"
TERMS_JSON = SERIES / "glossary-terms.json"
PAPERS = [SERIES / f"paper-{i}" for i in (1, 2, 3, 4, 5)]


# ---------------------------------------------------------------------------
# Skip-region tokenizer
# ---------------------------------------------------------------------------

# Regexes for skip-region detection.
_MATH_INLINE_DOLLAR = re.compile(r"(?<!\\)\$(?:\\\$|[^$])*?\$")
_MATH_INLINE_PAREN = re.compile(r"\\\(.*?\\\)", re.DOTALL)
_MATH_DISPLAY_BRACKET = re.compile(r"\\\[.*?\\\]", re.DOTALL)
_MATH_ENV = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?|eqnarray\*?|displaymath)\}"
    r".*?"
    r"\\end\{\1\}",
    re.DOTALL,
)

# Verbatim-like environments: their contents are code/literal and must not be
# wrapped. \begin{lstlisting} (listings package), \begin{verbatim}, etc.
_VERBATIM_ENV = re.compile(
    r"\\begin\{(lstlisting|verbatim|minted|alltt|Verbatim)\}"
    r".*?"
    r"\\end\{\1\}",
    re.DOTALL,
)

# LaTeX commands whose first brace-arg must be skipped (they're not body text).
# For these, the recorded skip region is just (open_brace, end_brace).
_SKIP_COMMANDS_SINGLE_ARG = {
    "label", "ref", "cref", "Cref", "eqref", "pageref",
    "cite", "citep", "citet", "citealp", "citeauthor", "citeyear",
    "input", "include", "bibliography", "bibitem",
    "section", "subsection", "subsubsection", "paragraph", "subparagraph",
    "section*", "subsection*", "subsubsection*", "paragraph*",
}

# Commands where the FULL span (\command{arg}) is skipped (to prevent re-wrapping).
# \nogls{Q} and \glsterm{key}{surface} both need the backslash included.
_SKIP_COMMANDS_FULL_SINGLE_ARG = {"nogls"}

# \glsterm has TWO brace args; the full span from \glsterm to end of second arg is skipped.
_SKIP_COMMANDS_DOUBLE_ARG = {"glsterm"}


def _find_brace_arg(text: str, open_idx: int) -> int:
    """Given an index pointing at '{', return index just past the matching '}'.

    Handles nested braces. Returns -1 if no match (malformed input).
    """
    if text[open_idx] != "{":
        return -1
    depth = 1
    i = open_idx + 1
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            i += 2  # skip escaped chars (handles \{, \})
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return i if depth == 0 else -1


def _find_bracket_arg(text: str, open_idx: int) -> int:
    """Given an index pointing at '[', return index just past matching ']'.

    Used to skip over LaTeX optional args like \\citep[\\S 3.4]{key}. LaTeX
    optargs don't nest in general; we honor \\] as an escape just in case.
    Returns -1 if no match.
    """
    if open_idx >= len(text) or text[open_idx] != "[":
        return -1
    i = open_idx + 1
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            i += 2
            continue
        if text[i] == "]":
            return i + 1
        i += 1
    return -1


def _find_command_skip_regions(text: str) -> list[tuple[int, int]]:
    """For each \\command{arg} where command is in the skip set, mark a region.

    - _SKIP_COMMANDS_SINGLE_ARG: records (open_brace, close_brace) — the arg only.
    - _SKIP_COMMANDS_FULL_SINGLE_ARG (nogls): records (cmd_start, close_brace) — full span.
    - _SKIP_COMMANDS_DOUBLE_ARG (glsterm): records (cmd_start, end_of_second_arg) — full span.
    """
    regions: list[tuple[int, int]] = []
    all_skip = (
        _SKIP_COMMANDS_SINGLE_ARG
        | _SKIP_COMMANDS_FULL_SINGLE_ARG
        | _SKIP_COMMANDS_DOUBLE_ARG
    )
    cmd_re = re.compile(r"\\([A-Za-z]+\*?)")
    for m in cmd_re.finditer(text):
        name = m.group(1)
        if name not in all_skip:
            continue
        # Skip whitespace and any optional [...] args before the mandatory
        # brace. natbib's \citep accepts up to two optional args:
        # \citep[pre][post]{key}. Without this loop the lookahead form
        # silently failed for any \cmd[...]{...} usage.
        i = m.end()
        while i < len(text) and text[i].isspace():
            i += 1
        while i < len(text) and text[i] == "[":
            close = _find_bracket_arg(text, i)
            if close == -1:
                break
            i = close
            while i < len(text) and text[i].isspace():
                i += 1
        if i >= len(text) or text[i] != "{":
            continue
        open1 = i
        end1 = _find_brace_arg(text, open1)
        if end1 == -1:
            continue
        if name in _SKIP_COMMANDS_DOUBLE_ARG:
            j = end1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] == "{":
                end2 = _find_brace_arg(text, j)
                if end2 != -1:
                    regions.append((m.start(), end2))
                    continue
            regions.append((m.start(), end1))
        elif name in _SKIP_COMMANDS_FULL_SINGLE_ARG:
            regions.append((m.start(), end1))
        else:
            # _SKIP_COMMANDS_SINGLE_ARG: record brace-arg span only
            regions.append((open1, end1))
    return regions


def _find_comment_regions(text: str) -> list[tuple[int, int]]:
    """Each '%' (not escaped as '\\%') begins a line comment that runs to '\\n'."""
    regions: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        if text[i] == "%" and (i == 0 or text[i - 1] != "\\"):
            j = text.find("\n", i)
            if j == -1:
                j = len(text)
            regions.append((i, j))
            i = j
        else:
            i += 1
    return regions


def find_skip_regions(text: str) -> list[tuple[int, int]]:
    """Return [(start, end), ...] of regions the regex MUST NOT match in.

    Skip set:
      - Math: $...$, \\(...\\), \\[...\\], \\begin{equation|align|...}
      - Verbatim envs: \\begin{lstlisting|verbatim|minted|alltt|Verbatim}
      - Reference/cite/anchor command args (\\label, \\citep, etc.)
        — including \\cmd[opt]{arg} forms with optional bracket args.
      - Section-shaped commands' brace args (\\section, \\paragraph, etc.)
      - Comments (% to end of line, ignoring escaped \\%)
      - Already-wrapped: \\glsterm{}{} (entire 2-arg call)
      - Author-disabled: \\nogls{} (entire arg)

    Captions are NOT skipped (per design — captions are reading surfaces).
    """
    regions: list[tuple[int, int]] = []
    for pattern in (_MATH_INLINE_DOLLAR, _MATH_INLINE_PAREN,
                    _MATH_DISPLAY_BRACKET, _MATH_ENV, _VERBATIM_ENV):
        for m in pattern.finditer(text):
            regions.append((m.start(), m.end()))
    regions.extend(_find_command_skip_regions(text))
    regions.extend(_find_comment_regions(text))
    return regions


# ---------------------------------------------------------------------------
# Regex builder
# ---------------------------------------------------------------------------

def build_term_regex(records: list[dict]) -> re.Pattern:
    """Compile one big alternation regex from term records.

    Surface forms (primary + aliases) >= 2 chars only. Terms shorter than
    that would catastrophically false-positive (Q, K, V are glossary
    headers but not auto-link targets).

    Sorted longest-first so e.g. 'FA-3' matches before 'FA' is considered.

    Mark loose-case terms with an inline flag group like (?i:tokenization).
    """
    parts: list[str] = []
    surfaces: list[tuple[str, bool]] = []
    for r in records:
        forms = [r["primary_form"]] + list(r.get("aliases", []))
        for s in forms:
            if len(s) >= 2:
                surfaces.append((s, bool(r.get("case_strict", True))))
    surfaces.sort(key=lambda x: len(x[0]), reverse=True)
    for s, strict in surfaces:
        esc = re.escape(s)
        if strict:
            parts.append(esc)
        else:
            parts.append(f"(?i:{esc})")
    if not parts:
        return re.compile(r"(?!)")
    pattern = r"\b(" + "|".join(parts) + r")\b"
    return re.compile(pattern)


# ---------------------------------------------------------------------------
# Body wrapper
# ---------------------------------------------------------------------------

@dataclass
class ApplyResult:
    text: str
    keys_used: set[str]


def _build_surface_to_key(records: list[dict]) -> dict[str, str]:
    """Map every surface form (primary + aliases) -> key.

    For case-loose terms, both the original-case and lowercased forms
    map to the key; lookup uses lowercased input.
    """
    table: dict[str, str] = {}
    for r in records:
        for s in [r["primary_form"]] + list(r.get("aliases", [])):
            if len(s) < 2:
                continue
            if r.get("case_strict", True):
                table[s] = r["key"]
            else:
                table[s.lower()] = r["key"]
    return table


def _is_in_skip(pos: int, regions: list[tuple[int, int]]) -> bool:
    for s, e in regions:
        if s <= pos < e:
            return True
    return False


def wrap_body(text: str, records: list[dict]) -> ApplyResult:
    """Wrap unwrapped body-text occurrences of glossary terms in the text.

    Idempotent. Already-wrapped \\glsterm{...}{...} regions are skip
    regions; the wrapper leaves them alone but counts their key as used.
    """
    skip = find_skip_regions(text)
    surface_map = _build_surface_to_key(records)
    rx = build_term_regex(records)
    out: list[str] = []
    keys_used: set[str] = set()
    last = 0
    for m in rx.finditer(text):
        s, e = m.start(), m.end()
        if _is_in_skip(s, skip):
            continue
        surface = m.group(0)
        key = surface_map.get(surface) or surface_map.get(surface.lower())
        if key is None:
            continue
        out.append(text[last:s])
        out.append(rf"\glsterm{{{key}}}{{{surface}}}")
        keys_used.add(key)
        last = e
    out.append(text[last:])
    new_text = "".join(out)
    for m in re.finditer(r"\\glsterm\{([^{}]+)\}\{", text):
        keys_used.add(m.group(1))
    return ApplyResult(text=new_text, keys_used=keys_used)


# ---------------------------------------------------------------------------
# Per-paper emitters
# ---------------------------------------------------------------------------

def _format_kb_cite(kb_cite: str | None) -> str:
    """`shazeer2019 §2.4` -> ` \\citep[\\S 2.4]{shazeer2019}`. Returns '' if None."""
    if not kb_cite:
        return ""
    m = re.match(r"^([A-Za-z][A-Za-z0-9-]*)\s*§\s*(.+)$", kb_cite.strip())
    if m:
        key, sec = m.group(1), m.group(2).strip()
        return rf" \citep[\S {sec}]{{{key}}}"
    return f" [{kb_cite}]"


def render_glossary_section(records: list[dict], used_keys: set[str]) -> str:
    """Render paper-N/glossary-section.tex content for the keys used in this paper."""
    by_key = {r["key"]: r for r in records}
    items: list[str] = []
    for key in sorted(used_keys):
        r = by_key.get(key)
        if r is None:
            continue
        primary = r["primary_form"]
        aliases = r.get("aliases", [])
        full_def = r["full_def"]
        cite = _format_kb_cite(r.get("kb_cite"))
        alias_part = f" ({', '.join(aliases)})" if aliases else ""
        items.append(
            rf"\item[\hypertarget{{glossary:{key}}}{{{primary}}}{alias_part}]"
            f"\n{full_def}{cite}"
        )
    body = "\n\n".join(items)
    header = (
        "% AUTO-GENERATED by mark_glossary_terms.py\n"
        "% DO NOT EDIT BY HAND. Source of truth: theory/kb/glossary.md\n"
        "%\n"
        r"\section*{Glossary}" "\n"
        r"\addcontentsline{toc}{section}{Glossary}" "\n"
        r"\label{sec:glossary}" "\n\n"
        r"\begin{description}" "\n"
    )
    footer = "\n" + r"\end{description}" "\n"
    return header + body + footer


def render_tooltip_table(records: list[dict], used_keys: set[str]) -> str:
    """Render paper-N/glossary-tooltips.tex with one \\def per used key."""
    by_key = {r["key"]: r for r in records}
    lines = ["% AUTO-GENERATED by mark_glossary_terms.py",
             "% DO NOT EDIT BY HAND. Per-paper short-def table for \\pdftooltip.",
             "% Source of truth: theory/kb/glossary.md",
             "%",
             r"\makeatletter"]
    for key in sorted(used_keys):
        r = by_key.get(key)
        if r is None:
            continue
        sd = r["short_def"]
        sd = re.sub(r"\$([^$]+)\$", r"\1", sd)
        sd = sd.replace("\\", "")
        sd = sd.replace("%", "\\%")
        lines.append(rf"\def\@gls@def@{key}{{{sd}}}")
    lines.append(r"\makeatother")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

SECTION_GLOB = "sections/*.tex"


def _atomic_write(path: Path, content: str) -> None:
    """Write content to a tmp file then rename — never partial-write a section."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    try:
        tmp.write(content)
        tmp.flush()
        Path(tmp.name).replace(path)
    finally:
        tmp.close()


def process_paper(paper_dir: Path, records: list[dict], dry_run: bool = False) -> dict:
    """Wrap body terms in every section file under paper_dir/sections/.
    Emit per-paper glossary-section.tex and glossary-tooltips.tex.

    Returns a report dict.
    """
    sections = sorted(paper_dir.glob(SECTION_GLOB))
    keys_used: set[str] = set()
    per_section_keys: dict[str, list[str]] = {}
    sections_modified: list[str] = []
    for sec in sections:
        original = sec.read_text(encoding="utf-8")
        result = wrap_body(original, records)
        per_section_keys[sec.name] = sorted(result.keys_used)
        keys_used |= result.keys_used
        if result.text != original:
            sections_modified.append(sec.name)
            if not dry_run:
                _atomic_write(sec, result.text)
    section_path = paper_dir / "glossary-section.tex"
    tooltips_path = paper_dir / "glossary-tooltips.tex"
    section_content = render_glossary_section(records, keys_used)
    transitive = wrap_body(section_content, records)
    tooltips_content = render_tooltip_table(records, keys_used | transitive.keys_used)
    if not dry_run:
        _atomic_write(section_path, transitive.text)
        _atomic_write(tooltips_path, tooltips_content)
    return {
        "paper": paper_dir.name,
        "sections_modified": sections_modified,
        "keys_used": sorted(keys_used | transitive.keys_used),
        "per_section_keys": per_section_keys,
    }


def run_conflicts(records: list[dict], reports: list[dict]) -> int:
    """Print a conflict-scan report. Returns an exit code (0 == clean)."""
    all_keys = {r["key"] for r in records}
    used = set()
    for rep in reports:
        used |= set(rep["keys_used"])
    unused = sorted(all_keys - used)
    print(f"# Conflict-scan report ({len(records)} terms, {len(used)} matched, "
          f"{len(unused)} never matched)")
    if unused:
        print("\n## Glossary terms with zero body matches anywhere:")
        for k in unused[:80]:
            print(f"  - {k}")
        if len(unused) > 80:
            print(f"  ... and {len(unused) - 80} more")
    over: list[tuple[str, str, int]] = []
    for rep in reports:
        per_section = rep["per_section_keys"]
        flat: dict[str, int] = {}
        for keys in per_section.values():
            for k in keys:
                flat[k] = flat.get(k, 0) + 1
        for k, n in flat.items():
            if n > 100:
                over.append((rep["paper"], k, n))
    if over:
        print("\n## Possibly over-broad matches (>100 hits in one paper):")
        for paper, k, n in sorted(over, key=lambda x: -x[2]):
            print(f"  - {paper}/{k}: {n} hits")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Print intended changes; write nothing.")
    p.add_argument("--conflicts", action="store_true",
                   help="Run scan and print conflict report (also a dry run).")
    p.add_argument("--paper", type=int, choices=(1, 2, 3, 4, 5),
                   help="Limit to one paper for fast iteration.")
    args = p.parse_args(argv)

    if not TERMS_JSON.exists():
        print(f"error: {TERMS_JSON} not found — run gen_glossary.py first")
        return 1
    data = json.loads(TERMS_JSON.read_text(encoding="utf-8"))
    records = data["terms"]
    target_papers = [PAPERS[args.paper - 1]] if args.paper else PAPERS
    dry = args.dry_run or args.conflicts
    reports = [process_paper(p, records, dry_run=dry) for p in target_papers]

    mentions = {r["paper"]: {"keys_used": r["keys_used"],
                             "sections_modified": r["sections_modified"]}
                for r in reports}
    if not dry:
        (SERIES / "glossary-mentions.json").write_text(
            json.dumps(mentions, indent=2) + "\n", encoding="utf-8")

    if args.conflicts:
        return run_conflicts(records, reports)

    for r in reports:
        print(f"{r['paper']}: {len(r['sections_modified'])} sections modified, "
              f"{len(r['keys_used'])} terms used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
