"""Tests for mark_glossary_terms.py — tokenizer, regex, body-wrap, emitters."""
from __future__ import annotations

from mark_glossary_terms import find_skip_regions


def test_skip_inline_math():
    text = r"The shape is $B \times S \times D$ and matters."
    regions = find_skip_regions(text)
    starts = [r[0] for r in regions]
    ends = [r[1] for r in regions]
    assert any(text[s:e].startswith("$") for s, e in regions)
    math_start = text.index("$")
    math_end = text.index("$", math_start + 1) + 1
    assert (math_start, math_end) in regions


def test_skip_display_math():
    text = r"Before \[ x = 1 \] after."
    regions = find_skip_regions(text)
    s = text.index(r"\[")
    e = text.index(r"\]") + 2
    assert (s, e) in regions


def test_skip_equation_environment():
    text = "before \\begin{equation}\n  E = mc^2\n\\end{equation} after"
    regions = find_skip_regions(text)
    s = text.index(r"\begin{equation}")
    e = text.index(r"\end{equation}") + len(r"\end{equation}")
    assert (s, e) in regions


def test_skip_label_arg():
    text = r"See \label{sec:rope} for details."
    regions = find_skip_regions(text)
    s = text.index("{")
    e = text.index("}") + 1
    assert (s, e) in regions


def test_skip_citep_arg():
    text = r"as shown by \citep{vaswani2017} and"
    regions = find_skip_regions(text)
    s = text.index("{")
    e = text.index("}") + 1
    assert (s, e) in regions


def test_skip_section_title_arg():
    text = r"\section{Attention is all you need}"
    regions = find_skip_regions(text)
    s = text.index("{")
    e = text.index("}") + 1
    assert (s, e) in regions


def test_skip_caption_NOT_skipped():
    """Captions are wrapped (per design: random-access > caption cleanliness)."""
    text = r"\caption{The RoPE rotation diagram.}"
    regions = find_skip_regions(text)
    s = text.index("{")
    e = text.index("}") + 1
    assert (s, e) not in regions


def test_skip_already_glsterm_wrapped():
    text = r"the \glsterm{rope}{RoPE} encoding"
    regions = find_skip_regions(text)
    cmd_start = text.index(r"\glsterm")
    cmd_end = text.index("}", text.index("}") + 1) + 1  # second }
    assert (cmd_start, cmd_end) in regions


def test_skip_nogls_wrapped():
    text = r"the bare \nogls{Q} symbol"
    regions = find_skip_regions(text)
    cmd_start = text.index(r"\nogls")
    cmd_end = text.index("}") + 1
    assert (cmd_start, cmd_end) in regions


def test_skip_line_comment():
    text = "before % this RoPE is in a comment\nRoPE on next line"
    regions = find_skip_regions(text)
    s = text.index("%")
    e = text.index("\n", s)
    assert (s, e) in regions


# ---------------------------------------------------------------------------
# Regex builder tests
# ---------------------------------------------------------------------------

from mark_glossary_terms import build_term_regex


def _records(*specs):
    """Helper: tiny term records for regex tests."""
    out = []
    for s in specs:
        if isinstance(s, dict):
            out.append(s)
        else:
            key, primary, aliases, case_strict = s
            out.append({
                "key": key,
                "primary_form": primary,
                "aliases": aliases,
                "case_strict": case_strict,
            })
    return out


def test_regex_word_boundaries_skip_substring():
    rx = build_term_regex(_records(("rope", "RoPE", [], True)))
    assert rx.search("iRoPE") is None
    assert rx.search("uses RoPE encoding").group(0) == "RoPE"


def test_regex_longest_first_for_hyphen_overlap():
    rx = build_term_regex(_records(
        ("fa", "FA", [], True),
        ("fa-3", "FA-3", [], True),
    ))
    m = rx.search("we use FA-3 here")
    assert m.group(0) == "FA-3"


def test_regex_case_strict_no_lowercase_match():
    rx = build_term_regex(_records(("rope", "RoPE", [], True)))
    assert rx.search("the rope is short") is None
    assert rx.search("the RoPE works") is not None


def test_regex_case_loose_matches_both_cases():
    rx = build_term_regex(_records(("tokenization", "Tokenization", [], False)))
    assert rx.search("Tokenization is the first step") is not None
    assert rx.search("tokenization is the first step") is not None


def test_regex_alias_matches():
    rx = build_term_regex(_records(
        ("t5", "T5", ["Text-to-Text Transfer Transformer"], True),
    ))
    m1 = rx.search("Google's T5 model")
    assert m1.group(0) == "T5"
    m2 = rx.search("the Text-to-Text Transfer Transformer paper")
    assert m2.group(0) == "Text-to-Text Transfer Transformer"


def test_regex_skips_terms_under_2_chars():
    rx = build_term_regex(_records(("q", "Q", [], True)))
    if rx.pattern:
        assert rx.search("the Q value here") is None


# ---------------------------------------------------------------------------
# Body wrapper tests
# ---------------------------------------------------------------------------

from mark_glossary_terms import wrap_body, ApplyResult


def test_wrap_simple_body():
    records = _records(("rope", "RoPE", [], True))
    text = "We use RoPE for positional encoding."
    result = wrap_body(text, records)
    assert result.text == r"We use \glsterm{rope}{RoPE} for positional encoding."
    assert result.keys_used == {"rope"}


def test_wrap_skips_inside_math():
    records = _records(("rope", "RoPE", [], True))
    text = r"$\text{RoPE}$ is the math; RoPE is the prose."
    result = wrap_body(text, records)
    assert r"\text{RoPE}" in result.text
    assert r"\glsterm{rope}{RoPE}" in result.text
    assert result.text.count(r"\glsterm{") == 1


def test_wrap_idempotent():
    records = _records(("rope", "RoPE", [], True))
    text = r"\glsterm{rope}{RoPE} is already wrapped, RoPE next."
    result1 = wrap_body(text, records)
    result2 = wrap_body(result1.text, records)
    assert result1.text == result2.text
    assert result1.text.count(r"\glsterm{") == 2


def test_wrap_preserves_surface_form():
    records = _records(
        ("t5", "T5", ["Text-to-Text Transfer Transformer"], True),
    )
    text = "The T5 paper and the Text-to-Text Transfer Transformer paper."
    result = wrap_body(text, records)
    assert r"\glsterm{t5}{T5}" in result.text
    assert r"\glsterm{t5}{Text-to-Text Transfer Transformer}" in result.text


# ---------------------------------------------------------------------------
# Per-paper emitter tests
# ---------------------------------------------------------------------------

from mark_glossary_terms import render_glossary_section, render_tooltip_table


def test_render_glossary_section_basic():
    records = _records({
        "key": "rope",
        "primary_form": "RoPE",
        "aliases": ["Rotary Position Embedding"],
        "short_def": "Rotates queries and keys.",
        "full_def": "Rotates queries and keys by position-dependent angles.",
        "case_strict": True,
        "kb_cite": "su2021 §3.4",
    })
    out = render_glossary_section(records, used_keys={"rope"})
    assert r"\section*{Glossary}" in out
    assert r"\hypertarget{glossary:rope}" in out
    assert "RoPE" in out
    assert "Rotary Position Embedding" in out
    assert "Rotates queries and keys by position-dependent angles." in out
    assert r"\citep[\S 3.4]{su2021}" in out


def test_render_glossary_section_scopes_to_used_keys():
    records = _records(
        {"key": "rope", "primary_form": "RoPE", "aliases": [],
         "short_def": "x", "full_def": "x", "case_strict": True, "kb_cite": None},
        {"key": "gqa", "primary_form": "GQA", "aliases": [],
         "short_def": "y", "full_def": "y", "case_strict": True, "kb_cite": None},
    )
    out = render_glossary_section(records, used_keys={"rope"})
    assert "RoPE" in out
    assert "GQA" not in out


def test_render_tooltip_table():
    records = _records({
        "key": "rope", "primary_form": "RoPE", "aliases": [],
        "short_def": "Rotary positional encoding.", "full_def": "...",
        "case_strict": True, "kb_cite": None,
    })
    out = render_tooltip_table(records, used_keys={"rope"})
    assert r"\def\@gls@def@rope{Rotary positional encoding.}" in out
