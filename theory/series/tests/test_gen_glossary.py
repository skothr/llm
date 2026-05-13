"""Tests for gen_glossary.py — parser, key derivation, JSON emission."""
from __future__ import annotations

from gen_glossary import parse_entries, is_case_strict, derive_key, short_def


def test_parse_plain_bullet_entry():
    md = "- **RoPE** — Positional encoding that rotates query and key vectors."
    entries = parse_entries(md.splitlines())
    assert len(entries) == 1
    e = entries[0]
    assert e["primary_form"] == "RoPE"
    assert e["aliases"] == []
    assert e["full_def"] == "Positional encoding that rotates query and key vectors."


def test_parse_parenthetical_alias():
    md = "- **T5 (Text-to-Text Transfer Transformer)** — Google's encoder-decoder model."
    entries = parse_entries(md.splitlines())
    assert len(entries) == 1
    e = entries[0]
    assert e["primary_form"] == "T5"
    assert e["aliases"] == ["Text-to-Text Transfer Transformer"]


def test_parse_parenthetical_lowercase_skipped():
    # Parentheticals that don't look acronym-shaped or capitalized are NOT aliases.
    md = "- **Token Embedding** — A learned lookup (input projection)."
    entries = parse_entries(md.splitlines())
    assert entries[0]["aliases"] == []  # "input projection" lowercase → not alias


def test_parse_kb_cite_trailing():
    md = "- **MQA (Multi-Query Attention)** — Single shared K, V across heads. `[shazeer2019 §2.4]`"
    entries = parse_entries(md.splitlines())
    e = entries[0]
    assert e["kb_cite"] == "shazeer2019 §2.4"
    # The kb-cite must be stripped from full_def:
    assert "shazeer2019" not in e["full_def"]
    assert e["full_def"].rstrip().endswith("heads.")


def test_parse_kb_cite_absent():
    md = "- **LLM** — Large Language Model. A neural network."
    entries = parse_entries(md.splitlines())
    assert entries[0]["kb_cite"] is None


def test_parse_kb_cite_with_trailing_period():
    """Entries that end `[paper §X]`.` (note trailing period) must still
    strip the kb-cite cleanly — otherwise the cite leaks into full_def
    and downstream LaTeX wrapping (\\texttt) turns it monospace."""
    md = "- **MQA** — Single shared K, V. `[shazeer2019 §2.4]`."
    entries = parse_entries(md.splitlines())
    e = entries[0]
    assert e["kb_cite"] == "shazeer2019 §2.4"
    assert "shazeer2019" not in e["full_def"]


def test_parse_multiline_bullet_joins_continuations():
    """Indented continuation lines after a bullet are joined into the
    full_def. Real entries in theory/kb/glossary.md routinely span 3-8
    lines; the parser must capture the whole text, not just line 1."""
    md = (
        "- **Activation patching** — the canonical causal-intervention method\n"
        "  in MI: identify which model components matter for a behavior by\n"
        "  swapping their activations between forward passes on paired\n"
        "  (clean, corrupted) prompts and measuring the effect on the output."
    )
    entries = parse_entries(md.splitlines())
    assert len(entries) == 1
    e = entries[0]
    assert "swapping their activations" in e["full_def"]
    assert "measuring the effect on the output." in e["full_def"]
    # The continuation lines are joined with single spaces (not newlines).
    assert "method\n" not in e["full_def"]
    assert "method in MI" in e["full_def"]


def test_parse_multiline_bullet_ends_at_next_bullet():
    """A continuation line that starts with `- ` begins the next entry,
    not a continuation of the current one."""
    md = (
        "- **Term A** — definition of A continues on\n"
        "  this indented second line.\n"
        "- **Term B** — definition of B."
    )
    entries = parse_entries(md.splitlines())
    assert len(entries) == 2
    assert entries[0]["primary_form"] == "Term A"
    assert "this indented second line." in entries[0]["full_def"]
    assert entries[1]["primary_form"] == "Term B"
    assert entries[1]["full_def"] == "definition of B."


def test_parse_multiline_bullet_kb_cite_spans_lines():
    """KB citations like `[paper §X; kb/excerpts/...]` sometimes wrap
    across two indented lines in the source. After joining, the
    trailing-cite regex should still match and strip cleanly."""
    md = (
        "- **Alignment faking** — Operational sub-form of deceptive alignment:\n"
        "  a model complies with a behavior it normally refuses.\n"
        "  `[greenblatt2024-alignment-faking §abstract;\n"
        "  kb/excerpts/greenblatt2024-alignment-faking#abstract]`"
    )
    entries = parse_entries(md.splitlines())
    e = entries[0]
    assert e["primary_form"] == "Alignment faking"
    assert "Operational sub-form" in e["full_def"]
    assert "model complies" in e["full_def"]
    # The KB-cite should be stripped from full_def AND captured in kb_cite.
    assert "greenblatt2024" not in e["full_def"]
    assert e["kb_cite"] is not None
    assert "greenblatt2024-alignment-faking" in e["kb_cite"]


def test_case_strict_acronym_detected():
    assert is_case_strict("RoPE") is True
    assert is_case_strict("GQA") is True
    assert is_case_strict("FlashAttention") is True
    assert is_case_strict("MoE") is True


def test_case_strict_first_cap_not_strict():
    assert is_case_strict("Tokenization") is False
    assert is_case_strict("Embedding") is False


def test_case_strict_lowercase_not_strict():
    assert is_case_strict("attention") is False


def test_derive_key_simple():
    assert derive_key("RoPE") == "rope"
    assert derive_key("FlashAttention") == "flashattention"
    assert derive_key("Multi-Head Attention") == "multi-head-attention"


def test_derive_key_collision_suffix():
    used: set[str] = {"rope"}
    assert derive_key("Rope", used=used) == "rope-2"


def test_short_def_truncates_to_one_sentence():
    full = "Positional encoding that rotates query and key vectors. Applied at every layer."
    assert short_def(full).startswith("Positional encoding")
    assert short_def(full).endswith(".") or len(short_def(full)) <= 140
    assert "Applied at every layer" not in short_def(full)


def test_short_def_long_sentence_truncated_to_140():
    full = "x" * 200
    s = short_def(full)
    assert len(s) <= 140
