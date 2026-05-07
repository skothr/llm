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
    assert derive_key("RoPE", []) == "rope"
    assert derive_key("FlashAttention", []) == "flashattention"
    assert derive_key("Multi-Head Attention", ["mha"]) == "multi-head-attention"


def test_derive_key_collision_suffix():
    used: set[str] = {"rope"}
    assert derive_key("Rope", [], used=used) == "rope-2"


def test_short_def_truncates_to_one_sentence():
    full = "Positional encoding that rotates query and key vectors. Applied at every layer."
    assert short_def(full).startswith("Positional encoding")
    assert short_def(full).endswith(".") or len(short_def(full)) <= 140
    assert "Applied at every layer" not in short_def(full)


def test_short_def_long_sentence_truncated_to_140():
    full = "x" * 200
    s = short_def(full)
    assert len(s) <= 140
