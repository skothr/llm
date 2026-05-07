"""Pytest setup for theory/series/ test suite.

Adds theory/series/ to sys.path so tests can import gen_glossary and
mark_glossary_terms as top-level modules.
"""
from __future__ import annotations
import sys
from pathlib import Path

THEORY_SERIES = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THEORY_SERIES))
