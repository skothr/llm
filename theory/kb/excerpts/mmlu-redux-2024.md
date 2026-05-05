---
paper_key: mmlu-redux-2024
title: "Are We Done with MMLU?"
authors: Polo, Drapal, Sucholutsky, Coria, Yurochkin, Sun, Ravfogel, Vidgen, Schoelkopf, Maia
year: 2024
venue: arXiv (2406.04127); NAACL 2025
arxiv: 2406.04127
local_pdf: null
type: excerpts
note: PDF not yet downloaded; excerpts compiled from the arXiv 2406.04127 abstract / project README. The headline numbers (6.49% overall error rate, 57% Virology error rate) are widely cited and quoted in the Phase 1 sweep §eval-methodology. PDF verification still pending.
---

# Excerpts — Polo et al. 2024, "Are We Done with MMLU?" (MMLU-Redux)

## Abstract — the audit finding {#abstract}

[PDF-VERIFY] Excerpting from the publicly-cited summary; full
abstract verification requires PDF download.

> [We] systematically analyze the [MMLU] benchmark and find error
> rates that compromise its current usability. […] **6.49% of MMLU
> questions contain errors** that range from incorrect ground truth
> labels to ambiguous question text and answers.

> Per-subject errors are heavy-tailed: in the Virology subject of
> MMLU, **57% of the analyzed questions contain errors**.

(Phrasing as cited across the Phase 1 sweep and downstream
literature; exact wording in arXiv v1/v2 of the paper to be
PDF-verified.)

## What the audit means

A perfect oracle scores at most ~93.5% on gold-labeled MMLU
because some gold labels are wrong. Frontier models at 90%+ are
within the verifier-error band — the benchmark *literally cannot
discriminate* models that close at headline-score level. Subject-
level scores (especially Virology) are dominated by verifier noise
and should not be reported in isolation.

## Methodological contribution

MMLU-Redux is a **methodology audit**, not a new benchmark. It
publishes per-question corrections so any subsequent run can use
the audited gold labels. Adoption: partial. The Open LLM Leaderboard
moved to MMLU-Pro for its primary headline; individual research
papers still use raw MMLU for in-distribution sanity checks.

## Implications for evaluation discipline

- **Reported MMLU rankings between frontier models inside the
  verifier-error band are unstable** to gold-label corrections.
- **Subject-specific MMLU scores need per-subject error rates**
  reported alongside.
- **The audit cadence is too slow.** MMLU was released 2020 / ICLR
  2021; the audit is 2024 — three years post-release. A faster
  cadence would help, but no organization has committed to ongoing
  benchmark audits across the field's eval suites.

## Companion / downstream

This audit is referenced in the contamination + verifier-mis-spec
discussions of:

- `kb/notes/evaluation/eval-methodology.md` §2.3
- `kb/notes/evaluation/knowledge-benchmarks.md` §2.3, §3
- `kb/notes/evaluation/reasoning-benchmarks.md` §2.1 (MMLU-Pro
  rationale)
