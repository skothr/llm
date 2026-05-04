---
paper_key: contamination-survey-2025
title: "Benchmarking LLMs Under Data Contamination: From Static to Dynamic Evaluation"
authors: (et al., EMNLP 2025 — see arXiv listing)
year: 2025
venue: EMNLP 2025
arxiv: 2502.17521
local_pdf: null
type: excerpts
note: Excerpts from the arXiv abstract page (verbatim). Detailed contamination taxonomy not available without PDF access — `[PDF-VERIFY]` flagged where needed.
---

# Excerpts — Contamination Survey 2025

## Abstract — the framing {#abstract}

> Data contamination has received increasing attention in the era of
> large language models (LLMs) due to their reliance on vast Internet-
> derived training corpora.

The structural argument:

> a transformation from static to dynamic benchmarking [is the field's
> response to contamination concerns]. Static methods enhance existing
> benchmarks, while dynamic approaches create evolving evaluation
> standards.

## Critical gap {#sec-gap}

> the lack of standardized criteria for evaluating dynamic benchmarks

This is the methodological frontier as of EMNLP 2025: how do we know
a "dynamic" benchmark is actually doing what its design promises? The
answer is not yet community-standardized.

## Methodological scope {#sec-scope}

> propose a series of optimal design principles for dynamic benchmarking
> and analyze the limitations of existing dynamic benchmarks

`[PDF-VERIFY]` The detailed taxonomy of contamination types
(input contamination, label contamination, indirect contamination,
generative contamination) is referenced but not enumerated in the
abstract. The MathArena (arXiv 2505.23281) and LiveCodeBench projects
are the canonical examples of "dynamic benchmarking" — both append
post-cutoff problems to defeat training-data leakage.
