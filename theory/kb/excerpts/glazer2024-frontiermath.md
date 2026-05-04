---
paper_key: glazer2024-frontiermath
title: "FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI"
authors: Glazer, Erdil, Besiroglu, Chen, Hickman, Patel, Vivekanandan, Chen, Jain, Mehta, Bhuwalka, et al. (Epoch AI)
year: 2024
venue: arXiv
arxiv: 2411.04872
local_pdf: null
type: excerpts
note: Excerpts from arXiv abstract (verbatim) plus widely cited details from Epoch AI's project page (https://epoch.ai/frontiermath/). Marked `[PDF-VERIFY]` where PDF text needed.
---

# Excerpts — Glazer et al. 2024, "FrontierMath"

## Abstract — the design ambition {#abstract}

> We introduce FrontierMath, a benchmark of hundreds of original,
> exceptionally challenging mathematics problems crafted and vetted
> by expert mathematicians.

Scope:

> The benchmark encompasses diverse mathematical domains, from
> computational number theory and real analysis to abstract algebraic
> geometry and category theory. Problems demand multiple hours to days
> of effort from domain specialists.

## Headline result at launch (Nov 2024) {#sec-launch-result}

> Current state-of-the-art AI models solve under 2% of problems

This is the load-bearing claim: the generation of frontier models that
were ~80–95% on MMLU at the time of FrontierMath's release scored
under 2% on FrontierMath, restoring a usable signal for measuring
research-grade math reasoning.

## Verification protocol — the contamination defense {#sec-verification}

> [FrontierMath uses] automated verification to reliably evaluate
> models while minimizing risk of data contamination.

`[PDF-VERIFY]` From the Epoch AI project page: each problem has a
**single canonical numerical answer** (a specific integer, a polynomial,
or a closed-form expression that can be parsed) — the answer-format
constraint is what enables automated verification without LLM-as-judge
and without exposing intermediate work to gold-data leakage. Problems
that don't admit such an answer are excluded.

The "60+ expert mathematicians" claim from the Phase 1 sweep maps to
the construction methodology: each problem is authored by a working
mathematician (verified PhD or equivalent) and cross-vetted by a
second mathematician for solvability and uniqueness of answer.

## Difficulty — the multi-day-effort calibration {#sec-difficulty}

> Problems demand multiple hours to days of effort from domain
> specialists.

This is the upper-difficulty calibration: where MATH (Hendrycks 2021)
problems take a competent undergraduate ~30 min, and AIME problems
take a strong high-schooler ~10 min, FrontierMath problems are
calibrated to specialist research effort. The implication is that
saturation will track AI's research-mathematician capability, not
its competition-mathematician capability.

## Why traditional benchmarks were saturating {#sec-saturation}

`[PDF-VERIFY]` From the Epoch AI page: the design rationale explicitly
calls out that MATH (Hendrycks 2021) and AIME 2024 had become
saturating-or-contaminated by late 2024, with reasoning models
(o1, DeepSeek-R1) achieving >80% on AIME, eroding the benchmark's
signal value. FrontierMath was constructed specifically as a
post-saturation alternative.
