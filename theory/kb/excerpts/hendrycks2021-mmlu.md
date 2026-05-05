---
paper_key: hendrycks2021-mmlu
title: "Measuring Massive Multitask Language Understanding"
authors: Hendrycks, Burns, Basart, Zou, Mazeika, Song, Steinhardt
year: 2020 (arXiv); 2021 (ICLR)
venue: ICLR 2021
arxiv: 2009.03300
local_pdf: null
type: excerpts
note: PDF not yet downloaded; excerpts from arXiv 2009.03300 abstract (verified via prior searches). Foundational paper introducing the MMLU benchmark. Used in `kb/notes/evaluation/knowledge-benchmarks.md` and `kb/notes/evaluation/reasoning-benchmarks.md` as the canonical reference for the saturated-knowledge-benchmark lineage.
---

# Excerpts — Hendrycks et al. 2021, "Measuring Massive Multitask Language Understanding (MMLU)"

## Abstract — what MMLU is {#abstract}

> We propose a new test to measure a text model's multitask accuracy.
> The test covers 57 tasks including elementary mathematics, US
> history, computer science, law, and more. To attain high accuracy
> on this test, models must possess extensive world knowledge and
> problem solving ability. We find that while most recent models have
> near random-chance accuracy, the very largest GPT-3 model improves
> over random chance by almost 20 percentage points on average.
> However, on every one of the 57 tasks, the best models still need
> substantial improvements before they can reach expert-level
> accuracy. Models also have lopsided performance and frequently do
> not know when they are wrong. Worse, they still have near random-
> chance accuracy on some socially important subjects such as morality
> and law. By comprehensively evaluating the breadth and depth of a
> model's academic and professional understanding, our test can be
> used to analyze models across many tasks and to identify important
> shortcomings.

## Construction summary {#sec-construction}

[PDF-VERIFY] The full construction protocol — exam-bank sources,
manual filtering criteria, the 4-way MCQ format selection rationale,
the human-baseline collection methodology — needs PDF verification
before propagating as hard claims. The publicly-known summary:

- 57 subjects, grouped into four broad areas (humanities, social
  sciences, STEM, other-professional).
- 15,908 multiple-choice questions, 4 options each.
- Items sourced from publicly-available exam banks: AP exams, GRE,
  MCAT, professional licensing exams.
- Random-guess baseline: 25%.
- Human-expert baseline (cited): ~89.8% averaged across subjects.

## At-launch results {#sec-launch}

The headline 2020 finding: GPT-3 175B at launch scored ~43.9%, near
random in many subjects. Strong on humanities, weaker on calculations
and specialized professional knowledge. The "lopsided performance"
phrasing in the abstract refers to this distribution; the
"frequently do not know when they are wrong" phrasing refers to a
calibration finding (model confidence weakly correlated with
accuracy on incorrect items).

## Saturation status (2026-05) {#sec-saturation}

Frontier models exceed 90% on aggregate MMLU as of 2026; the
benchmark is **saturated** at the level of meaningful
discrimination. The MMLU-Redux audit (`mmlu-redux-2024`) found a
6.49% baseline error rate, putting the perfect-oracle ceiling at
~93.5%, *inside* the verifier-error band of frontier models.

Lineage successors (MMLU-Pro, GPQA Diamond, HLE) are deep-dived in
`kb/notes/evaluation/reasoning-benchmarks.md`. MMLU itself remains
in nearly every model card as a fast in-distribution smoke test.
