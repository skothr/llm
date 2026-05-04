---
paper_key: wang2024-mmlu-pro
title: "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark"
authors: Wang, Ma, Wang, Wei, Lyu, Yang, et al. (TIGER Lab)
year: 2024
venue: NeurIPS 2024 D&B (Spotlight)
arxiv: 2406.01574
local_pdf: null
type: excerpts
note: Excerpts from arXiv abstract (verbatim). Distractor-construction protocol details tagged `[PDF-VERIFY]`.
---

# Excerpts — Wang et al. 2024, "MMLU-Pro"

## Abstract — the saturation response {#abstract}

> [MMLU-Pro is] an enhanced dataset designed to extend the mostly
> knowledge-driven MMLU benchmark by integrating more challenging,
> reasoning-focused questions and expanding the choice set from four
> to ten options.

Three structural changes from MMLU:

1. **More choices.** 4-way → 10-way MCQ. Random-guess ceiling drops
   from 25% → 10%, expanding the dynamic range above the noise floor.
2. **Reasoning-focused.** Phase 1 sweep flag confirmed: original MMLU
   was retrieval-heavy, MMLU-Pro shifts toward problems that require
   multi-step reasoning.
3. **De-noised.** "Eliminates trivial and noisy questions from the
   original MMLU."

## Headline discrimination delta {#sec-delta}

> MMLU-Pro not only raises the challenge, causing a significant drop in
> accuracy by 16% to 33% compared to MMLU but also demonstrates greater
> stability under varying prompts.

Quantification of the prompt-sensitivity improvement:

> Model sensitivity to prompt variations decreased from 4-5% on MMLU
> to approximately 2% on MMLU-Pro

Tested across 24 prompt styles. The 4–5% prompt-induced score variance
on MMLU is significant — it can reorder leaderboard positions —
and MMLU-Pro halves it, partly via the 10-way design (more distractors
make the answer-letter prior less load-bearing) and partly via the
question selection.

## CoT effect — diagnostic for "real reasoning" {#sec-cot}

> Chain-of-Thought reasoning showed marked improvements on MMLU-Pro,
> contrasting with the original MMLU where direct answering performed
> comparably—indicating the benchmark includes substantially more
> complex reasoning questions.

This is the load-bearing methodological claim: a benchmark where
CoT and direct-answer give the same score is dominated by retrieval
(no compute spent on reasoning helps). MMLU-Pro's CoT-vs-direct gap
is evidence that the benchmark's hard items genuinely demand
intermediate-step computation.
