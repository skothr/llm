---
paper_key: wang2024-mmlu-pro
title: "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark"
authors: Wang, Ma, Wang, Wei, Lyu, Yang, et al. (TIGER Lab)
year: 2024
venue: NeurIPS 2024 D&B (Spotlight)
arxiv: 2406.01574
local_pdf: theory/sources/papers/wang2024-mmlu-pro.pdf
type: excerpts
---

# Excerpts — Wang et al. 2024, "MMLU-Pro"

## Abstract — full verbatim {#abstract}

> In the age of large-scale language models, benchmarks like the Massive
> Multitask Language Understanding (MMLU) have been pivotal in pushing
> the boundaries of what AI can achieve in language comprehension and
> reasoning across diverse domains. However, as models continue to
> improve, their performance on these benchmarks has begun to plateau,
> making it increasingly difficult to discern differences in model
> capabilities. This paper introduces MMLU-Pro, an enhanced dataset
> designed to extend the mostly knowledge-driven MMLU benchmark by
> integrating more challenging, reasoning-focused questions and
> expanding the choice set from four to ten options. Additionally,
> MMLU-Pro eliminates the trivial and noisy questions in MMLU. Our
> experimental results show that MMLU-Pro not only raises the challenge,
> causing a significant drop in accuracy by 16% to 33% compared to MMLU
> but also demonstrates greater stability under varying prompts. With 24
> different prompt styles tested, the sensitivity of model scores to
> prompt variations decreased from 4-5% in MMLU to just 2% in MMLU-Pro.
> Additionally, we found that models utilizing Chain of Thought (CoT)
> reasoning achieved better performance on MMLU-Pro compared to direct
> answering, which is in stark contrast to the findings on the original
> MMLU, indicating that MMLU-Pro includes more complex reasoning
> questions.

## §3.2 Dataset Construction Pipeline — Option Augmentation and Expert Review {#sec-3-2-construction}

> MMLU-Pro is distinctive from MMLU in the following aspects:
> 1. MMLU-Pro has ten options, which contain 3x more distractors than MMLU. By increasing the
>    distractor numbers, we significantly reduce the probability of correct guess by chance to boost the
>    benchmark's difficulty and robustness.
> 2. MMLU-Pro increases the portion of challenging college-level exam problems. These questions
>    require LLM to perform deliberate reasoning in different domains to derive the final answer.
> 3. We integrate two rounds of expert reviews to reduce the noise of the dataset. The first round is
>    based on expert verification. In the second round, we utilize the SoTA LLMs to identify potential
>    errors and employ annotators to perform more targeted verification.

The distractor-augmentation step uses GPT-4-Turbo to generate six additional choices per question (from four to ten), explicitly designed as "plausible distractors that necessitate discerning reasoning for correct selection." In experiments the authors confirm "GPT-4-Turbo does not gain additional advantage from such an augmentation procedure," ruling out the concern that the augmenter simply adds easy distractors its sibling model already avoids.

## §5 / §6.1 Model Rankings and Score Compression {#sec-6-1-rankings}

> the score difference between models like GPT-4o, Claude-3-Opus, and GPT-4-Turbo has widened
> from about 2% on MMLU to around 9% on MMLU-Pro.

> GPT-4o emerges as the strongest model with an overall accuracy of 72.6%, showing superior
> performance across all subjects.

From Table 2 (§5), the headline accuracy figures for frontier closed-source models under 5-shot CoT:

| Model | MMLU-Pro Overall |
|---|---|
| GPT-4o | 72.6% |
| Gemini-1.5-Pro | 69.0% |
| Claude-3-Opus | 68.5% |
| GPT-4-Turbo | 63.7% |
| Claude-3-Sonnet | 56.8% |

The MMLU vs. MMLU-Pro compression/expansion effect is the load-bearing discrimination claim: models clustered at 78–82% on MMLU spread to a 10% range on MMLU-Pro, making the benchmark actionable for leaderboard differentiation where MMLU was not.

## §6.2 CoT vs. Direct Answering — Benchmark Diagnostic {#sec-6-2-cot}

> GPT-4o improves by 1.5% using the Chain of Thought (CoT) method compared to
> direct answering on MMLU, while on MMLU-Pro, its improvement reaches 19.1%. Similarly,
> GPT-4-Turbo shows a 15.3% increase in performance using CoT over direct answering on MMLU-
> Pro, although its performance slightly decreases by 0.2% on MMLU.

> These findings indicate that the MMLU-Pro benchmark is specifically designed to assess deeper and more complex
> reasoning skills, as evidenced by the enhanced performance of models using chain-of-thought (CoT),
> highlighting its focus on professional-level problem-solving.

The CoT-vs-direct delta is the sharpest diagnostic for whether a benchmark measures retrieval or reasoning. MMLU's near-zero (or negative) CoT effect confirms it is a knowledge-lookup task; MMLU-Pro's 15–19% CoT gains on frontier models confirm multi-step reasoning is genuinely required. Table 3 (§6.2) provides the full breakdown across models.

## §6.3 Robustness Under Prompt Variation {#sec-6-3-robustness}

> On the MMLU benchmark, the influence of
> these prompts generally ranges between 4-5%, with peaks up to 10.98%. In contrast, on the
> MMLU-Pro benchmark, the impact of prompt changes is generally around 2%, with a maximum of
> 3.74%. This reduced variability highlights an improvement in consistency and reliability over the
> original MMLU benchmark, ensuring more reliable assessments of language models' capabilities.

Tested with 24 different but reasonable prompt styles (Figure 5, §6.3). The reduction from 4–5% to 2% prompt sensitivity is partly structural — with 10 answer choices rather than 4, the answer-letter prior (A/B/C/D position bias) carries less weight per choice, so surface rephrasing of the stem changes the prior less. The maximum peak (10.98% on MMLU vs. 3.74% on MMLU-Pro) matters for leaderboard stability: a 10% range across prompts can reorder model rankings entirely.

[Verified from PDF on 2026-05-12] Added §3.2 (#sec-3-2-construction), §6.1 rankings (#sec-6-1-rankings), §6.2 CoT (#sec-6-2-cot), §6.3 robustness (#sec-6-3-robustness). Abstract verified verbatim (matches PDF exactly). `note:` field removed; front-matter cleaned. Prior paraphrased quotes in #sec-changes, #sec-delta, #sec-cot replaced with verbatim PDF text.
