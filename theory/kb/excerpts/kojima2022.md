---
paper_key: kojima2022
title: "Large Language Models are Zero-Shot Reasoners"
authors: Kojima, Gu, Reid, Matsuo, Iwasawa
year: 2022
venue: NeurIPS
arxiv: 2205.11916
local_pdf: null
type: excerpts
note: |
  Verbatim abstract from the arXiv abstract page (v4, Jan 2023).
  PDF was not downloaded during this Phase 2 pass (sandbox/curl
  restriction); body-section quotes are pending PDF acquisition.
---

# Excerpts — Kojima et al. 2022, "Large Language Models are Zero-Shot Reasoners"

## Abstract — verbatim {#abstract}

> Pretrained large language models (LLMs) are widely used in many
> sub-fields of natural language processing (NLP) and generally known
> as excellent few-shot learners with task-specific exemplars. Notably,
> chain of thought (CoT) prompting, a recent technique for eliciting
> complex multi-step reasoning through step-by-step answer examples,
> achieved the state-of-the-art performances in arithmetics and
> symbolic reasoning, difficult system-2 tasks that do not follow the
> standard scaling laws for LLMs. While these successes are often
> attributed to LLMs' ability for few-shot learning, we show that LLMs
> are decent zero-shot reasoners by simply adding "Let's think step by
> step" before each answer. Experimental results demonstrate that our
> Zero-shot-CoT, using the same single prompt template, significantly
> outperforms zero-shot LLM performances on diverse benchmark
> reasoning tasks including arithmetics (MultiArith, GSM8K, AQUA-RAT,
> SVAMP), symbolic reasoning (Last Letter, Coin Flip), and other
> logical reasoning tasks (Date Understanding, Tracking Shuffled
> Objects), without any hand-crafted few-shot examples, e.g.
> increasing the accuracy on MultiArith from 17.7% to 78.7% and GSM8K
> from 10.4% to 40.7% with large InstructGPT model (text-davinci-002),
> as well as similar magnitudes of improvements with another off-the-
> shelf large model, 540B parameter PaLM. The versatility of this
> single prompt across very diverse reasoning tasks hints at untapped
> and understudied fundamental zero-shot capabilities of LLMs,
> suggesting high-level, multi-task broad cognitive capabilities may
> be extracted by simple prompting. We hope our work not only serves
> as the minimal strongest zero-shot baseline for the challenging
> reasoning benchmarks, but also highlights the importance of
> carefully exploring and analyzing the enormous zero-shot knowledge
> hidden inside LLMs before crafting finetuning datasets or few-shot
> exemplars.

## Headline numbers {#headline}

- Trigger phrase: **"Let's think step by step"**.
- MultiArith with text-davinci-002: 17.7% → 78.7%.
- GSM8K with text-davinci-002: 10.4% → 40.7%.
- Comparable gains on PaLM 540B.

## §3 Two-stage zero-shot CoT protocol {#sec-3}

[NOTE — pdf-not-available] Targets: the verbatim two-stage
specification:

1. **Reasoning extraction:** prompt with $x_* + $ "Let's think step by
   step" and decode $z_*$.
2. **Answer extraction:** prompt with $x_* + z_* + $ "Therefore, the
   answer is" (or similar) and decode $y_*$.

## §4 Empirical results {#sec-4}

[NOTE — pdf-not-available] Targets: the per-benchmark gains tables
across MultiArith, GSM8K, AQUA-RAT, SVAMP, Last Letter, Coin Flip,
Date Understanding, Tracking Shuffled Objects. The instruction-tuned
text-davinci-002 numbers are most cited.

## §5 Discussion — why zero-shot CoT works {#sec-5}

[NOTE — pdf-not-available] Targets: the authors' interpretation —
zero-shot CoT recovers latent capabilities present after instruction
tuning; the trigger phrase serves as a soft retrieval cue into
pretraining distribution patterns.

## Notes for the populator

- The exact trigger-phrase variants tested (e.g., "Let's solve this
  problem", "First,") are documented in the paper; recall that "Let's
  think step by step" was the strongest among tested variants.
- Section numbering anchors to v4 (Jan 2023) revision.
- Download command: `curl -sL -o theory/sources/papers/kojima2022_
  zero_shot_cot.pdf "https://arxiv.org/pdf/2205.11916"`.
