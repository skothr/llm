---
paper_key: wang2022-self-consistency
title: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
authors: Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou
year: 2022
venue: ICLR 2023
arxiv: 2203.11171
local_pdf: null
type: excerpts
note: |
  Verbatim abstract from the arXiv abstract page (v4, Mar 2023).
  PDF was not downloaded during this Phase 2 pass (sandbox/curl
  restriction); body-section quotes are pending PDF acquisition.
---

# Excerpts — Wang et al. 2022, "Self-Consistency Improves Chain of Thought Reasoning"

## Abstract — verbatim {#abstract}

> Chain-of-thought prompting combined with pre-trained large language
> models has achieved encouraging results on complex reasoning tasks.
> In this paper, we propose a new decoding strategy, self-consistency,
> to replace the naive greedy decoding used in chain-of-thought
> prompting. It first samples a diverse set of reasoning paths instead
> of only taking the greedy one, and then selects the most consistent
> answer by marginalizing out the sampled reasoning paths. Self-
> consistency leverages the intuition that a complex reasoning problem
> typically admits multiple different ways of thinking leading to its
> unique correct answer. Our extensive empirical evaluation shows that
> self-consistency boosts the performance of chain-of-thought
> prompting with a striking margin on a range of popular arithmetic
> and commonsense reasoning benchmarks, including GSM8K (+17.9%),
> SVAMP (+11.0%), AQuA (+12.2%), StrategyQA (+6.4%) and ARC-challenge
> (+3.9%).

## Headline numbers {#headline}

- GSM8K: +17.9% over greedy CoT.
- SVAMP: +11.0%.
- AQuA: +12.2%.
- StrategyQA: +6.4%.
- ARC-challenge: +3.9%.

## §3 Self-consistency decoding strategy {#sec-3}

[NOTE — pdf-not-available] Targets: the verbatim three-step protocol:

1. Sample $N$ reasoning paths $z^{(1)}, \ldots, z^{(N)}$ at
   temperature $T > 0$.
2. Extract answers $y^{(1)}, \ldots, y^{(N)}$.
3. Return $\hat y = \arg\max_y \sum_n \mathbb{1}[y^{(n)} = y]$.

The intuition formulated by the authors: complex reasoning problems
admit multiple distinct correct reasoning paths that all converge on
the same answer; majority over diverse traces concentrates probability
mass on the correct $y$ even when individual traces err.

## §4 Empirical evaluation {#sec-4}

[NOTE — pdf-not-available] Targets: per-benchmark gains tables; ablation
on $N$ (sample count); ablation on $T$ (temperature). The reported
gains are stable across $N \in [10, 40]$ with diminishing returns
beyond $N \approx 40$.

## §5 Limitations {#sec-5}

[NOTE — pdf-not-available] Targets: the authors' acknowledgement that
self-consistency requires a low-cardinality answer space (math
answers, multiple-choice). For free-form generation, the analogue is
best-of-$N$ under a verifier — see `kb/notes/reasoning/process-
supervision.md` for that line.

## Notes for the populator

- The "self-consistency" decoding mechanism is mathematically a
  (uniform) majority vote over decoded answers; the paper's
  formulation as "marginalizing out the sampled reasoning paths" is
  the Bayesian framing.
- Section numbering anchors to the ICLR 2023 / arXiv v4 (Mar 2023)
  version.
- Download command: `curl -sL -o theory/sources/papers/wang2022_self
  _consistency.pdf "https://arxiv.org/pdf/2203.11171"`.
