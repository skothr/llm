---
paper_key: wei2022-cot
title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
authors: Wei, Wang, Schuurmans, Bosma, Ichter, Xia, Chi, Le, Zhou
year: 2022
venue: NeurIPS
arxiv: 2201.11903
local_pdf: null
type: excerpts
note: |
  Verbatim abstract from the arXiv abstract page (v6, Jan 2023).
  PDF was not downloaded during this Phase 2 pass (sandbox/curl
  restriction); section-level quotes inside the body of the paper
  are pending PDF acquisition. Section anchors are stable for
  follow-up population.
---

# Excerpts — Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

## Abstract — verbatim {#abstract}

> We explore how generating a chain of thought — a series of intermediate
> reasoning steps — significantly improves the ability of large language
> models to perform complex reasoning. In particular, we show how such
> reasoning abilities emerge naturally in sufficiently large language
> models via a simple method called chain-of-thought prompting, where a
> few chain of thought demonstrations are provided as exemplars in
> prompting. Experiments on three large language models show that
> chain-of-thought prompting improves performance on a range of
> arithmetic, commonsense, and symbolic reasoning tasks. The empirical
> gains can be striking. For instance, prompting a 540B-parameter
> language model with just eight chain-of-thought exemplars achieves
> state-of-the-art accuracy on the GSM8K benchmark of math word
> problems, surpassing even fine-tuned GPT-3 with a verifier.

## §1 Introduction — what CoT prompting is {#sec-1}

[NOTE — pdf-not-available] Section anchored for follow-up population.
Targets: the headline definition ("chain of thought — a series of
intermediate reasoning steps"); the few-shot exemplar formulation;
the headline result on math reasoning at 540B scale.

## §3.1 Few-shot CoT prompt construction {#sec-3-1}

[NOTE — pdf-not-available] Targets: the exact specification of the
few-shot exemplar format. Wei et al. used **8 exemplars** for
arithmetic-reasoning prompts (verifiable from the paper's Table 1
caption, broadly cited downstream). Construction of the reasoning-trace
component; the answer-extraction template ("The answer is X").

## §3.3 Emergent ability finding {#sec-3-3}

[NOTE — pdf-not-available] Targets: the scale-thresholding result. CoT
helps PaLM 540B but does not help (and can hurt) sub-100B models on
GSM8K. The exact figures from Figure 4. Verifying the emergence
inflection point and the smaller-model degradation is load-bearing
for the §3.3 claim in `kb/notes/reasoning/chain-of-thought.md`.

## §6 Limitations and discussion {#sec-6}

[NOTE — pdf-not-available] Targets: the authors' explicit limitations:
applicability mainly to math, commonsense, symbolic; failure on tasks
where CoT demonstrations are hard to construct; explicit hedging that
"chain-of-thought" does not necessarily reflect model "reasoning" in
any cognitive sense.

## §3.2 Datasets covered {#sec-3-2}

[NOTE — pdf-not-available] Targets: GSM8K, SVAMP, ASDiv, AQuA, MAWPS
(arithmetic); CommonsenseQA, StrategyQA, Date, Sports
(commonsense / symbolic); Last Letter, Coin Flip (symbolic). Cross-
reference to `kb/notes/evaluation/reasoning-benchmarks.md`.

## Notes for the populator

- The paper went through several arXiv revisions; v6 (Jan 2023) is the
  canonical NeurIPS-version. Anchor section numbers against v6.
- Section 4 is robustness analysis (perturbations to exemplars); useful
  for the faithfulness discussion in `chain-of-thought.md` but not
  cited in the current draft.
- Equation numbers: this paper has no display equations (it is a
  prompting paper). Citations should be of the form `[wei2022-cot
  §3.1]`, not `[wei2022-cot §3.1 eq.X]`.
- Download command: `curl -sL -o theory/sources/papers/wei2022-cot_
  chain_of_thought.pdf "https://arxiv.org/pdf/2201.11903"`.
