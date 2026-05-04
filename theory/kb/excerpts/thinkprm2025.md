---
paper_key: thinkprm2025
title: "Process Reward Models That Think"
authors: Khalifa, et al.
year: 2025
venue: arXiv
arxiv: 2504.16828
local_pdf: null
type: excerpts
note: |
  Verbatim abstract from the arXiv abstract page (v1, Apr 2025).
  PDF was not downloaded during this Phase 2 pass (sandbox/curl
  restriction); body-section quotes are pending PDF acquisition.
  Section anchors are stable for follow-up population.
---

# Excerpts — Khalifa et al. 2025, "Process Reward Models That Think"

## Abstract — verbatim {#abstract}

> Step-by-step verifiers — also known as process reward models (PRMs)
> — are a key ingredient for test-time scaling. PRMs require step-level
> supervision, making them expensive to train. This work aims to build
> data-efficient PRMs as verbalized step-wise reward models that verify
> every step in the solution by generating a verification chain-of-
> thought (CoT). We propose ThinkPRM, a long CoT verifier fine-tuned
> on orders of magnitude fewer process labels than those required by
> discriminative PRMs. Our approach capitalizes on the inherent
> reasoning abilities of long CoT models, and outperforms LLM-as-a-
> Judge and discriminative verifiers — using only 1% of the process
> labels in PRM800K — across several challenging benchmarks.
> Specifically, ThinkPRM beats the baselines on ProcessBench, MATH-500,
> and AIME '24 under best-of-N selection and reward-guided search. In
> an out-of-domain evaluation on a subset of GPQA-Diamond and
> LiveCodeBench, our PRM surpasses discriminative verifiers trained on
> the full PRM800K by 8% and 4.5%, respectively. Lastly, under the
> same token budget, ThinkPRM scales up verification compute more
> effectively compared to LLM-as-a-Judge, outperforming it by 7.2% on
> a subset of ProcessBench.

## Headline numbers {#headline}

- Uses only **1%** of PRM800K process labels.
- Out-of-domain gains: +**8%** on GPQA-Diamond, +**4.5%** on
  LiveCodeBench (vs full-PRM800K discriminative verifiers).
- +**7.2%** on ProcessBench vs LLM-as-a-Judge at matched token
  budget.

## §3 ThinkPRM mechanism {#sec-3}

[NOTE — pdf-not-available] Targets: the verbatim specification of
ThinkPRM — a small reasoning LLM that, given a problem and a partial
trace, emits its own short CoT evaluating each step before producing
a verdict token. The generative-PRM substrate is a long-CoT-finetuned
LLM (likely an R1-distill); training is on the 1% subset of PRM800K
plus self-generated synthetic verification traces.

## §4 Empirical headline {#sec-4}

[NOTE — pdf-not-available] Targets: per-benchmark gains tables across
ProcessBench, MATH-500, AIME 2024, GPQA-Diamond (out-of-domain),
LiveCodeBench (out-of-domain). Best-of-$N$ accuracy at fixed $N$;
reward-guided search accuracy.

## §5 Discussion — when generative PRMs win {#sec-5}

[NOTE — pdf-not-available] Targets: the authors' framing of when a
generative PRM outperforms a discriminative one — concentrated where
each step has a verifiable formal certificate (algebraic verification,
type checking) the verifier's CoT can apply.

## Notes for the populator

- The "1% of PRM800K" claim is structural: ~8,000 step labels vs the
  full ~800,000. The 100× reduction is the headline data-efficiency
  number.
- Generative-PRM cost asymmetry: best-of-$N$ with a generative PRM
  costs $O(N \times L_\text{verify})$ tokens — note when populating
  §3.
- Download command: `curl -sL -o theory/sources/papers/thinkprm2025
  .pdf "https://arxiv.org/pdf/2504.16828"`.
