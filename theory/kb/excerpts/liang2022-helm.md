---
paper_key: liang2022-helm
title: "Holistic Evaluation of Language Models"
authors: Liang, Bommasani, Lee, Tsipras, Soylu, Yasunaga, Zhang, Narayanan, Wu, Kumar, et al. (Stanford CRFM)
year: 2022
venue: arXiv (Stanford CRFM)
arxiv: 2211.09110
local_pdf: theory/sources/papers/liang2022-helm.pdf
type: excerpts
note: Excerpts from arXiv abstract (verbatim). The 7-metric x 16-scenario matrix and the "core scenarios" methodology are documented on https://crfm.stanford.edu/helm/. `[PDF-VERIFY]` flags claims relying on the public site rather than the (1000+ page) PDF.
---

# Excerpts — Liang et al. 2022, "HELM"

## Abstract — the missing-axis argument {#abstract}

> Language models (LMs) are becoming the foundation for almost all major
> language technologies, but their capabilities, limitations, and risks
> are not well understood.

The structural claim: previous benchmarks measured single axes
(accuracy on a task), HELM measures **multiple axes per scenario**.

> we measure 7 metrics ... for each of 16 core scenarios when possible
> (87.5% of the time).

## The 7-metric framework {#sec-metrics}

> accuracy, calibration, robustness, fairness, bias, toxicity, and
> efficiency

This is the load-bearing taxonomic claim that subsequent eval
frameworks (Open LLM Leaderboard, BIG-bench, lm-eval-harness) inherit
in part: each scenario is a `(prompt-distribution, gold-answer-protocol,
metric-bundle)` triple, not just a `(prompt, answer)` pair.

Metric definitions used in HELM (`[PDF-VERIFY]` from the public site):

- **Accuracy** — task-specific (exact match, F1, BLEU, etc.).
- **Calibration** — Brier score / ECE between predicted probability and
  empirical accuracy on multiple-choice tasks.
- **Robustness** — accuracy delta under perturbed prompts (typos,
  capitalization, paraphrase).
- **Fairness** — accuracy-gap across demographic-subgroup-conditioned
  prompts.
- **Bias** — demographic representation in open-ended generation.
- **Toxicity** — Perspective-API toxicity rate of generations.
- **Efficiency** — wall-clock and energy / inference call.

## The "before HELM" coverage gap {#sec-coverage}

> Before HELM, models on average were evaluated on just 17.9% of the
> core HELM scenarios.

> [HELM evaluates] 30 prominent language models ... on all 42 scenarios

Methodologically this is the headline contribution — the comparability
issue is solved by running every model under the same conditions, not
by trusting individual reported numbers from individual papers.

## Output discipline {#sec-output-discipline}

> [HELM] release[s] all raw model prompts and completions

This is the reproducibility move: the dataset of (prompt, response)
pairs is part of the artifact, not just the aggregate scores. This
distinction matters for the contamination story — scores can be
re-derived later under tighter rules without re-running the model.
