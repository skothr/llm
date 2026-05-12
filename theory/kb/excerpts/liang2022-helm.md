---
paper_key: liang2022-helm
title: "Holistic Evaluation of Language Models"
authors: Liang, Bommasani, Lee, Tsipras, Soylu, Yasunaga, Zhang, Narayanan, Wu, Kumar, et al. (Stanford CRFM)
year: 2022
venue: Transactions on Machine Learning Research (08/2023)
arxiv: 2211.09110
local_pdf: theory/sources/papers/liang2022-helm.pdf
type: excerpts
---

# Excerpts — Liang et al. 2022, "HELM"

## Abstract {#abstract}

> Language models (LMs) are becoming the foundation for almost all major language tech-
> nologies, but their capabilities, limitations, and risks are not well understood. We present
> Holistic Evaluation of Language Models (HELM) to improve the transparency of language
> models. First, we taxonomize the vast space of potential scenarios (i.e. use cases) and met-
> rics (i.e. desiderata) that are of interest for LMs. Then we select a broad subset based on
> coverage and feasibility, noting what's missing or underrepresented (e.g. question answering
> for neglected English dialects, metrics for trustworthiness). Second, we adopt a multi-metric
> approach: We measure 7 metrics (accuracy, calibration, robustness, fairness, bias, toxicity,
> and efficiency) for each of 16 core scenarios to the extent possible (87.5% of the time), en-
> suring that metrics beyond accuracy don't fall to the wayside, and that trade-offs across
> models and metrics are clearly exposed. We also perform 7 targeted evaluations, based on
> 26 targeted scenarios, to more deeply analyze specific aspects (e.g. knowledge, reasoning,
> memorization/copyright, disinformation). Third, we conduct a large-scale evaluation of 30
> prominent language models (spanning open, limited-access, and closed models) on all 42 sce-
> narios, including 21 scenarios that were not previously used in mainstream LM evaluation.
> Prior to HELM, models on average were evaluated on just 17.9% of the core HELM sce-
> narios, with some prominent models not sharing a single scenario in common. We improve
> this to 96.0%: now all 30 models have been densely benchmarked on a set of core scenarios
> and metrics under standardized conditions. Our evaluation surfaces 25 top-level findings
> concerning the interplay between different scenarios, metrics, and models. For full trans-
> parency, we release all raw model prompts and completions publicly for further analysis,
> as well as a general modular toolkit for easily adding new scenarios, models, metrics, and
> prompting strategies. We intend for HELM to be a living benchmark for the community,
> continuously updated with new scenarios, metrics, and models.

Abstract verified verbatim from PDF (arXiv:2211.09110v2, published TMLR 08/2023).

## §1.1 — The 7-metric, 16-scenario design {#sec-1-1-design}

> HELM currently implements a core set of 16 scenarios and 7 (categories
> of) metrics. Our scenarios, which are triples of (task, domain, language), span 6 user-facing
> tasks (e.g. question answering, information retrieval, summarization, toxicity detection),
> several domains (e.g. news, books), and currently only English (though we cover several
> English varieties such as African-American English and the English varieties spoken in
> different English-speaking countries). And our 7 categories of metrics reflect a range of
> societal considerations (i.e. accuracy, calibration, robustness, fairness, bias, toxicity,
> efficiency). We emphasize that while we have specific quantitative metrics for all of these
> considerations, they (e.g. fairness) are complex and contested social constructs that can be
> operationalized in many different ways.

Each scenario is a `(task, domain, language)` triple; each run is a `(scenario, adaptation method,
metric)` triple. The benchmark measures 98 of 112 possible (core scenario, metric) pairs (87.5%),
making explicit what is missing.

## §4.3–§4.5 — Metric definitions (Accuracy, Calibration, Robustness) {#sec-4-metrics}

> Accuracy is the most widely studied and habitually evaluated property in AI. Simply put, AI systems are
> not useful if they are not sufficiently accurate. Throughout this work, we will use accuracy as an umbrella
> term for the standard accuracy-like metric for each scenario. This refers to the exact-match accuracy in text
> classification, the F1 score for word overlap in question answering, the MRR and NDCG scores for information
> retrieval, and the ROUGE score for summarization, among others.

> When machine learning models are integrated into broader systems, it is critical for these models to be simul-
> taneously accurate (i.e. frequently correct) and able to express their uncertainty (so that their errors can be
> appropriately anticipated and accommodated). ... To quantify calibration, we compute the
> expected calibration error (ECE; Naeini et al., 2015; Guo et al., 2017), which measures the difference between
> the model's predicted probability and the fraction of times the model is correct. By default, we use 10-bins
> with an equal number of probabilities per bin.

> Towards this goal, we measure the robustness of different models by evaluating them on transformations
> of an instance. That is, given a set of transformations for a given instance, we measure the worst-case
> performance of a model across these transformations. ... we restrict ourselves to perturbations that
> are both natural and relatively mild—e.g., capitalization, common misspellings.

The remaining four metric categories (Fairness, Bias, Toxicity, Efficiency) follow the same
pattern: each is operationalized quantitatively for the same 16 core scenarios, not relegated
to separate bespoke datasets.

## §1.2 — Headline cross-model findings {#sec-1-2-findings}

> 1. The benefits of instruction-tuning. Across the core scenarios, we find that text-davinci-002
>    performs best on our accuracy, robustness, and fairness metrics, with Anthropic-LM v4-s3 (52B)
>    being in the top 3 for all 3 metrics (despite being more than 10× smaller in model scale compared
>    to TNLG v2 (530B), which is the second most accurate and fair) ... Given the very strong
>    performance of both models, and that they are the only instruction-tuned models we evaluate
>    (beyond the much smaller OpenAI model variants), this suggests instruction-tuning provides a
>    broad set of advantages.

> 3. Calibration. We observe that the relationship between accuracy and calibration depends on the
>    scenario and adaptation procedure. As an example, for HellaSwag, improving accuracy worsens
>    calibration, whereas for OpenBookQA, improving accuracy improves calibration.

> 4. Robustness and fairness perturbations. Across all scenarios, we observe strong correlations
>    between accuracy, robustness, and fairness, where robustness and fairness metrics consider worst-case
>    accuracy over a set of perturbations (e.g. typos for robustness, dialect alteration for fairness). ...
>    We also see serious drops in some cases: for example, on NarrativeQA, TNLG v2 (530B) precipitously
>    drops from 72.6% standard accuracy (i.e. the third-most accurate model) to 38.9% accuracy in the
>    presence of robustness perturbations.

The structural finding: no single model dominates all 7 metric axes. Instruction-tuning
(text-davinci-002, Anthropic-LM v4-s3) creates broad multi-axis advantage over raw scale
alone, but robustness and calibration tradeoffs persist even for the strongest models.

## §1.2 — Coverage gap and standardization {#sec-1-2-coverage}

> Prior to our effort, on average models were evaluated on 17.9% of our core scenarios, even
> after compiling evaluations dispersed across different prior works. We improve this to 96.0%.

> For full transparency, we release all raw model prompts and completions publicly for further
> analysis, as well as a general modular toolkit for easily adding new scenarios, models, metrics,
> and prompting strategies.

The comparability contribution: every model evaluated under identical few-shot prompting
conditions on the same scenarios. The (prompt, response) corpus is released so scores can
be re-derived under tighter contamination rules without re-running models.

[Verified from PDF on 2026-05-12] Added §1.1 design, §4.3–4.5 metric definitions, §1.2 findings, §1.2 coverage. Abstract replaced with full verbatim text from PDF.
