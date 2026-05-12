---
paper_key: contamination-survey-2025
title: "Benchmarking LLMs Under Data Contamination: From Static to Dynamic Evaluation"
authors: (et al., EMNLP 2025 — see arXiv listing)
year: 2025
venue: EMNLP 2025
arxiv: 2502.17521
local_pdf: theory/sources/papers/contamination-survey-2025.pdf
type: excerpts
note: PDF download blocked (curl/wget restricted by sandbox in this Phase 2.5 deepening pass); full abstract retrieved verbatim via WebFetch on arXiv 2502.17521 abstract page (2026-05-05). Detailed contamination taxonomy not available without PDF access — `[PDF-VERIFY]` flagged where needed.
---

# Excerpts — Contamination Survey 2025

## #abstract — full verbatim

> Data contamination has received increasing attention in the era of
> large language models (LLMs) due to their reliance on vast Internet-
> derived training corpora. To mitigate the risk of potential data
> contamination, LLM benchmarking has undergone a transformation from
> static to dynamic benchmarking. In this work, we conduct an in-depth
> analysis of existing static to dynamic benchmarking methods aimed at
> reducing data contamination risks. We first examine methods that
> enhance static benchmarks and identify their inherent limitations. We
> then highlight a critical gap--the lack of standardized criteria for
> evaluating dynamic benchmarks. Based on this observation, we propose
> a series of optimal design principles for dynamic benchmarking and
> analyze the limitations of existing dynamic benchmarks. This survey
> provides a concise yet comprehensive overview of recent advancements
> in data contamination research, offering valuable insights and a
> clear guide for future research efforts. We maintain a GitHub
> repository to continuously collect both static and dynamic
> benchmarking methods for LLMs.

(Verified verbatim from arXiv 2502.17521 abstract via WebFetch
2026-05-05.)

## Methodological scope {#sec-scope}

The survey's three structural moves (paraphrasing the abstract, with
verbatim quotes for load-bearing parts):

1. **Examine static-benchmark enhancements** ("methods that enhance
   static benchmarks") — these include rewrites/paraphrases of MMLU,
   contaminant-detection auditing, etc. — and identify "their inherent
   limitations."
2. **Highlight the dynamic-benchmark evaluation gap** — "the lack of
   standardized criteria for evaluating dynamic benchmarks." This is
   the methodological frontier as of EMNLP 2025.
3. **Propose optimal design principles** for dynamic benchmarking.

`[PDF-VERIFY]` The detailed taxonomy of contamination types (input
contamination, label contamination, indirect contamination,
generative contamination) is referenced in subsequent literature but
not enumerated in this paper's abstract; needs PDF read for the
canonical taxonomy. The MathArena (arXiv 2505.23281) and LiveCodeBench
projects are the canonical examples of "dynamic benchmarking" — both
append post-cutoff problems to defeat training-data leakage.
