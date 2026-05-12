---
paper_key: contamination-survey-2025
title: "Benchmarking Large Language Models Under Data Contamination: A Survey from Static to Dynamic Evaluation"
authors: Chen, Chen, Li, Jiang, Wan, He, Ran, Gu, Li, Xie, Ray
year: 2025
venue: EMNLP 2025
arxiv: 2502.17521
local_pdf: theory/sources/papers/contamination-survey-2025.pdf
type: excerpts
---

# Excerpts — Contamination Survey 2025

## Abstract {#abstract}

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

Abstract verified verbatim from PDF (matches arXiv 2502.17521 abstract verified
2026-05-05). Title in PDF is "Benchmarking Large Language Models Under Data
Contamination: A Survey from Static to Dynamic Evaluation" — updated in front matter.

## §2.1 Contamination taxonomy — exact and syntactic types {#sec-taxonomy}

> Data contamination arises when LLM training data Dtrain improperly overlaps with
> evaluation data Dtest, undermining performance validity.

> Exact Contamination. Exact contamination occurs when there is any exact duplicate
> in the benchmark dataset
>
>     ∃d  s.t.  d ∈ Dtrain and d ∈ Dtest
>
> In other words, there exists a data point d that is in both Dtrain and Dtest. Common
> cases include verbatim test examples appearing in training corpora, code snippets
> from benchmark implementations, or documentation leaks.

> Syntactic Contamination. Syntactic contamination occurs when a test data point
> could be found in the training dataset after a syntactic transformation, such that
>
>     ∃d  s.t.  Fsyntactic(d) ∈ Dtrain and d ∈ Dtest
>
> where Fsyntactic denotes syntactic transformations like punctuation normalization,
> whitespace modification, synonym substitution, morphological variations, or syntactic
> paraphrasing while preserving lexical meaning.

> Whether such syntactic contamination constitutes true contamination is debated, as it
> is difficult to separate memorization from reasoning. In this work, we treat such
> transformations as contamination, since some NLP tasks rely heavily on syntax.

The two-type taxonomy (exact and syntactic) is the paper's formal backbone. The
paper's Figure 2 taxonomy tree also enumerates contamination sources across
pre-training, post-training, and fine-tuning phases, but the formal definitions are
restricted to these two structural types.

## §2.2 Contamination sources — pre-training, post-training, fine-tuning {#sec-sources}

> Data contamination can occur during the pre-training, post-training, or fine-tuning
> phases of LLM development. Unlike traditional models with clear separations between
> training and evaluation data, LLMs are pre-trained on massive, diverse datasets—often
> scraped from the web (e.g., FineWeb (Penedo et al., 2024)), increasing the risk of
> evaluation data overlap.

> Additionally, many LLMs keep their training data proprietary (Dubey et al., 2024;
> Yang et al., 2024), complicating the accurate assessment of their true performance and
> highlighting the need for fair and reliable benchmarks. This opacity further
> exacerbates data contamination, as it impedes the community's ability to verify and
> mitigate potential overlaps between training and evaluation data.

> The issue gained public attention when Meta's LLaMA 4 faced allegations of using a
> non-public version fine-tuned for benchmark gains (Babic, 2025), raising concerns
> about evaluation transparency—despite Meta's denial of test set exposure.

[INTUITION] The three-phase source model (pre-train / post-train / fine-tune) matters
because each phase has different mitigation levers: timestamp filtering addresses
pre-training overlap; label-protection and encryption address post-training; detector
tools address fine-tuning. No single mitigation covers all three.

## §3 Static-benchmark post-hoc detection methods {#sec-detection}

The paper's taxonomy tree (Figure 2) classifies post-hoc detection under
"Methods for Mitigation" for static benchmarks. Verbatim from Figure 2 caption
and adjacent text:

> Post-hoc Detection: Direct overlap detection (Touvron et al., 2023; Radford et al.,
> 2019; Brown et al., 2020; ...), memorization through masked inputs (Ranaldi et al.,
> 2024; Chang et al., 2023), partial completions (Anil et al., 2023; Golchin and
> Surdeanu, 2024), preference for original over paraphrased test cases (Duarte et al.,
> 2024; Golchin and Surdeanu, 2023; Zong et al., 2024).

The four post-hoc detection families are:
1. **Direct overlap detection** — n-gram or string-matching between training corpus
   and test set (requires training data access).
2. **Memorization via masked inputs** — probe model completion of masked test
   examples to detect verbatim memorization.
3. **Partial completions** — feed prefix of test item and check if model completes the
   known answer at above-chance rate.
4. **Preference for original over paraphrase** — contaminated models prefer the
   verbatim original over a semantically equivalent paraphrase.

[INTUITION] These four families span a gradient of required access: (1) requires
training data; (2)–(4) require only API access to the model. The preference test (4) is
the most widely deployable in practice.

## §6 Headline conclusion — static methods' vulnerability and dynamic methods' gaps {#sec-conclusion}

> This survey reviews the literature on data contamination in LLM benchmarking,
> analyzing both static and dynamic methods. We find that static methods, though
> consistent, become more vulnerable to contamination as training datasets grow. While
> dynamic methods show promise, they face challenges in reliability and
> reproducibility. Future research should focus on standardized dynamic evaluation,
> and practical mitigation tools.

[INTUITION] The paper does not report a single headline contamination prevalence rate
across surveyed models — it is a methodology survey, not an empirical measurement
study. The closest prevalence claim is the Schaeffer (2023) citation in §2.1: "pretraining
on test data can significantly distort evaluation outcomes," and the Balloccu et al. (2024)
finding on "closed-source LLMs" — both cited rather than independently measured.
For concrete prevalence numbers, see primary detection studies (Shi et al. 2024,
Golchin & Surdeanu 2024) rather than this survey.

[Verified from PDF on 2026-05-12] Added §sec-taxonomy (formal exact/syntactic
definitions with math), §sec-sources (three-phase contamination source model),
§sec-detection (four post-hoc detection families), §sec-conclusion (headline findings).
Abstract verified verbatim from PDF. Title corrected in front matter.
