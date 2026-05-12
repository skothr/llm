---
paper_key: harmbench2024
title: "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal"
authors: Mazeika, Phan, Yin, Zou, Wang, Mu, Sakhaee, Li, Basart, Li, Forsyth, Hendrycks
year: 2024
venue: arXiv 2402.04249; ICML 2024 Workshop
arxiv: 2402.04249
local_pdf: theory/sources/papers/harmbench2024.pdf
type: excerpts
note: PDF not yet downloaded; excerpts from arXiv 2402.04249 abstract (verified via WebSearch 2026-05-04). The standardized red-team evaluation framework. Used in `kb/notes/alignment/safety-evaluation.md` §2.1.
---

# Excerpts — Mazeika et al. 2024, "HarmBench"

## Abstract — full text {#abstract}

> Automated red teaming holds substantial promise for uncovering and
> mitigating the risks associated with the malicious use of large
> language models (LLMs), yet the field lacks a standardized
> evaluation framework to rigorously assess new methods. To address
> this issue, we introduce HarmBench, a standardized evaluation
> framework for automated red teaming. We identify several desirable
> properties previously unaccounted for in red teaming evaluations
> and systematically design HarmBench to meet these criteria. Using
> HarmBench, we conduct a large-scale comparison of 18 red teaming
> methods and 33 target LLMs and defenses, yielding novel insights.
> We also introduce a highly efficient adversarial training method
> that greatly enhances LLM robustness across a wide range of
> attacks, demonstrating how HarmBench enables codevelopment of
> attacks and defenses.

(Verified via WebSearch 2026-05-04 against arXiv 2402.04249 abstract.)

## Structure {#sec-structure}

The framework includes:

- **510 carefully curated behaviors** spanning four functional
  categories: standard text, contextual, copyright (text-only),
  multimodal.
- **18 red-team attack methods** evaluated head-to-head: GCG,
  AutoDAN, PAIR, TAP, GBDA, AutoPrompt, etc.
- **33 target LLMs and defenses** evaluated.
- A fine-tuned **HarmBench-Judge** classifier (Llama-2-13B family,
  fine-tuned on labeled harmful/non-harmful generations) used as
  the standardized judge.

GitHub: https://github.com/centerforaisafety/HarmBench

## Methodological contribution

HarmBench is designed for **co-development of attacks and defenses**:
the same suite is the substrate for measuring (a) how good a new
attack is, (b) how robust a new defense is, (c) head-to-head
comparison of either dimension. Prior to HarmBench, jailbreak papers
reported numbers on incompatible behavior subsets with incompatible
judges, making cross-paper comparison impossible. HarmBench's fixed
behavior set + fixed judge stack is the answer.

[PDF-VERIFY] The full attack-method bake-off results, the per-defense
robustness numbers, and the efficient-adversarial-training method
details require PDF verification before propagating as hard claims.
The structural commitments above are sufficient to ground the §2.1
references in `kb/notes/alignment/safety-evaluation.md`.

## Adoption

HarmBench is widely cited in 2024-2026 model-card safety sections and
is used by UK AISI / US AISI as a substrate for pre-deployment
evaluation `[FORUM-SIGNAL: Phase 1 sweep §safety-evaluation; AISI
public reports]`.
