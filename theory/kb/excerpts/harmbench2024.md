---
paper_key: harmbench2024
title: "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal"
authors: Mazeika, Phan, Yin, Zou, Wang, Mu, Sakhaee, Li, Basart, Li, Forsyth, Hendrycks
year: 2024
venue: arXiv 2402.04249; ICML 2024 Workshop
arxiv: 2402.04249
local_pdf: theory/sources/papers/harmbench2024.pdf
type: excerpts
---

# Excerpts — Mazeika et al. 2024, "HarmBench"

## Abstract {#abstract}

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

(Verified verbatim from PDF, arXiv:2402.04249v2.)

## §3.1 ASR Metric Definition {#sec-3-1-asr}

> Following Perez et al. (2022); Zou et al. (2023), we formulate the red teaming task as designing test cases
> {x₁, x₂, ..., xN} in order to elicit a given behavior y from one or more target LLMs.
>
> The primary measure of a red teaming method's success is its attack success rate (ASR) on a given target model,
> which is the percentage of test cases that elicit the behavior from the target model.

Formally, let $f$ be a target model, $g$ a red teaming method, and $c$ a classifier mapping completion $x'$ and behavior $y$ to $\{0,1\}$. The ASR is:

$$\text{ASR}(y, g, f) = \frac{1}{N} \sum_{i} c(f_T(x_i), y)$$

(Eq. as typeset in §3.1 of the PDF.) The classifier $c$ is the HarmBench-Judge (Llama-2-13B fine-tuned on
labeled completions); using a held-out classifier prevents the metric from being gamed by the attack
method it is meant to evaluate.

## §4.1 Behavior Taxonomy — 510 Behaviors, 4 Functional Categories {#sec-4-1-behaviors}

> HarmBench contains 510 unique harmful behaviors, split into 400 textual behaviors and 110
> multimodal behaviors. We designed the behaviors to violate laws or norms, such that most reasonable people would not
> want a publicly available LLM to exhibit them.

> Functional categories. HarmBench contains the following 4 functional categories of behavior: standard behaviors,
> copyright behaviors, contextual behaviors, and multimodal behaviors. These categories contain 200, 100, 100, and 110
> behaviors, respectively.

The functional split is load-bearing for attack-defense comparability: contextual and multimodal
behaviors require providing highly specific accompanying context or images, exposing attack methods
that rely only on text-only prompt injection. Standard behaviors (200) are the column that most
directly maps to prior AdvBench results.

## §6.1 Headline Finding — No Uniform Attack or Defense {#sec-6-1-no-universal}

> Notably, no current attack or defense is uniformly effective. All attacks have low ASR on at least one LLM,
> and all LLMs have poor robustness against at least one attack. This illustrates the importance of running
> large-scale standardized comparisons, which are enabled by HarmBench.

> Our large-scale comparison reveals several interesting properties that revise existing findings and
> assumptions from prior work. Namely, we find that no current attack or defense is uniformly effective
> and robustness is independent of model size.

The "robustness is independent of model size" corollary is a direct revision of the intuition that
scaling alone improves safety — the finding implies that training data and alignment procedures
dominate over raw parameter count for attack resistance.

[Verified from PDF on 2026-05-12] Added §3.1-asr, §4.1-behaviors, §6.1-no-universal. Abstract verified verbatim.
