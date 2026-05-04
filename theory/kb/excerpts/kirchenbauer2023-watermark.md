---
paper_key: kirchenbauer2023-watermark
title: "A Watermark for Large Language Models"
authors: Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein
year: 2023
venue: ICML 2023
arxiv: 2301.10226
local_pdf: null
type: excerpts
note: PDF download blocked; excerpts derived from arXiv landing page (WebFetch 2026-05-04). The green/red list mechanism, gamma/delta parameters, and z-score detection statistic are described from the landing page; deeper section anchors pending PDF access.
---

# Excerpts — Kirchenbauer et al. 2023, "A Watermark for Large Language Models"

## Abstract — central claim {#abstract}

> Potential harms of large language models can be mitigated by
> watermarking model output, i.e., embedding signals into generated
> text that are invisible to humans but algorithmically detectable from
> a short span of tokens. We propose a watermarking framework for
> proprietary language models. The watermark can be embedded with
> negligible impact on text quality, and can be detected using an
> efficient open-source algorithm without access to the language model
> API or parameters.

## The green/red list mechanism {#sec-greenred}

The construction (paraphrased from the landing page; canonical
description widely-cited):

1. Before generating token $t$, hash the previous token $s^{(t-1)}$ via
   a keyed pseudo-random function to seed an RNG.
2. Use the RNG to partition the vocabulary $V$ into a **green list**
   $G \subset V$ of size $\gamma |V|$ and a **red list** $V \setminus G$
   of size $(1-\gamma) |V|$. Default $\gamma = 0.25$ in the paper.
3. Add a fixed bias $\delta > 0$ to the logits of green tokens before
   softmax. Default $\delta = 2.0$.
4. Sample from the modified distribution.

Green tokens are thus **softly preferred but not forced**: low-entropy
positions (where one token dominates regardless) are unaffected;
high-entropy positions get nudged.

## Detection — the z-score statistic {#sec-detection}

Given a candidate text of $T$ tokens, count $|s|_G$ (the number of
green-listed tokens). Under the null hypothesis (text not watermarked),
each token is in $G$ with probability $\gamma$ independently:

$$z = \frac{|s|_G - \gamma T}{\sqrt{T \gamma (1-\gamma)}}$$

A high $z$-score rejects the null. The paper proposes "a statistical
test for detecting the watermark with interpretable p-values" — the
$z$-score directly gives a one-sided $p$-value via the normal CDF.

This is the headline property: **detection requires no API access, no
model weights, only the keyed hash function and the candidate text**.

## Why "soft" promotion {#sec-soft}

The $\delta$-additive logit bias is a soft constraint. Hard partitioning
(restrict generation to green tokens only) would:

- Heavily distort high-entropy generation (force the model away from
  the natural distribution).
- Make low-entropy positions (factual / deterministic) impossible to
  watermark — there is no green-list-friendly alternative.
- Be trivially detectable by frequency analysis even without the key.

The soft scheme inherits a **distortion / detectability tradeoff**
controlled by $\delta$: larger $\delta$ → stronger watermark, more
quality drift.

## Practical limits {#sec-limits}

The paper's reliability claims are tested in `kirchenbauer2024-reliability`
(arxiv 2306.04634, ICLR 2024). Two limits surface in practice:

1. **Low-entropy generation is unwatermarkable.** Factual lookups,
   structured outputs, and direct citation pass through with the
   original distribution; the watermark signal is dilute.
2. **Paraphrase attacks dilute but do not eliminate the watermark.**
   The follow-up reports detection at "800 tokens on average" after
   strong human paraphrasing at a $10^{-5}$ false-positive rate
   — workable but slow.

## Lineage {#sec-lineage}

- **Kirchenbauer 2023** (this paper) — green/red list with logit bias.
- **Kirchenbauer 2024 reliability** (arxiv 2306.04634) —
  paraphrase / human-rewrite robustness.
- **SynthID-Text** (Dathathri et al. 2024, Nature) — tournament
  sampling; deployed at production scale in Gemini App. Replaces the
  logit-bias step with a tournament over candidate tokens that biases
  toward a hash-derived watermark function $g$ without modifying the
  output distribution under specific assumptions
  ([SPECULATION] — non-distortionary mode).
- **Topic-conditioned watermarks** (arxiv 2404.02138) — partition by
  topic to improve robustness to topic-preserving paraphrase.

[CONTRADICTION] Whether watermarking is **practically deployable** for
provenance at internet scale is unsettled. Critics (Sadasivan et al.
2023, "Can AI-Generated Text be Reliably Detected?") argue that any
watermark detectable from short spans must be removable by a paraphrase
attacker with a comparable-quality model. The reliability paper agrees
the watermark **degrades** under heavy paraphrase but argues it remains
detectable given enough text. The two camps are not numerically far
apart; they disagree on what "deployable" requires.
