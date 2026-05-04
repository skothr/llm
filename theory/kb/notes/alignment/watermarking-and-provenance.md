---
topic: alignment/watermarking-and-provenance
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - kirchenbauer2023-watermark
  - kirchenbauer2024-reliability
  - dathathri2024-synthid
secondary_sources: []
related_topics:
  - alignment/safety-evaluation
  - inference/structured-output
---

# Watermarking and provenance

**Watermarking** of LLM-generated text is the embedding of a
statistical signal — invisible to a casual human reader, but detectable
algorithmically — into the token stream produced by a sampling-time
intervention. **Provenance** is the broader project of certifying the
origin of generated content; watermarking is the technical primitive
most-discussed for text. (Image / audio provenance via C2PA is more
mature and is out-of-scope for this LLM-focused note.)

The canonical text-watermarking lineage runs from Kirchenbauer et al.
2023 (green/red list logit bias) through reliability follow-ups
(2024) to **SynthID-Text** (Dathathri et al. 2024, Nature; deployed in
the Gemini App). The frontier debate is whether any watermark
detectable from short spans can survive a paraphrase by a comparable-
quality model.

## 1. The canonical scheme — Kirchenbauer 2023

### 1.1 Construction

`kirchenbauer2023-watermark`
`[kirchenbauer2023-watermark §greenred;
kb/excerpts/kirchenbauer2023-watermark#sec-greenred]`:

For each generation step $t$:

1. Hash the previous token $s^{(t-1)}$ (or a longer suffix) via a
   keyed PRF to seed an RNG.
2. Use the RNG to partition the vocabulary $V$ into a **green list**
   $G_t \subset V$ of size $\gamma |V|$ and a **red list**
   $V \setminus G_t$ of size $(1-\gamma)|V|$. Default $\gamma = 0.25$.
3. Add a fixed bias $\delta > 0$ to the logits of green tokens before
   softmax. Default $\delta = 2.0$.
4. Sample from the modified distribution.

The construction is **stateless w.r.t. earlier history beyond the
hash window** — the green list at step $t$ depends only on
$s^{(t-1)}$ (or a small window) and the secret key. This makes both
generation and detection cheap.

### 1.2 Detection — the z-score statistic

Given a candidate text of $T$ tokens with $|s|_G$ green-listed tokens,
under the null (text not watermarked) each token is in $G$
independently with probability $\gamma$:

$$z = \frac{|s|_G - \gamma T}{\sqrt{T \gamma (1-\gamma)}}$$

A large $z$-score rejects the null
`[kirchenbauer2023-watermark §detection;
kb/excerpts/kirchenbauer2023-watermark#sec-detection]`. The headline
property: **detection requires no API access, no model weights, only
the keyed hash and the candidate text**.

### 1.3 Why "soft" promotion

A hard partition (force $G_t$-only generation) would (i) heavily
distort high-entropy positions, (ii) be impossible for low-entropy
positions where one specific token dominates, and (iii) be trivially
detectable by frequency analysis even without the key. The soft
$\delta$-additive scheme inherits a **distortion / detectability
tradeoff** — larger $\delta$ → stronger watermark, more quality drift
`[kirchenbauer2023-watermark §soft;
kb/excerpts/kirchenbauer2023-watermark#sec-soft]`.

[INTUITION] The watermark is "free" at high-entropy positions
(many roughly-equivalent tokens, gentle nudge to green list barely
changes output) and "expensive" at low-entropy positions (one
correct answer, nudging away from it lowers quality). Factual
generation, code, and direct citation are inherently
unwatermarkable; prose and creative writing are. The canonical
form is the entropy-weighted average of $\delta$'s effect.

## 2. Reliability under attack

### 2.1 Paraphrase robustness — Kirchenbauer 2024

`kirchenbauer2024-reliability` (`arxiv 2306.04634`, ICLR 2024)
addresses three attack scenarios: human rewriting, non-watermarked
LLM paraphrasing, and watermarked-span embedded in longer document.

The headline numerical claim: even after **strong human paraphrasing**,
the watermark is detectable after observing **800 tokens on average,
at a $10^{-5}$ false-positive rate**. The mechanism is preservation of
short n-grams: paraphrases preserve enough n-gram structure that the
green-list test still has signal, just at a slower accumulation rate.

### 2.2 The fundamental limit

[CONTRADICTION] Sadasivan et al. 2023 ("Can AI-Generated Text be
Reliably Detected?") argue that any watermark detectable from short
spans **must** be removable by a paraphrase attacker with a
comparable-quality model. Kirchenbauer 2024 agrees the watermark
**degrades** under heavy paraphrase but argues it remains detectable
*given enough text*. The two positions are not numerically far apart;
they differ on what "deployable" requires:

| Question | Sadasivan 2023 view | Kirchenbauer 2024 view |
|---|---|---|
| Can short spans be reliably detected? | No | Yes for moderate paraphrase, degraded for strong |
| Can long passages be detected? | Difficult under strong attack | Yes, ~800 tokens at FPR $10^{-5}$ |
| Is text watermarking *deployable* for casual use? | No | Yes for long-form |
| Is it *deployable* for adversarial use? | No | Marginal |

This is an active 2024–2026 debate.

## 3. SynthID-Text (Dathathri et al. 2024, Nature)

### 3.1 Tournament sampling

[Note: PDF download blocked; the description below combines the
landing page summary, DeepMind's blog post, and widely-cited
secondary sources. Verbatim section excerpts pending PDF access.]

SynthID-Text replaces the logit-bias step of Kirchenbauer 2023 with a
**tournament** over candidate tokens. Sketch:

1. At each step, sample $N$ candidate next-tokens from the
   distribution.
2. Pair them up; for each pair, evaluate a hash-derived
   **watermark function** $g$ on each candidate; advance the
   candidate with higher $g$ value.
3. Repeat for $\log_2 N$ rounds until a single token wins.

The detection statistic averages $g$-values across the generated
tokens; high values indicate the tournament was applied.

### 3.2 Distortionary vs non-distortionary modes

SynthID-Text is described in two modes:

- **Distortionary** — biases the output distribution measurably;
  stronger watermark.
- **Non-distortionary** — under specific assumptions, the marginal
  output distribution is preserved (the watermark is detectable but
  the distribution of generated text is statistically identical to
  the unwatermarked sampler).

[SPECULATION] The non-distortionary mode is the headline novelty
relative to Kirchenbauer 2023. Whether the assumption (sufficient
candidate diversity at each step) holds for low-entropy generation
is the same fundamental limit Kirchenbauer 2023 hits.

### 3.3 Production deployment

SynthID-Text is reported as deployed in the Gemini App (Google
DeepMind blog, 2024). This is the first production-scale text
watermark deployment by a frontier lab. Open-source release of
detection code accompanies the Nature publication.

[CONTRADICTION] Independent robustness analysis by SRI Lab ETH
("Probing Google DeepMind's SynthID-Text Watermark", 2024) finds the
watermark survives mild paraphrase but degrades under aggressive
attacks. Treat the deployed SynthID as **discovery-grade provenance**
(useful for stat-significant detection on long-form, defeated on
adversarial short-form), not as a hard authentication.

## 4. The provenance landscape — what watermarking is for

| Use case | Suitable? | Why |
|---|---|---|
| Detect AI-generated student essays | Marginal | Short text, possibly paraphrased; high false-positive cost |
| Train future LLMs without ingesting prior LLM output | Yes (long-form) | At training scale, even slow per-passage detection accumulates |
| Forensic attribution of a specific generated artifact | Marginal | Adversary will paraphrase; legal standards uncertain |
| Compliance with EU AI Act Art. 50 transparency | Partial | The Act requires "marking" of synthetic content; SynthID qualifies as a marking but enforcement is open |

## 5. Frontier and open questions (as of 2026-05)

- **Is paraphrase-robust watermarking achievable at all?** The
  Sadasivan vs Kirchenbauer debate is unresolved. New schemes
  (topic-conditioned watermarks, cross-text watermarks) trade off
  detectability and robustness in different regions, but no scheme
  reliably survives strong paraphrase by a comparable-quality model
  in short text.
- **Composition with structured output.** Constrained decoding
  (`inference/structured-output`, e.g., XGrammar) restricts the
  output distribution to grammar-valid tokens. Composing watermarking
  with structured output requires the green list to be intersected
  with the grammar's valid-next-token set. If the intersection is
  empty, watermarking degrades to no-op. No published study addresses
  this composition rigorously.
- **Reasoning-model regime.** Long CoT before a short answer means
  the answer (the user-visible content) is a small fraction of total
  generation. Watermarking the CoT may be high-bandwidth; watermarking
  the answer may be unworkable. How to allocate the watermark budget
  in reasoning models is open.
- **Cross-vendor standardization.** Each frontier lab's watermark is
  detector-locked to the lab's secret key; the EU AI Act requires
  *marking* but not interoperability. A C2PA-style cross-vendor
  standard for text does not exist.
- **Adversarial pressure from open-source models.** Open-weights
  models cannot enforce watermarking at inference (an end user can
  disable it). Watermarking is an API-only intervention; its coverage
  caps at the closed-source share of generated text. As open weights
  catch up to frontier closed-weights, watermark coverage shrinks.

## 6. See also

- `kb/notes/alignment/safety-evaluation.md` — provenance is one of
  several deployment-side controls; safety evals provide the others.
- `kb/notes/inference/structured-output.md` — constrained decoding
  composition with watermarking.
- `kb/notes/post-training/rlhf.md` — independent of watermarking, but
  the entropy-conditional argument (high-entropy positions are
  cheap to watermark) interacts with how RLHF tightens the output
  distribution.
