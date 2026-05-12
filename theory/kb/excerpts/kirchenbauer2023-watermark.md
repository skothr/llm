---
paper_key: kirchenbauer2023-watermark
title: "A Watermark for Large Language Models"
authors: Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein
year: 2023
venue: ICML 2023
arxiv: 2301.10226
local_pdf: theory/sources/papers/kirchenbauer2023-watermark.pdf
type: excerpts
---

# Excerpts — Kirchenbauer et al. 2023, "A Watermark for Large Language Models"

## Abstract {#abstract}

> Potential harms of large language models can be mitigated by watermarking model
> output, i.e., embedding signals into generated text that are invisible to humans but
> algorithmically detectable from a short span of tokens. We propose a watermarking
> framework for proprietary language models. The watermark can be embedded with
> negligible impact on text quality, and can be detected using an efficient open-source
> algorithm without access to the language model API or parameters. The watermark
> works by selecting a randomized set of "green" tokens before a word is generated,
> and then softly promoting use of green tokens during sampling. We propose a
> statistical test for detecting the watermark with interpretable p-values, and derive an
> information-theoretic framework for analyzing the sensitivity of the watermark. We
> test the watermark using a multi-billion parameter model from the Open Pretrained
> Transformer (OPT) family, and discuss robustness and security.

Abstract verified verbatim from PDF.

## §2 / §3 — Algorithm 1 (hard) and Algorithm 2 (soft) — the green/red-list construction {#sec-algorithm}

Algorithm 1 (hard red list) — verbatim from the paper:

> Algorithm 1 Text Generation with Hard Red List
>
>   Input: prompt, s^(−Np) ··· s^(−1)
>   for t = 0, 1, ··· do
>     1. Apply the language model to prior tokens s^(−Np) ··· s^(t−1) to get a
>        probability vector p^(t) over the vocabulary.
>     2. Compute a hash of token s^(t−1), and use it to seed a random number
>        generator.
>     3. Using this seed, randomly partition the vocabulary into a "green list" G and
>        a "red list" R of equal size.
>     4. Sample s^(t) from G, never generating any token in the red list.
>   end for

Algorithm 2 (soft red list) — input parameters and step 4 verbatim:

> Algorithm 2 Text Generation with Soft Red List
>
>   Input: prompt, s^(−Np) ··· s^(−1)
>          green list size, γ ∈ (0, 1)
>          hardness parameter, δ > 0
>   for t = 0, 1, ··· do
>     1. Apply the language model to prior tokens ... to get a logit vector l^(t)
>        over the vocabulary.
>     2. Compute a hash of token s^(t−1), and use it to seed a random number
>        generator.
>     3. Using this random number generator, randomly partition the vocabulary into
>        a "green list" G of size γ|V|, and a "red list" R of size (1 − γ)|V|.
>     4. Add δ to each green list logit. Apply the softmax operator to these modified
>        logits to get a probability distribution over the vocabulary.
>     5. Sample the next token, s^(t), using the watermarked distribution p̂^(t).
>   end for

> Rather than strictly prohibiting the red list tokens, Algorithm 2 adds a constant δ
> to the logits of the green list tokens. The soft red list rule adaptively enforces the
> watermark in situations where doing so will have little impact on quality, while
> almost ignoring the watermark rule in the low entropy case where there is a clear
> and unique choice of the "best" word.

The primary experimental configuration in the paper uses γ = 0.5, δ = 2.0
(Table 1, Table 2, Figure 6 unattacked baseline). The earlier excerpt's claim of
γ=0.25 as "default" was incorrect — γ=0.25 appears in some parameter sweeps but
γ=0.5 is the main-text running example.

## §3.1 — The z-score detection statistic {#sec-zscore}

> A more robust detection approach uses a one proportion z-test to evaluate the null
> hypothesis. If the null hypothesis is true, then the number of green list tokens,
> denoted |s|_G, has expected value T/2 and variance T/4. The z-statistic for this
> test is
>
>     z = 2(|s|_G − T/2) / √T.                                                 (2)
>
> We reject the null hypothesis and detect the watermark if z is above a chosen
> threshold.

For the soft red list with arbitrary γ, the general z-statistic is:

$$z = \frac{|s|_G - \gamma T}{\sqrt{T \gamma (1-\gamma)}} \tag{3}$$

> The process for detecting the soft watermark is identical to that for the hard
> watermark. We assume the null hypothesis (1) and compute a z-statistic using
> Equation (2). We reject the null hypothesis and detect the watermark if z is greater
> than a threshold. For arbitrary γ we have
>
>     z = (|s|_G − γT) / √(T γ(1 − γ)).                                        (3)

The null hypothesis stated in the paper:

> H0: The text sequence is generated with no knowledge of the red list rule.      (1)

> Suppose we choose to reject the null hypothesis if z > 4. In this case, the
> probability of a false positive is 3 × 10^−5, which is the one-sided p-value
> corresponding to z > 4.

Detection requires no access to the model API or weights — only the keyed hash
function and the candidate text. This is the headline practical property: a third party
(e.g., a social media platform) can run the detector independently.

## §7 — Robustness against token-substitution attacks {#sec-robustness}

> Three types of attacks are possible. Text insertion attacks add additional tokens
> after generation that may be in the red list and may alter the red list computation of
> downstream tokens. Text deletion removes tokens from the generated text, potentially
> removing tokens in the green list and modifying downstream red lists. Text
> substitution swaps one token with another, potentially introducing one red list token,
> and possibly causing downstream red listing.

On the T5 span-substitution attack (replacing up to fraction ε of tokens using
T5-Large as a paraphrase model):

> While this attack is effective at increasing the number of red list tokens in the text,
> as shown in Figure 6, we only measure a decrease in watermark strength of 0.01 AUC
> when ε = 0.1. While the watermark removal is more successful at a larger budget of
> 0.3, the average PPL of attacked sequences increases by 3× in addition to requiring
> more model calls.

> The initial, unattacked watermark is a γ = 0.5, δ = 2.0 soft watermark generated
> using multinomial sampling.

From Table 2 (empirical error rates at z = 4 threshold, T = 200 tokens):

> All soft watermarks at δ = 2.0 incur at most 1.6% (8/500) type-II error at z = 4.
> No type-II errors occurred for the hardest watermarks with δ = 10.0 and γ = 0.25.

> Note that, especially on longer text fragments such as essays, a few sentences that
> are partially or not at all paraphrased can be sufficient to trigger watermark
> detection at a statistically significant threshold.

[ANALOGY] The robustness argument is essentially an information-theoretic one: an
attacker substituting fraction ε of tokens can inject at most ~2εT red-list violations
(one per flipped token, one for its downstream neighbor), but the watermark signal
from the remaining (1−2ε)T tokens still dominates at realistic ε values. The formal
analysis is in §2.

[CONTRADICTION] Whether watermarking is practically deployable for provenance at
internet scale is unsettled. Critics (Sadasivan et al. 2023, "Can AI-Generated Text be
Reliably Detected?") argue that any watermark detectable from short spans must be
removable by a paraphrase attacker with a comparable-quality model. The paper's
response: "If the attacker had an equally strong language model at hand, there would
be no need to use the watermarked API, the attacker could generate their own text."

## Lineage {#sec-lineage}

- **Kirchenbauer 2023** (this paper) — green/red list with logit bias, z-score detector.
- **Kirchenbauer 2024 reliability** (arxiv 2306.04634, ICLR 2024) — paraphrase /
  human-rewrite robustness study.
- **SynthID-Text** (Dathathri et al. 2024, Nature) — tournament sampling; deployed
  at production scale in Gemini App. Replaces logit-bias with a tournament over
  candidate tokens biased toward a hash-derived watermark function.
- **Topic-conditioned watermarks** (arxiv 2404.02138) — partition by topic to
  improve robustness to topic-preserving paraphrase.

[Verified from PDF on 2026-05-12] Added §sec-algorithm (Algorithm 1 verbatim,
Algorithm 2 steps verbatim), §sec-zscore (Eq. 2 and Eq. 3 verbatim, null hypothesis
H0 verbatim), §sec-robustness (T5 attack results verbatim, Table 2 headline numbers).
Abstract verified verbatim. Corrected prior error: γ=0.5 (not 0.25) is the
primary experimental default.
