---
topic: inference/speculative-decoding
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - leviathan2023      # original speculative sampling
  - cai2024-medusa     # multi-head, no draft model
  - li2024-eagle       # feature-level speculation
secondary_sources:
  - chen2023-speculative   # concurrent DeepMind derivation
  - eagle3-2025            # NeurIPS 2025; abandons feature prediction
related_topics:
  - architecture/multi-token-prediction
  - inference/kv-cache-management
  - inference/serving-systems
---

# Speculative decoding

Autoregressive decoding is bandwidth-bound: at every step the entire
KV cache and all model parameters are read from HBM to produce a
*single* token. Speculative decoding breaks the per-step quadratic
$N \times $ FLOP-bandwidth product by **proposing $\gamma$ candidate
tokens cheaply (draft) and verifying them in one parallel pass of the
target model**. The mathematical contribution that makes it useful —
not just an approximation — is a rejection-sampling scheme that
**preserves the target model's output distribution exactly**.

The literature splits along two cuts:
- *Draft mechanism*: smaller draft model (Leviathan 2023, EAGLE),
  multiple heads on the same model (Medusa), or no auxiliary model
  at all (Lookahead/Jacobi, n-gram).
- *Verification semantics*: distribution-preserving (rejection
  sampling) vs. relaxed (typical acceptance, deterministic argmax).

## 1. The exact-distribution rejection rule

Let $M_p$ be the **target model** (large, slow) with conditional
$p(x_t | x_{<t})$, and $M_q$ the **draft model** (small, fast) with
$q(x_t | x_{<t})$. The core algorithm `[leviathan2023 §2.1;
kb/excerpts/leviathan2023#sec-2-1]`:

1. Sample $\gamma$ tokens from $M_q$ autoregressively.
2. Run $M_p$ once in parallel on the $\gamma+1$ prefixes (target
   parameters and KV are read once).
3. Sequentially accept/reject each draft token using the rule below.
4. If a token is rejected, sample its replacement from a corrected
   distribution.

The **speculative-sampling rejection rule** is `[leviathan2023 §2.3;
kb/excerpts/leviathan2023#sec-2-3]`:

> To sample $x \sim p(x)$, we instead sample $x \sim q(x)$, keeping it
> if $q(x) \le p(x)$, and in case $q(x) > p(x)$ we reject the sample
> with probability $1 - \tfrac{p(x)}{q(x)}$ and sample $x$ again from
> an adjusted distribution $p'(x) = \operatorname{norm}(\max(0, p(x) -
> q(x)))$ instead.

This is exactly Metropolis-Hastings with proposal $q$ and target $p$,
specialized to discrete distributions. The acceptance probability per
token is $\min(1, p(x)/q(x))$. **The output is i.i.d. from $p$ exactly
— no quality degradation, regardless of how poor $q$ is.** This is the
load-bearing property; everything else is engineering on the
acceleration factor.

Concurrent work by Chen et al. (DeepMind, arXiv 2302.01318)
`[chen2023-speculative]` derives the same rule from a different
direction, calling it "speculative sampling." The two papers are
treated as joint priors in the field.

## 2. Speedup analysis

Define the **acceptance rate** $\alpha = E_{x \sim q}[\min(1, p(x)/q(x))]$.
By Corollary 3.6 `[leviathan2023 §3.2; kb/excerpts/leviathan2023#sec-3-2]`,

$$\alpha = E[\min(p, q)] = 1 - D_{LK}(p, q), \tag{1}$$

where $D_{LK}$ is the symmetric divergence
$D_{LK}(p, q) = \sum_x |p(x) - M(x)|$ with $M = (p+q)/2$. So $\alpha$
is essentially "1 minus how different $q$ is from $p$" by total
variation. Under the i.i.d. assumption, the expected number of tokens
generated per algorithm call is the capped geometric
`[leviathan2023 §3.1 Eq.1; kb/excerpts/leviathan2023#sec-3-1]`:

$$E(\#\text{generated tokens}) = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}. \tag{2}$$

With $c$ as the per-token cost ratio of $M_q$ to $M_p$ (typically
$c \ll 0.05$), the **expected walltime improvement factor** is
`[leviathan2023 §3.3 Theorem 3.8;
kb/excerpts/leviathan2023#sec-3-3]`

$$\text{walltime improvement} = \frac{1 - \alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)}. \tag{3}$$

Empirical $\alpha$ on T5-XXL-11B with a T5-Small-77M draft is around
$0.75$ on translation tasks; with a bigram model, $\alpha \approx 0.20$
`[leviathan2023 §3.6 Table 3; kb/excerpts/leviathan2023#sec-3-6]`.
The original paper reports 2–3× wallclock speedup on T5-XXL.

[INTUITION] The "free lunch" of speculative decoding is *not* free
arithmetic — it's free *bandwidth*. The target model still does $\gamma$
forward passes of work in expectation; what changes is that those
passes are done *in parallel on a single batch* rather than $\gamma$
serial steps. Because decoding is bandwidth-bound (loading KV cache
and weights once is the dominant cost), running them in parallel
amortizes that cost. The arithmetic-operations cost actually *increases*
slightly (Theorem 3.11) — but on real hardware it doesn't matter.

[INTUITION] The rejection rule's correctness depends only on
$x \sim q$, not on what $q$ is. So one can swap drafters per request,
per task, even per token without re-deriving anything. The choice of
$q$ is purely a speed/cost tradeoff — bad drafts just make $\alpha$ low.

## 3. The draft-model design space

### 3.1 Smaller-from-same-family drafts (Leviathan 2023, Chen 2023)

Pick a small model from the same family as the target (e.g. T5-Small
draft for T5-XXL target, LLaMA-7B draft for LLaMA-70B target). The
distribution $q$ is naturally close to $p$ if both are trained on
the same corpus with the same tokenizer.

Limitation `[li2024-eagle §1; kb/excerpts/li2024-eagle#abstract]`:

> Despite the 7B model's potential as a draft model, its high
> overhead diminishes acceleration gains. Training a new,
> appropriately sized draft model specifically for speculative
> sampling is not an ideal solution either due to the high cost.

So the family-matched drafter approach hits a practical wall: too-
small drafts have low $\alpha$, too-large drafts have high $c$, and
training a custom-size drafter is itself expensive.

### 3.2 Medusa — multiple heads, no draft model

**Medusa** `[cai2024-medusa]` removes the auxiliary draft model. It
adds $K$ extra LM-head-shaped MLPs (Medusa heads) on top of the
existing backbone's last hidden state $h_t$. The $k$-th head predicts
the token at offset $t+k+1$ `[cai2024-medusa §2.1.1;
kb/excerpts/cai2024-medusa#sec-2-1-1]`:

$$p_t^{(k)} = \operatorname{softmax}\!\left( W_2^{(k)} \cdot (\operatorname{SiLU}(W_1^{(k)} h_t) + h_t) \right), \tag{4}$$

with $W_1^{(k)}$ initialized to zero (so the head starts as a pass-
through producing the original LM head's prediction).

To verify a *tree* of candidate continuations rather than a single
chain, Medusa uses **tree attention** with an attention mask that
encodes the tree's parent-child structure
`[cai2024-medusa §2.1.2; kb/excerpts/cai2024-medusa#sec-2-1-2]`.
Candidates are formed by Cartesian product of top-$s_k$ predictions
from each head; with $s_1 = 4, s_2 = 3, s_3 = 2$ that's
$\sum_k \prod_i s_i$ tokens evaluated in one parallel pass.

Two training regimes:

- **Medusa-1**: heads are trained with the backbone *frozen*. Loss is
  a weighted sum of cross-entropies, $\lambda_k = 0.8^k$
  `[cai2024-medusa §2.2.1 Eq.1; kb/excerpts/cai2024-medusa#sec-2-2-1]`.
  Lossless — the original LM head is unchanged, so the model's normal
  distribution is preserved.
- **Medusa-2**: heads are trained jointly with backbone, with
  $\mathcal{L}_{\text{Medusa-2}} = \mathcal{L}_{\text{LM}} + \lambda_0
  \mathcal{L}_{\text{Medusa-1}}$ to prevent backbone drift
  `[cai2024-medusa §2.2.2 Eq.2;
  kb/excerpts/cai2024-medusa#sec-2-2-2]`. Higher speedup, but
  requires careful training.

**Typical acceptance** `[cai2024-medusa §2.3.1;
kb/excerpts/cai2024-medusa#sec-2-3-1]` is Medusa's optional tradeoff:
instead of full rejection sampling, accept any token whose target-
model probability exceeds a threshold of the form
$\min(\epsilon, \delta \exp(-H(p)))$ (entropy-adaptive). This breaks
strict distribution preservation under temperature ≠ 0, in exchange
for higher acceptance rates.

Headlines: Medusa-1 ~2.2×, Medusa-2 ~2.3–2.8× on Vicuna 7B/13B/33B
`[cai2024-medusa abstract; kb/excerpts/cai2024-medusa#abstract]`.

[CONTRADICTION] Medusa with rejection sampling *is* distribution-
preserving (it's just speculative sampling with the multi-head
proposal). Medusa with typical acceptance is *not*. Papers/blogs
sometimes conflate these. The distinction matters for any deployment
that cares about reproducible sampling.

### 3.3 EAGLE — feature-level speculation

**EAGLE** `[li2024-eagle]` reframes the drafter as predicting *features*
(the second-to-top-layer hidden state) rather than tokens. The
intuition: token sequences have high local entropy (many plausible
next-tokens), but feature sequences are more predictable.

The EAGLE drafter is a single transformer decoder layer (~0.24B for
LLaMA2-7B target, ~0.99B for LLaMA2-70B target) that takes
$(F_{1:i}, T_{2:i+1})$ and predicts $f_{i+1}$
`[li2024-eagle §3.1; kb/excerpts/li2024-eagle#sec-3-1]`. The shifted
token sequence $T_{2:i+1}$ resolves the inherent uncertainty in
feature autoregression — at the time of predicting $f_{i+1}$, the
sampled token $t_{i+1}$ that branched the actual generation is known
and conditioned on. Without this trick, feature-space autoregression is
fundamentally unable to commit to which branch was taken.

Training combines regression and classification losses
`[li2024-eagle §3.2; kb/excerpts/li2024-eagle#sec-3-2]`:

$$L = \text{SmoothL1}(f_{i+1}, \hat f_{i+1}) + 0.1 \cdot \text{CE}(p_{i+2}, \hat p_{i+2}). \tag{5}$$

The classification loss applies the *target* model's frozen LM head to
both true and predicted features and matches the resulting
distributions, ensuring the drafter optimizes the *useful* objective
(token agreement) and not just feature L2 distance.

EAGLE is **distribution-preserving** by inheritance — it uses Leviathan-
style rejection sampling at the verification step
`[li2024-eagle §3.3; kb/excerpts/li2024-eagle#sec-3-3]`. Speedups on
LLaMA2-Chat 70B at temperature=0: 3.01×; at temperature=1: 2.67×
`[li2024-eagle §1 Fig.1; kb/excerpts/li2024-eagle#sec-1-figure-1]`.

EAGLE-3 `[eagle3-2025]` (NeurIPS 2025) abandons feature prediction
entirely in favor of direct token prediction with multi-layer feature
fusion, claiming up to 6.5× and 1.4× over EAGLE-2. The "training-time
test" technique trains the drafter on its own outputs rather than
the target's features.

### 3.4 Model-free drafts — Lookahead, Jacobi, n-gram

When no drafter is available, you can still get speedup via:

- **Jacobi decoding**: parallel fixed-point iteration over multiple
  positions until they stop changing. Provably converges to greedy
  decoding output but with variable iteration count.
- **Lookahead decoding** (`[FORUM-SIGNAL: arXiv 2402.02057]`):
  combines Jacobi iteration with an n-gram pool harvested from
  past generations, used to guess multi-token continuations.
- **Prompt-lookup decoding**: for tasks with high prompt-output
  overlap (summarization, edit tasks), draft from the prompt itself.

These methods have $\alpha < 0.6$ typically (lower than draft-model
methods) but have $c = 0$ (no extra forward passes), which can win
when the drafter would be expensive
`[leviathan2023 §3.6; kb/excerpts/leviathan2023#sec-3-6]`.

## 4. Where does the KV cache go?

A subtlety: each draft step creates KV state in the target model's
cache, but only the *accepted* tokens' KV state should persist.
PagedAttention handles this naturally (`[kwon2023 §4;
kb/excerpts/kwon2023#sec-4-1]`) since rejected blocks are simply not
committed; the request continues from the accepted prefix. EAGLE's
tree-structured drafts add a wrinkle: the verifier needs tree-
attention masking that allows children to attend to their unique
ancestor chains — this is the same mechanism Medusa uses
`[cai2024-medusa §2.1.2;
kb/excerpts/cai2024-medusa#sec-2-1-2]`. Production stacks (vLLM,
SGLang) integrate speculative decoding on top of paged KV; an extra
"draft KV pool" is allocated per-request.

## 5. Comparison summary

| Method | Auxiliary model? | $\alpha$ (typical) | $c$ | Distribution-preserving? |
|---|---|---|---|---|
| Leviathan/Chen 2023 | small same-family drafter | 0.5–0.8 | 0.01–0.1 | yes |
| Medusa-1 (frozen) | none (heads) | 0.6–0.7 | ~0 | yes (with rejection) |
| Medusa-2 (joint) | none (heads, joint-trained) | 0.7–0.8 | ~0 | yes (with rejection) / no (with typical-acc) |
| Lookahead / Jacobi | none | 0.4–0.6 | 0 | yes |
| EAGLE | feature-drafter (~3% target params) | ~0.8 | small | yes |
| EAGLE-3 | token-drafter w/ feature fusion | higher | small | yes |

Reported headline speedups are not directly comparable — they depend on
hardware, workload (greedy vs. sampling), batch size, and which
"vanilla" baseline is taken. The headline numbers above pull from each
paper's most-favorable configuration.

## 6. Frontier and open questions (as of 2026-05)

- **Reasoning trace acceleration.** EAGLE-2 and EAGLE-3 numbers are
  on standard chat benchmarks (MT-bench, HumanEval). Performance on
  o1/R1-style 32k+ reasoning traces is not benchmarked publicly. The
  draft-acceptance rate likely changes for chain-of-thought patterns.
- **Speculative decoding + grammar constraints.** Combining XGrammar-
  style structured-output decoding with speculative drafts is a 2026
  research thread (`[FORUM-SIGNAL: arXiv 2603.03305 / 2601.04426]`).
  The interaction is non-trivial because rejected drafts may have
  branched into invalid grammar states.
- **Multi-token prediction vs. speculative decoding.** DeepSeek-V3
  `[deepseek-v3]` uses *multi-token prediction* (MTP) heads at training
  time, which both improve quality and double as a built-in speculative
  drafter. This is a structural alternative to bolt-on speculative
  decoding; treated in
  `kb/notes/architecture/multi-token-prediction.md`.
- **Adversarial prompts.** $\alpha$ depends on the prompt distribution.
  An adversarial prompt that maximizes the divergence between $p$
  and $q$ degrades acceleration to near 1×. The robustness of
  $\alpha$ across deployment workloads is not well-characterized.
- **Tree shape optimization.** EAGLE-2 introduces dynamic tree
  construction; the optimal tree shape is workload-dependent and
  active research
  `[FORUM-SIGNAL: arxiv 2406.16858]`.

## 7. See also

- `kb/notes/architecture/multi-token-prediction.md` — DeepSeek-V3's
  MTP heads as a structural alternative.
- `kb/notes/inference/kv-cache-management.md` — paged KV layout and
  copy-on-write make speculative drafts cheap to manage.
- `kb/notes/inference/serving-systems.md` — vLLM and SGLang's
  speculative-decoding integration; production tradeoffs.
