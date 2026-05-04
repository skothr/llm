---
topic: scaling/mu-transfer
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - yang2022-mup
secondary_sources:
  - u-mup2024     # u-μP: μP + unit-scaling for FP8
  - meta-llama4   # uses "MetaP" cross-scale HP transfer (Meta's variant)
related_topics:
  - training/optimization
  - training/mixed-precision-and-stability
  - architecture/transformer-overview
  - scaling/chinchilla
---

# μ-Transfer (Maximal Update Parametrization)

The Maximal Update Parametrization (μP) is the unique-up-to-redundancy
weight + LR + multiplier parametrization in which **optimal
hyperparameters are invariant under width scaling**. Its operational
payoff, μTransfer, lets you tune learning rate, init scales, and
per-layer multipliers on a small *proxy* model and copy them
zero-shot to the full-scale *target* — for transformer LMs this saves
on the order of 90%+ of HP-tuning compute at frontier scale
`[yang2022-mup §1; kb/excerpts/yang2022-mup#sec-1-contributions]`.

This note covers the μP rules (§1), the mechanism of why standard
parametrization fails to transfer (§2), the practical recipe (§3),
variants/lineage (§4), and current open questions (§5–6).

## 1. Formal definition

### 1.1 The μP scaling rules — Table 3

For a multi-layer neural network with hidden width $n$, every linear
layer falls into one of three categories based on whether its
input/output dimensions are infinite-with-width (h) or finite
(input/output projections):

| | Input weights & all biases | Output weights | Hidden weights |
|---|---|---|---|
| **Init. variance** | $1/\text{fan\_in}$ | $1/\text{fan\_in}^2$ | $1/\text{fan\_in}$ |
| **SGD LR** | $\text{fan\_out}$ | $1/\text{fan\_in}$ | $1$ |
| **Adam LR** | $1$ | $1/\text{fan\_in}$ | $1/\text{fan\_in}$ |

`[yang2022-mup §4 Table 3; kb/excerpts/yang2022-mup#sec-4-table-3]`.
Standard parametrization (SP) uses init variance $1/\text{fan\_in}$
everywhere and uniform LR — boldface deviations from SP are the
*only* changes μP requires.

For Transformers, μP has one additional structural change
`[yang2022-mup §4 Definition 4.1; kb/excerpts/yang2022-mup#sec-4-definition-4-1]`:

> **Definition 4.1.** The Maximal Update Parametrization (μP) for a
> Transformer is given by Table 3 and **$1/d$ attention** instead of
> $1/\sqrt{d}$, i.e. the attention logit is calculated as $q^\top k / d$
> instead of $q^\top k / \sqrt{d}$ where query $q$ and key $k$ have
> dimension $d$.

This is a structural deviation from Vaswani's scaled dot-product
attention. The Vaswani $1/\sqrt{d_k}$ scaling is correct *at
initialization* (when $q, k$ are uncorrelated, so $\mathrm{Var}(q^\top
k) = d_k$ and dividing by $\sqrt{d_k}$ keeps unit variance). After even
one SGD step, $q$ and $k$ become correlated and
$\mathrm{Var}(q^\top k) = \Theta(d)$ — so $1/d$ is the correct
training-time rescaling. See §2.3.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $n$ | Hidden width (the dimension being scaled, e.g., $d_{\text{model}}$) |
| $\text{fan\_in}$ | Input dimension of a layer |
| $\text{fan\_out}$ | Output dimension of a layer |
| "input weight" | Layer with finite (constant) input dim, infinite output dim — e.g., the embedding |
| "output weight" | Layer with infinite input dim, finite output dim — e.g., the language-model head, the unembed |
| "hidden weight" | Layer with infinite input AND output dim — e.g., FFN matrices, attention QKV/O |
| $\eta$ | "master" learning rate (the dimensionless base) |
| $\eta_W$ | Effective LR for weight matrix $W$ — equals $\eta$ times Table 3's column for $W$ |

### 1.2 The abc-parametrization framework

Yang's framework (developed in Tensor Programs IV–V) characterizes a
parametrization by three functions of width $n$ for each parameter
$W$: an **a**bc-parametrization $(a_W, b_W, c_W)$ where:

- $a_W$ controls *parameter multiplier*: weight is used as $n^{-a_W} W$
- $b_W$ controls *initialization*: $W$ entries drawn $\sim n^{-b_W}
  \cdot \mathcal{N}(0, 1)$
- $c_W$ controls *learning rate*: $\eta_W = n^{-c_W} \eta$

μP corresponds to a specific choice of $(a_W, b_W, c_W)$ for each
weight; SP is a different choice. The space of "stable" choices
(parametrizations where infinite-width limit exists and is non-trivial)
is characterized in Yang & Hu 2021 (Tensor Programs IV), and **μP is
the unique stable choice that has feature learning** (i.e., the
infinite-width limit has weight updates of order $\Theta(1)$ in the
right metric, not order $0$ or $\infty$)
`[yang2022-mup §2; kb/excerpts/yang2022-mup#sec-2-clt]`.

## 2. Mechanism — why SP fails to transfer

### 2.1 The Central-Limit-Theorem analogy

`[yang2022-mup §2; kb/excerpts/yang2022-mup#sec-2-clt]`:

For sums $x_1 + \cdots + x_n$ of $n$ iid mean-0, variance-1 inputs,
the right scaling factor $c_n$ for nontrivial limit is
$c_n = 1/\sqrt{n}$ (CLT). If $c_n = 1$, the variance blows up; if
$c_n = 1/n$, the sum vanishes.

The **width** of a neural network plays the role of $n$ in the CLT.
Each hidden activation is a sum over $\Theta(n)$ contributions; the
right rescaling is $1/\sqrt{n}$ at init. SP gets *initialization*
right (both SP and μP use $1/\sqrt{n}$ for hidden init variance) but
gets the *learning-rate scaling and output-weight init* wrong.

### 2.2 Empirical demonstration — SP doesn't transfer, μP does

Yang et al. show CIFAR-10 MLPs with width $n \in \{256, 512, ..., 8192\}$:

> The optimal learning rate shifts by roughly an order of magnitude as
> the width increases from 256 to 8192; using the optimal learning of
> the smallest model on the largest model gives very bad performance,
> if not divergence.
> `[yang2022-mup §3; kb/excerpts/yang2022-mup#sec-3-mlp-sp]`

Under μP with the same widths, the optimal LR is **stable across
widths** — the LR-vs-loss curves overlap. This is the headline
empirical claim.

### 2.3 What blows up in SP — logits, attention logits, not embeddings

`[yang2022-mup §5; kb/excerpts/yang2022-mup#sec-5-blowup]`:

> Logits and attention logits, but not word embeddings, of a Transformer
> blow up with width in SP after 1 step of training. In contrast, all
> three are well-behaved with width in μP.

The asymmetry is operationally important: in SP, the only way to
prevent logit blow-up at large width is to *scale the LR down*, but
that simultaneously makes the input embeddings *not learn at all*.
SP's defect is not just "wrong global LR" — it's that **no single
global LR** works for both input layers and output layers at large
width. μP fixes this by giving the layer types *different* LR
scalings: $\eta_{\text{output-weight}} \propto 1/n$ in Adam,
$\eta_{\text{input-weight}} = \eta$ (constant) in Adam, etc.

### 2.4 The $1/d$ vs $1/\sqrt{d}$ attention scaling

In SP-Transformer, after 1 step of SGD, queries and keys at the same
position become correlated. The pre-softmax score
$q^\top k = \sum_{i=1}^d q_i k_i$ then has variance $\Theta(d)$ by
Law-of-Large-Numbers (not CLT), so dividing by $\sqrt{d}$ gives
variance $\Theta(\sqrt{d})$ — pre-softmax logits *blow up* with width.

Vaswani's original $1/\sqrt{d_k}$ argument in Attention Is All You Need
`[vaswani2017 §3.2.1 fn.4; kb/excerpts/vaswani2017#sec-3-2-1]` is correct
*at initialization* (uncorrelated $q, k$), and remains the canonical
choice in standard-parametrization Transformers (which constitute the
overwhelming majority of deployed models). μP's $1/d$ scaling is the
*correct training-time rescaling under the μP family* — when you
already commit to μP for the rest of the network, attention also needs
the matching rescaling to keep optimal HPs width-invariant.

[CONTRADICTION] In practice, many production μP-trained models keep
the $1/\sqrt{d_k}$ scaling and merely insert a tunable constant —
they are using a "μP-inspired" variant rather than strict Definition
4.1. The empirical cost of this looseness is small at production
widths but theoretically violates the unique-stable-feature-learning
characterization.

## 3. Mechanism — the μTransfer practical recipe

`[yang2022-mup §1, §7; kb/excerpts/yang2022-mup#sec-1-contributions]`:

1. **Parametrize target in μP.** Use the released PyTorch package or
   apply Table 3's rules manually. Decide which dimensions are
   "infinite" (typically $d_{\text{model}}$ and $d_{\text{ff}}$) and
   which are "finite" (the embedding's vocab dim, output head's
   vocab dim, and the head dim $d_k$ if $d_k$ is held fixed across
   sizes).
2. **Pick a small proxy.** Same depth, same vocab, same data, same
   tokenizer, but with $d_{\text{model}}^{\text{proxy}} \ll
   d_{\text{model}}^{\text{target}}$ — typical ratio 1/16 to 1/32.
3. **Sweep on proxy.** Tune master LR $\eta$, per-layer multipliers
   ($\alpha_{\text{output}}, \alpha_{\text{attn}}$, ...), Adam betas,
   init standard deviation. Standard small-model HP-search techniques
   apply.
4. **Copy α-values to target, run once.** Do not retune at scale. The
   physical LR for each layer is $\eta_W = \eta \cdot (\text{Table 3
   column for } W)$ at the target width.

The cost: tuning + 1 target run, vs. multiple target runs without
μTransfer. Yang et al. report μTransfer to GPT-3 6.7B at total tuning
cost = 7% of one pretraining run
`[yang2022-mup abstract; kb/excerpts/yang2022-mup#abstract]`.

### 3.1 What transfers — and what doesn't

`[yang2022-mup §6; kb/excerpts/yang2022-mup#sec-6-what-transfers]`,
`[yang2022-mup §6.1; kb/excerpts/yang2022-mup#sec-6-1-caveats]`:

**Transfers across width:**
- Master learning rate $\eta$
- Adam beta1, beta2
- LR schedule shape (cosine vs linear vs constant)
- Per-layer init multipliers
- Most parameter multipliers

**Transfers across batch size, sequence length, training steps**
(empirically, but not theoretically guaranteed):
- Mostly yes, within a "reasonable range" (batch ≥ 32, seqlen ≥ 128,
  steps ≥ 5000).

**Transfers across depth** (PARTIAL):
- The optimal init standard deviation does *not* transfer well across
  depth.
- For pre-LN transformers, the LR schedule shape transfers; for
  post-LN, depth transfer is more brittle.
- Practical recipe: fix init std at the depth you'll target, sweep
  other HPs.

**Does NOT transfer:**
- Regularization (weight decay, dropout) — these depend on data size
  and overfitting risk, not on the parametrization.
- Hyperparameters that interact with the regularization vs.
  optimization trade-off.

## 4. Variants and lineage

| Year | Method | Contribution | Status |
|---|---|---|---|
| 2021 | Tensor Programs IV (Yang & Hu) | Theoretical foundation: feature-learning width-infinity limits, μP for SGD | Foundational |
| 2022 | Tensor Programs V `[yang2022-mup]` | μP for Adam, μTransfer methodology, GPT-3 6.7B demo | Canonical |
| 2023 | Tensor Programs VI / Depth-μP | Extends to depth axis: principled depth-scaling HP transfer | ICLR 2024 |
| 2024 | u-μP `[u-mup2024]` | Combines μP with unit-scaling; native FP8 training without dynamic rescaling | ICLR 2025 Spotlight |
| 2024 | Cerebras Practitioner's Guide | Industry adoption documentation | Production-validated |
| 2026 | Llama 4 "MetaP" `[meta-llama4]` | Meta's variant for cross-scale HP transfer (details thin in tech report) | Production at scale |

[CONTRADICTION] Frontier labs publicly disclose μP usage unevenly.
Cerebras has explicit guidance documents. Microsoft (Phi-4) and
Anthropic do not disclose. Google's Gemma 3 tech report mentions HP
transfer but doesn't explicitly cite μP. The community working
assumption (as of early 2026) is that "μP-like" parametrization is
default at frontier; whether each lab uses strict Definition 4.1 vs.
loose μP-inspired variants is opaque.

### 4.1 u-μP — μP + unit scaling for FP8

The 2024 Graphcore paper integrates μP with unit-scaling: every layer
output is dynamically (or, for u-μP, statically and predictably)
rescaled to have unit variance. The combination enables **native FP8
training** because the predictable activation magnitudes don't require
runtime per-tensor rescaling
`[u-mup2024; kb/index/papers.json#u-mup2024]`. This is the practical
bridge between μP theory and FP8 production training (which DeepSeek-V3
and Llama 4 both use at scale).

## 5. Intuitions and analogies

[ANALOGY] **μP is the "right" parametrization in the same sense that
$1/\sqrt{n}$ is the right CLT scaling.** The CLT picks
$1/\sqrt{n}$ uniquely as the rescaling that gives the sum a non-
trivial limit (not zero, not infinity). The μP table picks the
parametrization uniquely such that *every* hidden activation, every
gradient component, and every weight update has $\Theta(1)$ magnitude
in the width-infinity limit. Both are statements about the unique
"correct" rescaling for asymptotic regularity. The canonical math is
Yang's abc-parametrization classification (§1.2).

[INTUITION] **Why "tune small, run big" works.** Under μP, the optimal
HPs are *width-invariant* in a precise mathematical sense (the
infinite-width limit has the same optimal HP as any finite width). So
the proxy at width $n_0$ and the target at width $n \gg n_0$ are not
"different problems requiring different HPs"; they're approximations
to the same limit, and the limit's optimal HP is the relevant
quantity.

[INTUITION] **The "MetaP" and "MetaP-like" variants you read about
in the tech reports.** Most production "μP" is loose — labs apply
some subset of Table 3's rules (the LR scalings) but skip the strict
$1/d$ attention or the $1/n^2$ output-weight init. This typically
still helps a lot in practice (HP transfer becomes much more stable)
but loses the unique-feature-learning guarantee. Treat published
"MetaP / μP" claims as evidence-of-usage, not as evidence-of-strict-
adherence-to-Definition-4.1.

## 6. Frontier and open questions (as of 2026-05)

- **Independent large-scale validation.** Negative results for μP at
  10B+ scale are scarce in the literature, but so are independent
  positive results outside the original Microsoft/Cerebras/Graphcore
  ecosystem. The community would benefit from open replications.
- **MoE and hybrid architectures.** μP was derived for dense
  transformers. Whether DeepSeekMoE's fine-grained experts + shared
  experts admit a clean μP analog is open. Mamba / SSM hybrid
  architectures have parametrization challenges too.
- **Precision interaction.** Scaling Laws for Precision (Kumar 2024)
  show training-data and inference-precision interact in non-trivial
  ways. u-μP partially addresses this for training, but the joint
  μP × precision × token count surface is undermapped.
- **Depth-μP at the deep extreme.** Tensor Programs VI extends to
  depth, but production models at $L = 80$+ (Gopher, Chinchilla,
  large dense base for MoE) sit beyond the regime explored. Whether
  Depth-μP is sufficient or if additional corrections are needed is
  open.
- **The $1/d$ attention question in practice.** The strict μP
  attention rescaling is rarely used in practice; whether the
  $1/\sqrt{d}$-variant μP loses meaningful HP-transfer accuracy at
  10B+ scale is empirically not nailed down.

## 7. See also

- `kb/notes/training/optimization.md` — base learning rate, AdamW,
  the optimization context μTransfer plugs into.
- `kb/notes/training/mixed-precision-and-stability.md` — u-μP and
  FP8 training; how μP enables low-precision regimes.
- `kb/notes/architecture/transformer-overview.md` — μP's $1/d$
  attention is a deviation from Vaswani's $1/\sqrt{d_k}$ scaling.
- `kb/notes/architecture/attention-mechanism.md` — Vaswani §3.2.1's
  variance argument vs. μP's Definition 4.1.
- `kb/notes/scaling/chinchilla.md` — the scaling-recipe parent topic.
  Chinchilla and μP are *complementary*: Chinchilla tells you how to
  spend FLOPs across $N$ and $D$; μP tells you what HPs to use at
  any $N$ given a small-scale sweep.
