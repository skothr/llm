---
topic: interpretability/sparse-autoencoders
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - rajamanoharan2024-jumprelu
  - gao2024-topk-saes
  - gemma-scope-2024
  - leask2025-sae-not-canonical
  - dunefsky2024-transcoders
secondary_sources:
  - templeton2024-scaling-monosemanticity  # Tier A but HTML-only at transformer-circuits.pub
  - bricken2023-monosemanticity              # Tier A but HTML-only
related_topics:
  - interpretability/mechanistic-interpretability
  - interpretability/circuit-tracing
  - interpretability/probing
  - architecture/ffn
---

# Sparse autoencoders (SAEs)

A **sparse autoencoder** (SAE) is an unsupervised, post-hoc method for
decomposing a language model's hidden activations into a (much wider)
dictionary of learned directions, only a few of which fire on any given
token. The hope — *not* yet uncontested as of 2025 — is that those
directions correspond to interpretable, monosemantic *features* of the
input. SAEs are the dominant 2024–2025 technique for converting
polysemantic neuron-basis activations into a basis a human can name.

## 1. Formal definition

Given a language-model activation $\mathbf{x} \in \mathbb{R}^n$ (typically
the residual stream at some layer, or an MLP/attention output), an SAE
is a pair of an encoder and a decoder
`[rajamanoharan2024-jumprelu §2 Eq.1-2; kb/excerpts/rajamanoharan2024-jumprelu#sec-2]`:

$$\mathbf{f}(\mathbf{x}) := \sigma\!\left(\mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}\right), \qquad \hat{\mathbf{x}}(\mathbf{f}) := \mathbf{W}_{\text{dec}} \mathbf{f} + \mathbf{b}_{\text{dec}} \tag{1}$$

with $\mathbf{W}_{\text{enc}} \in \mathbb{R}^{M \times n}$,
$\mathbf{W}_{\text{dec}} \in \mathbb{R}^{n \times M}$, $M \gg n$ (typically
$M = 8n$ to $1000n$), $\mathbf{f}(\mathbf{x}) \in \mathbb{R}^M_{\ge 0}$
the sparse **feature activations**, $\hat{\mathbf{x}}$ the reconstruction.
The columns of $\mathbf{W}_{\text{dec}}$ — written $\mathbf{d}_i$ — are
the **decoder feature directions** (or *latents*) into which the SAE
decomposes $\mathbf{x}$.

The training objective always combines a reconstruction term with some
sparsity-promoting term:

$$\mathcal{L}(\mathbf{x}) = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}(\mathbf{f}(\mathbf{x}))\|_2^2}_{\mathcal{L}_{\text{reconstruct}}} + \mathcal{L}_{\text{sparsity}} \tag{2}$$

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | LM activation being decomposed (a single position in the residual stream / MLP / attention output) |
| $n$ | dimension of $\mathbf{x}$, equal to $d_{\text{model}}$ (residual stream) or the MLP/attention dimension |
| $M$ | dictionary size (number of SAE latents); $M / n$ is the **expansion factor** |
| $\sigma$ | encoder activation: ReLU, TopK, or JumpReLU (architecture defines the SAE family) |
| $L_0$ | the **sparsity** of $\mathbf{f}(\mathbf{x})$ — number of non-zero entries (the headline metric) |
| $\mathbf{d}_i$ | $i$-th decoder column; the direction in $\mathbb{R}^n$ that latent $i$ writes to |

## 2. Mechanism — the four production families

The arithmetic in Eq. (1) is fixed; what varies across modern SAE
families is **(a)** the activation function $\sigma$, and **(b)** the
sparsity term in Eq. (2). These two choices co-determine the SAE's
position on the reconstruction-fidelity vs. sparsity Pareto frontier.

### 2.1 ReLU + L1 (Bricken 2023, Templeton 2024)

The original recipe of "Towards Monosemanticity" (Bricken et al. 2023)
and "Scaling Monosemanticity" (Templeton et al. 2024). Encoder is
$\sigma = \text{ReLU}$; sparsity is enforced via an $\ell_1$ penalty on
$\mathbf{f}$:

$$\mathcal{L}_{\text{sparsity}}^{\text{L1}} = \lambda \|\mathbf{f}(\mathbf{x})\|_1$$

`[gao2024-topk-saes §2.2 Eq.1; kb/excerpts/gao2024-topk-saes#sec-2-2-baseline]`.

Two known issues `[gao2024-topk-saes §2.3; kb/excerpts/gao2024-topk-saes#sec-2-3]`:

- **Shrinkage.** L1 biases all positive feature activations toward zero,
  so reconstructed activations systematically underestimate true feature
  magnitudes. Documented and mitigated by Wright & Sharkey 2024.
- **Reparameterization-non-invariance.** Scaling encoder rows up and
  decoder columns down preserves $\mathbf{W}_{\text{dec}} \mathbf{f}$
  but changes $\|\mathbf{f}\|_1$, so the L1 penalty is degenerate without
  a unit-norm constraint on $\mathbf{d}_i$ `[rajamanoharan2024-jumprelu §3]`.

### 2.2 TopK (OpenAI / Gao 2024)

Replace ReLU with the $K$-sparse activation: keep the top $K$ pre-
activations, zero the rest:

$$\mathbf{f}(\mathbf{x}) = \text{TopK}(\mathbf{W}_{\text{enc}}(\mathbf{x} - \mathbf{b}_{\text{pre}})) \tag{3}$$

`[gao2024-topk-saes §2.3 Eq.2; kb/excerpts/gao2024-topk-saes#sec-2-3]`.
With sparsity exactly equal to $K$ at every token, the L1 penalty is
unnecessary and $\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$.

Benefits: directly controllable $L_0$, no shrinkage, simpler hyperparameter
sweeps. Cost: a partial-sort operation per forward pass (slower than
elementwise), and the **dead-latent problem** — at large $M$, up to 90%
of latents stop activating without mitigation
`[gao2024-topk-saes §2.4; kb/excerpts/gao2024-topk-saes#sec-2-4]`. Two
fixes used by OpenAI:
1. Initialize encoder rows as the transpose of decoder columns.
2. Auxiliary "AuxK" loss that uses the top-$k_\text{aux}$ *dead* latents
   to model the residual reconstruction error.

With these mitigations a 16M-latent SAE on GPT-4 keeps only 7% dead.

### 2.3 JumpReLU + L0-via-STE (DeepMind / Rajamanoharan 2024)

Replace ReLU with a learned-threshold step:

$$\text{JumpReLU}_\theta(z) = z\, H(z - \theta) \tag{4}$$

where $H$ is the Heaviside step function and $\theta \in \mathbb{R}^M_+$
is a **per-feature learned threshold**
`[rajamanoharan2024-jumprelu §3 Eq.4, Eq.8; kb/excerpts/rajamanoharan2024-jumprelu#sec-2-activations]`.

Sparsity is then trained directly via an $\ell_0$ penalty:

$$\mathcal{L}_{\text{sparsity}}^{\text{L0}} = \lambda \|\mathbf{f}(\mathbf{x})\|_0 \tag{5}$$

The discontinuity in the JumpReLU and the non-differentiability of $\ell_0$
are handled with **straight-through estimators** (STEs): backward pass
uses a smooth surrogate (kernel-density-estimated derivative of the
expected loss) for both $\theta$ and the L0 term.

Empirical headline: at fixed $L_0$, JumpReLU SAEs match or slightly
beat TopK on reconstruction MSE on Gemma 2 9B residual stream / MLP /
attention activations, while training as fast as a vanilla ReLU SAE
(elementwise activation, no partial sort)
`[rajamanoharan2024-jumprelu §1 Pareto; kb/excerpts/rajamanoharan2024-jumprelu#sec-1-pareto]`.

### 2.4 Gemma Scope (DeepMind 2024) — the open release

**Gemma Scope** trains JumpReLU SAEs on every layer and sublayer (residual
stream, MLP output, attention output) of Gemma 2 2B, 9B, and select
layers of 27B; releases >2,000 SAEs covering >30M total learned features,
trained on 4–16B tokens each
`[gemma-scope-2024 §1; kb/excerpts/gemma-scope-2024-jumprelu-saes#sec-1]`.
Compute used: ~20% of GPT-3 pretraining budget. Practical impact: until
2024, SAE work was largely Anthropic-internal; Gemma Scope made open
SAE-based interpretability tractable for academic and independent
researchers.

### 2.5 Comparison table

| Variant | Encoder activation | Sparsity term | Pareto position (Gemma 2 9B) | Dead-latent risk | Compute |
|---|---|---|---|---|---|
| **ReLU + L1** (Bricken 2023) | ReLU | $\lambda \|\mathbf{f}\|_1$ + $\|\mathbf{d}_i\|_2 = 1$ | weakest | low | cheapest |
| **Gated** (Rajamanoharan 2024) | gated ReLU (two encoders) | $\lambda \|\cdot\|_1$ on gating path | mid | low | 2× ReLU |
| **TopK** (Gao 2024) | $\text{TopK}(\cdot)$ | none (exact $L_0 = K$) | strong | **high** without AuxK | partial sort |
| **JumpReLU** (Rajamanoharan 2024) | learned-threshold step | $\lambda \|\mathbf{f}\|_0$ via STE | strongest | low | as cheap as ReLU |
| **BatchTopK** (Bussmann 2024) | TopK across the batch | none | claimed > TopK / JumpReLU | similar to TopK | partial sort + batch op |

`[gemma-scope-2024 §2.1; kb/excerpts/gemma-scope-2024-jumprelu-saes]`,
`[gao2024-topk-saes §2.3]`,
`[rajamanoharan2024-jumprelu §1, §2.2]`.

## 3. Transcoders — the close cousin

A **transcoder** is structurally an SAE but trained on a different
objective: it learns to reproduce the *output* of an MLP sublayer from
its *input*, with a sparse hidden layer in between
`[dunefsky2024-transcoders §3.1 Eq.3-5; kb/excerpts/dunefsky2024-transcoders#sec-3-1]`:

$$\mathbf{z}_{\text{TC}}(\mathbf{x}) = \text{ReLU}(\mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}), \quad \text{TC}(\mathbf{x}) = \mathbf{W}_{\text{dec}} \mathbf{z}_{\text{TC}}(\mathbf{x}) + \mathbf{b}_{\text{dec}}$$

$$\mathcal{L}_{\text{TC}}(\mathbf{x}) = \|\text{MLP}(\mathbf{x}) - \text{TC}(\mathbf{x})\|_2^2 + \lambda_1 \|\mathbf{z}_{\text{TC}}(\mathbf{x})\|_1$$

Difference from an SAE on the same layer
`[dunefsky2024-transcoders §1 Fig.1; kb/excerpts/dunefsky2024-transcoders#sec-1-comparison]`:

- An **SAE on the MLP output** gives a sparse decomposition of the
  *activation* $\text{MLP}(\mathbf{x})$ — input-dependent.
- A **transcoder** gives a sparse approximation of the *function*
  $\text{MLP}(\cdot)$ — input-invariant.

This input-invariance is the substrate for **weight-based circuit
analysis through MLP sublayers**: the connection from a pre-MLP feature
$f_i$ to a post-MLP feature $f_j$ factorizes neatly into an
input-dependent term (the activation level) and an input-invariant term
(the bilinear product of encoder/decoder vectors)
`[dunefsky2024-transcoders §1 input-invariance; kb/excerpts/dunefsky2024-transcoders#sec-1-input-invariance]`.

Transcoders are the architectural primitive used by Anthropic's 2025
**circuit-tracing** framework for Claude 3.5 Haiku — see
`kb/notes/interpretability/circuit-tracing.md`.

## 4. Variants and lineage (chronological)

- **2013** — Makhzani & Frey: $k$-sparse autoencoders. The mathematical
  ancestor of TopK SAEs `[gao2024-topk-saes §2.3]`.
- **2023** — Cunningham et al., Bricken et al. (Anthropic, "Towards
  Monosemanticity"): ReLU + L1 SAE on a 1-layer transformer; recovers
  ~thousands of monosemantic features. Establishes the program.
- **2024-04** — Rajamanoharan et al. (DeepMind): Gated SAEs.
- **2024-05** — Templeton et al. (Anthropic, "Scaling Monosemanticity"):
  ReLU + L1 SAE scales to Claude 3 Sonnet (production model); discovers
  abstract / safety-relevant features (e.g., a "code-bug" feature, a
  "deception" feature, a "Golden Gate Bridge" feature usable for
  steering).
- **2024-06** — Gao et al. (OpenAI): TopK SAEs, scaling laws $L(C)$ and
  $L(N, K)$, 16M-latent SAE on GPT-4.
- **2024-07** — Rajamanoharan et al. (DeepMind): JumpReLU SAEs.
- **2024-08** — Lieberum et al. (DeepMind): Gemma Scope — JumpReLU SAEs
  released openly across Gemma 2 2B/9B/27B
  `[gemma-scope-2024 §1, §2.2; kb/excerpts/gemma-scope-2024-jumprelu-saes#sec-2]`.
- **2024-11** — Dunefsky et al. (NeurIPS): transcoders for input-invariant
  circuit analysis.
- **2024-12** — Bussmann: BatchTopK SAEs (variant of TopK that picks the
  top-K across the batch instead of per-token).
- **2025-02** — Leask et al. (ICLR 2025): "Sparse Autoencoders Do Not
  Find Canonical Units of Analysis" — meta-SAEs and SAE stitching show
  larger SAEs find features absent from smaller ones (incompleteness)
  and larger latents decompose into combinations of smaller latents
  (non-atomicity). [CONTRADICTION] with the Bricken/Templeton "true
  features" framing
  `[leask2025-sae-not-canonical §1 contributions; kb/excerpts/leask2025-sae-not-canonical#sec-1-techniques]`.
- **2025-03** — Lindsey et al. (Anthropic): circuit tracing replaces
  SAEs+MLPs with **cross-layer transcoders**, enabling end-to-end
  attribution graphs on Claude 3.5 Haiku.

## 5. Intuitions and analogies

[ANALOGY] An SAE is a **dictionary** that the model never had: each
column $\mathbf{d}_i$ of $\mathbf{W}_{\text{dec}}$ is a "word" in a
language the user gets to define after the fact, and the encoder
decides which words are relevant for a given activation. The analogy
returns to canonical form in §1: $\hat{\mathbf{x}} = \sum_i f_i
\mathbf{d}_i$, which is exactly a sparse linear combination of dictionary
atoms. The dictionary picture is helpful but is *not* a separate claim
about what the model "really" computes — see §6.

[INTUITION] Why $M \gg n$? The **superposition hypothesis** (Elhage et
al., "Toy Models of Superposition", 2022): a network with $n$ neurons
can represent more than $n$ approximately-orthogonal features as long as
they fire sparsely. SAEs lift this superposition by inflating the
basis: with $M = 30 n$ they can dedicate one direction per feature
without forcing orthogonality. The empirical scaling laws
$L(C, N)$ in `[gao2024-topk-saes §3]` are consistent with this:
larger SAEs continue to find new features rather than saturating
`[gao2024-topk-saes §1 Fig.1; kb/excerpts/gao2024-topk-saes#sec-1-headline]`.

[INTUITION] $L_0$ vs. reconstruction is the **only** Pareto axis you
care about during SAE training. Every architectural innovation (Gated,
TopK, JumpReLU, BatchTopK) is an attempt to push this curve down-and-left.
At fixed compute, the family that gets you lower MSE at a target $L_0$
wins. JumpReLU and BatchTopK are the current frontier as of 2025.

[ANALOGY] A **transcoder** is to an MLP what a **sparse Taylor expansion**
is to a smooth function: the original is a dense, hard-to-read black
box; the transcoder is a sparse linear combination of basis functions
that approximates it everywhere, giving you something you can actually
factor and reason about. The analogy returns to canonical form: the
transcoder loss in §3 is exactly the function-approximation error
plus a sparsity penalty.

## 6. Frontier and open questions (as of 2026-05)

- **Are SAE features canonical?** [CONTRADICTION] Leask et al. 2025
  argue not: meta-SAEs decompose large-SAE latents (e.g., an "Einstein"
  latent) into combinations of more-elementary latents ("scientist",
  "Germany", "famous person")
  `[leask2025-sae-not-canonical §1 Einstein; kb/excerpts/leask2025-sae-not-canonical#sec-1-einstein]`,
  and SAE stitching reveals novel latents present in larger SAEs that
  smaller SAEs cannot recover. Bricken/Templeton's "true features"
  framing assumes a canonical decomposition exists. As of 2026, the
  SAE community has largely moved to a pragmatic position (use SAEs as
  a useful, basis-dependent tool) but the theoretical question is open.
- **High-frequency features are less interpretable.** JumpReLU and
  TopK SAEs both produce a small fraction (<0.06% of latents) that
  fire on >10% of tokens. Manual interpretability ratings on these are
  systematically lower
  `[rajamanoharan2024-jumprelu §1 caveat; kb/excerpts/rajamanoharan2024-jumprelu#sec-1-caveat]`.
  Whether these are "garbage", "shared bias features", or "a different
  kind of true feature" is unresolved.
- **SAE evaluation is fractured.** $L_0$ + reconstruction MSE is the
  training metric, but downstream metrics (loss recovered when
  re-inserting reconstructions, Neuronpedia interpretability scores,
  feature recovery on toy models) often disagree. Gao 2024 introduces
  several evaluation metrics
  `[gao2024-topk-saes §1 contributions; kb/excerpts/gao2024-topk-saes#sec-1-headline]`
  but no unified benchmark is community-standard.
- **SAEs vs. probes vs. activation patching.** SAE features are
  *correlative* (they fire when a concept is present), not *causal*
  (they cause the model to use the concept). Probing
  (`kb/notes/interpretability/probing.md`) shares this property.
  Activation patching (`kb/notes/interpretability/activation-patching.md`)
  is causal but doesn't yield a feature dictionary. Combining the two
  — using SAE features as patching targets — is an active
  methodology; transcoders + circuit tracing
  (`kb/notes/interpretability/circuit-tracing.md`) is one consolidated
  answer.
- **SAEs at the production frontier.** Templeton 2024 (Claude 3
  Sonnet) and Anthropic's 2025 circuit-tracing on Claude 3.5 Haiku
  demonstrate that SAEs/transcoders work at production scale, but
  the compute cost (~20% of pretraining for Gemma Scope; comparable
  for Claude) is non-trivial. Whether SAE coverage scales with
  pretraining compute or saturates is unknown.

## 7. See also

- `kb/notes/interpretability/mechanistic-interpretability.md` — the
  broader research program; SAEs are the dominant feature-extraction
  tool but only one of several.
- `kb/notes/interpretability/circuit-tracing.md` — Anthropic's 2025
  framework that uses cross-layer transcoders + attribution graphs
  to scale circuit analysis to production models.
- `kb/notes/interpretability/activation-patching.md` — the causal
  intervention technique; SAE features can be the *targets* of
  patching, but the two methodologies are independent.
- `kb/notes/interpretability/probing.md` — supervised linear-probe
  readout of activations; conceptually parallel to SAE feature
  activations but trained against an external label.
- `kb/notes/architecture/ffn.md` — the MLP sublayer, which SAEs
  decompose at input/output and which transcoders replace as a
  sparse approximation.
