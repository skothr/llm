---
topic: architecture/ffn
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - vaswani2017
  - shazeer2020
  - touvron2023
secondary_sources:
  - radford2018
  - radford2019-gpt2
  - brown2020
  - meta-llama3
  - deepseek-v3
related_topics:
  - architecture/transformer-overview
  - architecture/moe
  - architecture/attention-mechanism
  - architecture/normalization
---

# Feed-forward network (FFN) sublayer

The FFN is one of two sublayers in every Transformer block (the other
is attention). It applies a **position-wise** non-linear transformation:
each token's $d_{\text{model}}$-dim residual-stream vector is processed
*independently of every other position*. The original 2017 form was a
2-matrix MLP with ReLU; the modern (2023+) form is **SwiGLU**, a
3-matrix gated structure that has become near-universal in production
LLMs.

## 1. Formal definition

### 1.1 Original Vaswani 2017 FFN

Per token (with biases):

$$\mathrm{FFN}(\mathbf{x}) = \max(0, \mathbf{x} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2 \tag{1}$$

`[vaswani2017 §3.3; kb/excerpts/vaswani2017#sec-3-3]` `[shazeer2020 §1 Eq.1; kb/excerpts/shazeer2020#sec-1]`

with $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$,
$W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, base
choice $d_{\text{ff}} = 4 d_{\text{model}}$ `[vaswani2017 §3.3]`.

### 1.2 Bias-free T5/decoder-only form

Modern LLMs drop biases:

$$\mathrm{FFN}_{\mathrm{ReLU}}(\mathbf{x}) = \max(\mathbf{x} W_1, 0) W_2 \tag{2}$$

`[shazeer2020 §1 Eq.2; kb/excerpts/shazeer2020#sec-1]`. GPT-2 introduced
GELU as the activation `[radford2019-gpt2 §2.3; kb/excerpts/radford2019-gpt2#sec-2-3]`:

$$\mathrm{FFN}_{\mathrm{GELU}}(\mathbf{x}) = \mathrm{GELU}(\mathbf{x} W_1) W_2$$

### 1.3 SwiGLU — the modern form (LLaMA, Mistral, Qwen, DeepSeek, …)

A **3-matrix gated structure** `[shazeer2020 §2 Eq.6; kb/excerpts/shazeer2020#sec-2-eq6]`:

$$\mathrm{FFN}_{\mathrm{SwiGLU}}(\mathbf{x}) = (\mathrm{Swish}_1(\mathbf{x} W) \otimes (\mathbf{x} V)) W_2 \tag{3}$$

with $W, V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$,
$W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, $\otimes$
elementwise product. $\mathrm{Swish}_1(z) = z \cdot \sigma(z)$ where
$\sigma$ is the sigmoid. Bias-free.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $d_{\text{model}}$ | residual-stream width |
| $d_{\text{ff}}$ | FFN hidden dimension |
| $W$ (or $W_1$) | up-projection matrix; gate projection in SwiGLU |
| $V$ | up-projection matrix; value projection in SwiGLU (3rd matrix) |
| $W_2$ | down-projection matrix |
| $\sigma$ | sigmoid function |
| $\mathrm{Swish}_\beta(z) = z \sigma(\beta z)$ | Swish activation; $\beta = 1$ in SwiGLU |
| $\mathrm{Swish}_1(z) = z \sigma(z)$ | the SiLU function |

## 2. Mechanism

### 2.1 Tensor flow with shapes

For input $X \in \mathbb{R}^{B \times N \times d_{\text{model}}}$:

**Vaswani 2-matrix:**
1. $H = \mathrm{ReLU}(X W_1)$ — shape $(B, N, d_{\text{ff}})$. Up-projection.
2. $Y = H W_2$ — shape $(B, N, d_{\text{model}})$. Down-projection.
   Parameters: $2 \cdot d_{\text{model}} d_{\text{ff}}$.

**SwiGLU 3-matrix:**
1. $G = X W$ — gate pre-activation, shape $(B, N, d_{\text{ff}})$.
2. $U = X V$ — value, shape $(B, N, d_{\text{ff}})$.
3. $H = \mathrm{Swish}_1(G) \otimes U$ — gated activation, shape
   $(B, N, d_{\text{ff}})$.
4. $Y = H W_2$ — shape $(B, N, d_{\text{model}})$.
   Parameters: $3 \cdot d_{\text{model}} d_{\text{ff}}$.

### 2.2 The 2/3 parameter-equalization rule

SwiGLU's third matrix means equal-parameter comparison requires
$d_{\text{ff}}^{\text{SwiGLU}} = \tfrac{2}{3} d_{\text{ff}}^{\text{ReLU}}$
`[shazeer2020 §2; kb/excerpts/shazeer2020#sec-2-eq6]`. So the modern
SwiGLU LLM uses

$$d_{\text{ff}} \approx \tfrac{2}{3} \cdot 4 d_{\text{model}} = \tfrac{8}{3} d_{\text{model}}$$

rounded for hardware alignment. LLaMA-1 7B: $d = 4096$, target $d_{\text{ff}}
= \tfrac{8}{3} \cdot 4096 \approx 10923$, **rounded to 11008** (multiple
of 256) `[touvron2023 §2; kb/excerpts/shazeer2020#sec-2-eq6]`. LLaMA-3
8B: same. LLaMA-3 70B: $d = 8192$, $d_{\text{ff}} = 28672$.

### 2.3 Compute and parameter cost per block

For a dense SwiGLU layer at $B = 1$, $N = 1$ (one decode step):

- **FLOPs**: $3 \cdot 2 \cdot d_{\text{model}} d_{\text{ff}} \approx 16 d_{\text{model}}^2$.
- **Parameters**: $3 \cdot d_{\text{model}} d_{\text{ff}} \approx 8 d_{\text{model}}^2$.

The FFN is by far the **parameter-heaviest** sublayer in a typical
decoder-only block. For LLaMA-3 8B with $d = 4096$, $d_{\text{ff}} =
14336$ (LLaMA-3 used a different $d_{\text{ff}}$ choice than 8/3; see
`[meta-llama3 §3.1]`), FFN params are 176M per block × 32 blocks ≈
5.6B — about 70% of total parameters.

This is precisely why **MoE replaces the FFN, not attention**: the FFN
is where you can scale parameter count without scaling per-token
compute. See `kb/notes/architecture/moe.md`.

## 3. Variants and lineage

### 3.1 Activation function progression

| Variant | $d_{\text{ff}}$ | Activation | Used by | Source |
|---|---|---|---|---|
| Vaswani 2017 | $4d$ | ReLU | original Transformer | `[vaswani2017 §3.3]` |
| BERT, T5, OPT | $4d$ | GELU | encoder-only & enc–dec era | `[devlin2019-bert §3.1]` |
| GPT-2/3 | $4d$ | GELU | decoder-only era | `[radford2019-gpt2 §2.3]` |
| LLaMA-1+ | $\tfrac{8}{3}d$ | SwiGLU | most modern decoder-only | `[touvron2023 §2]` |
| PaLM | $4d$ | SwiGLU (no $\tfrac{2}{3}$ factor) | Google's variant | `[chowdhery2022 §2]` |
| GLaM, T5.1.1 | $\tfrac{8}{3}d$ | GEGLU | Google models | `[shazeer2020 Tab.1; kb/excerpts/shazeer2020#sec-3-2]` |

### 3.2 Shazeer 2020 GLU-variant family

Shazeer 2020 systematically tested 7 variants on T5-base. Heldout-set
log-perplexity at 524K steps `[shazeer2020 §3.2 Tab.1; kb/excerpts/shazeer2020#sec-3-2]`:

| Variant | Activation | Perplexity |
|---|---|---|
| FFN_ReLU (baseline) | ReLU | 1.677 |
| FFN_GELU | GELU | 1.679 |
| FFN_Swish | Swish | 1.683 |
| FFN_GLU | sigmoid gate | 1.663 |
| FFN_Bilinear | linear gate | 1.648 |
| **FFN_GEGLU** | GELU gate | **1.633** |
| **FFN_SwiGLU** | Swish gate | **1.636** |
| FFN_ReGLU | ReLU gate | 1.645 |

GEGLU and SwiGLU are within 0.003 of each other; SwiGLU was adopted by
LLaMA, presumably for the smoother Swish surface. GEGLU appears in
Google models (GLaM, T5.1.1, mT5) `[shazeer2020 §3.2; kb/excerpts/shazeer2020#sec-3-2]`.

### 3.3 The "divine benevolence" status

Shazeer 2020's conclusion is famous for its honesty:

> "We offer no explanation as to why these architectures seem to work;
> we attribute their success, as all else, to divine benevolence."
> `[shazeer2020 §4; kb/excerpts/shazeer2020#sec-4]`

The empirical gain (~0.04 perplexity at T5-base) is small but
consistent and has held across scale. **No first-principles
derivation** of SwiGLU's advantage exists in this paper. Subsequent
theoretical work (gating as soft expert selection; bilinear interaction
as second-order feature mixing) is post-hoc.

### 3.4 GeGLU vs SwiGLU vs ReGLU at frontier scale

At frontier scale (8B+), no large-scale ablation has reproduced
Shazeer 2020's GEGLU > SwiGLU ranking; SwiGLU is the universal choice
in 2023–2026 publications. The decision is essentially **inertial**:
LLaMA-1 picked SwiGLU; everyone copied LLaMA-1.

[CONTRADICTION] Shazeer 2020 ranks GEGLU > SwiGLU on T5-base; LLaMA
chose SwiGLU and the field followed. No published frontier-scale
re-ablation. Whether GEGLU would still win at LLaMA-3 8B+ is unknown.

### 3.5 MoE expansion of the FFN

MoE replaces the single FFN with $E$ parallel SwiGLU FFNs ("experts")
plus a router $g(\mathbf{x}) \in \mathbb{R}^E$:

$$\mathrm{MoE}(\mathbf{x}) = \sum_{i \in \mathrm{TopK}(g(\mathbf{x}))} g_i(\mathbf{x}) \cdot \mathrm{FFN}_i(\mathbf{x})$$

Per-token compute is unchanged (only $k \ll E$ experts run); per-
parameter capacity grows by $E$. See `kb/notes/architecture/moe.md`
for the full treatment. Switch Transformer top-1 routing
`[fedus2021-switch §2.1]`, Mixtral top-2 `[mixtral2024 §2.2]`,
DeepSeekMoE fine-grained + shared experts `[deepseekmoe2024 §3]`.

## 4. Intuitions and analogies

[ANALOGY] The FFN is **a per-token key-value lookup table**: $W_1$
projects up to a wide intermediate space (the "keys"), the activation
selects which "key positions" fire, and $W_2$ projects back to the
residual stream (the "values"). This analogy returns to canonical form
via the explicit form $\mathrm{ReLU}(\mathbf{x} W_1) W_2 = \sum_k
[\mathrm{ReLU}(\mathbf{x} \cdot W_1[:,k])] \cdot W_2[k,:]$ — a sparse
sum of "value rows" weighted by which "key rows" the input activates.
Mechanistic-interpretability work on FFN neurons (Geva et al. 2021,
2022) treats each $d_{\text{ff}}$-index as a "key-value memory"; the
analogy is what motivates the term `[Tier B forum-signal]`.

[INTUITION] **SwiGLU's gain comes from multiplicative interactions.**
Vaswani's FFN computes a sum of features: $\mathrm{ReLU}(\mathbf{x}
W_1) W_2$. SwiGLU computes a *product* of two features:
$\mathrm{Swish}(\mathbf{x} W) \otimes (\mathbf{x} V)$. The product
$g \otimes u$ can express "this feature, only when that feature is
active" patterns that the sum cannot. The canonical form is Eq. 3;
the canonical empirical evidence is the 0.04 perplexity gap in
`[shazeer2020 §3.2; kb/excerpts/shazeer2020#sec-3-2]`.

[INTUITION] The $\tfrac{8}{3}$ rule is **purely a parameter-budget
choice**, not a derivation. Shazeer 2020 wanted to compare GLU variants
at fixed parameter count, so he shrank the inner dimension by $\tfrac{2}{3}$
to absorb the third matrix `[shazeer2020 §2; kb/excerpts/shazeer2020#sec-2-eq6]`.
LLaMA-1 inherited this; later models (PaLM, LLaMA-3) re-tune the FFN
ratio independently of $\tfrac{8}{3}$. No first-principles reason
$\tfrac{8}{3}$ is optimal.

## 5. Frontier and open questions (as of 2026-05)

- **MoE has displaced the dense-FFN-only design.** As of 2026 every
  publicly-disclosed frontier model is MoE (DeepSeek-V3, Llama 4,
  Qwen3 large, Gemini 2.5, Mistral Large). Dense persists at 8B–13B
  open scale (LLaMA-3 8B/70B, OLMo 2). The dense-FFN topic in this
  note covers what each MoE expert *is*. See
  `kb/notes/architecture/moe.md` for routing/specialization/load-
  balancing.
- **Activation re-ablation needed.** SwiGLU's frontier dominance is
  inertial. A frontier-scale (50B+) ablation across SwiGLU/GEGLU/
  ReGLU/Bilinear has not been published.
- **GLU-style attention.** Recent (2024–2025) work applies GLU gating
  to attention itself (e.g., arXiv 2507.00022). Adoption pending.
- **Theory of multiplicative interaction.** Why product-of-features
  beats sum-of-features in deep networks remains theoretically
  underdeveloped — Shazeer 2020's "divine benevolence" line still
  applies in 2026.

## 6. See also

- `kb/notes/architecture/transformer-overview.md` — where the FFN
  sublayer sits (post-attention, post-RMSNorm, residual-add).
- `kb/notes/architecture/moe.md` — MoE replaces this single FFN with
  $E$ parallel SwiGLU experts.
- `kb/notes/architecture/attention-mechanism.md` — the other sublayer.
- `kb/notes/architecture/normalization.md` — RMSNorm precedes the FFN
  in Pre-LN; surrounds it in Peri-LN.
