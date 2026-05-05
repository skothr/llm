---
topic: architecture/normalization
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - vaswani2017
  - zhang2019
  - xiong2020
  - peri-ln2025
  - olmo2
secondary_sources:
  - touvron2023
  - meta-llama3
  - gemma3
  - deepseek-v3
related_topics:
  - architecture/transformer-overview
  - architecture/attention-mechanism
  - training/mixed-precision-and-stability
---

# Normalization

Normalization layers are the **stability fix** that lets deep
Transformers train at all. Three axes vary across modern LLMs: **type**
(LayerNorm vs RMSNorm), **placement** (Post-LN, Pre-LN, Peri-LN), and
**special-purpose use** (QK-Norm on attention queries/keys). The
modern (2023+) frontier consensus is **RMSNorm + Pre-LN** as the
LLaMA-1 baseline, with **Peri-LN** and **QK-Norm** added in 2024–2025
production models for long-context stability.

## 1. Formal definition

### 1.1 LayerNorm

For input $\mathbf{x} \in \mathbb{R}^{d}$, LayerNorm computes the mean
and variance across the $d$-dim feature axis and normalizes:

$$\mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^{d}(x_i - \mu)^2 \tag{1}$$

$$\mathrm{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \boldsymbol{\gamma} + \boldsymbol{\beta} \tag{2}$$

with **learnable affine parameters** $\boldsymbol{\gamma}, \boldsymbol{\beta} \in \mathbb{R}^d$
and $\epsilon$ a small constant for numerical stability. Introduced by
Ba, Kiros & Hinton 2016 `[ba2016-layernorm §3.1; kb/excerpts/ba2016-layernorm#sec-3-1]`.

### 1.2 RMSNorm — drop the centering

RMSNorm (Zhang & Sennrich 2019) drops the mean-subtraction step:

$$\mathrm{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2} \tag{3}$$

$$\mathrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\mathrm{RMS}(\mathbf{x}) + \epsilon} \odot \boldsymbol{\gamma} \tag{4}$$

`[zhang2019 §3.1; kb/excerpts/zhang2019#sec-3-1]` — only the scale
$\boldsymbol{\gamma}$ remains; no shift $\boldsymbol{\beta}$. The
mean-subtraction in LN is dropped entirely, which Zhang & Sennrich
report **runs 7–64% faster** at matched quality
`[zhang2019 §1; kb/excerpts/zhang2019#sec-1]`.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | input feature vector ($d$-dim) |
| $d$ | feature dimension; for residual stream, $d = d_{\text{model}}$ |
| $\mu, \sigma^2$ | mean and variance across feature axis |
| $\boldsymbol{\gamma}, \boldsymbol{\beta}$ | learnable scale and shift parameters |
| $\epsilon$ | numerical stability constant ($\sim 10^{-5}$) |
| $\mathrm{Norm}(\cdot)$ | generic norm (either LN or RMSNorm) |

### 1.3 QK-Norm

A *targeted* RMSNorm applied to **queries and keys** before the
attention dot product:

$$\mathbf{Q} = \mathrm{RMSNorm}(\mathbf{Q}), \quad \mathbf{K} = \mathrm{RMSNorm}(\mathbf{K})$$

inside each attention head. Introduced for vision Transformers
(Henry et al. 2020) and adopted into LLM training stabilization in
OLMo 2 and LLaMA 3.1+ `[olmo2 §3.1]`.

## 2. Mechanism

### 2.1 Three placements

For a sublayer $F$ (attention or FFN) with input $\mathbf{x}$:

**Post-LN (Vaswani 2017 original):**
$$\mathbf{y} = \mathrm{Norm}(\mathbf{x} + F(\mathbf{x})) \tag{5}$$

`[vaswani2017 §5.4; kb/excerpts/vaswani2017#sec-5-4]`. Norm sits *on
the residual path* — every block accumulates norm-magnitude effects.

**Pre-LN (Xiong et al. 2020 / GPT-2 onward):**
$$\mathbf{y} = \mathbf{x} + F(\mathrm{Norm}(\mathbf{x})) \tag{6}$$

`[xiong2020 §3.1; kb/excerpts/xiong2020#sec-3-1]`. Norm sits *off the
residual path* — the residual stream itself is never normalized in-
place; only the sublayer input is.

**Peri-LN (Park et al. 2025):**
$$\mathbf{y} = \mathbf{x} + \mathrm{Norm}(F(\mathrm{Norm}(\mathbf{x}))) \tag{7}$$

`[peri-ln2025 §3]` — norm both before and after the sublayer. The
Vaswani-original applied Post-LN; Peri-LN composes both.

### 2.2 Why Pre-LN displaced Post-LN

Xiong et al. 2020 give a mean-field analysis of the gradient at
initialization `[xiong2020 §3.2; kb/excerpts/xiong2020#sec-3-2]`:

- **Post-LN** gradient at the input layer scales as $O(d \sqrt{L})$ with
  depth $L$ — exponential blow-up at increasing depth without
  warmup.
- **Pre-LN** gradient at the input layer scales as $O(d \sqrt{\ln L / L})$
  — bounded with depth.

Practically: Post-LN training on GPT-3-scale models requires careful
learning-rate warmup; Pre-LN trains stably **without warmup**
`[xiong2020 §1; kb/excerpts/xiong2020#sec-1]`. This was the headline
result that pushed every 2020+ decoder-only LLM to Pre-LN
`[radford2019-gpt2 §2.3; kb/excerpts/radford2019-gpt2#sec-2-3]`.

### 2.3 Pre-LN's failure mode: residual drift

Pre-LN has a known issue: the **residual stream norm grows with depth**
because every block's output is added (without renormalization) to
$\mathbf{x}$. By layer $L$ the stream norm is approximately
$\|\mathbf{x}^L\| \sim \sqrt{L} \|\mathbf{x}^0\|$, which can saturate
later sublayers' inputs to RMSNorm-of-very-large-numbers. The fix is a
**final RMSNorm** before the LM head, which is universal in modern
LLMs `[touvron2023 §2; meta-llama3 §3.1]`.

### 2.4 Peri-LN — restoring per-block normalization

Park et al. 2025 argue that Pre-LN's residual drift is the actual
limit, not Post-LN's gradient explosion `[peri-ln2025 §3]`. Peri-LN
adds a second norm *after* the sublayer (Eq. 7) that bounds the
sublayer's contribution to the residual stream. Empirically, Peri-LN
trains more stably than Pre-LN at $L > 50$ blocks and eliminates the
need for the final-RMSNorm hack. Adopted by Gemma 3
`[gemma3 §2.1]` and OLMo 2 `[olmo2 §3.1]`.

### 2.5 QK-Norm — fixing attention-logit explosion

Long-context training (sequence lengths $> 32\text{K}$) destabilizes
because the unnormalized $\mathbf{Q} \mathbf{K}^\top$ matrix can take
on extreme values when individual head dimensions accumulate large
products. Applying RMSNorm to $\mathbf{Q}$ and $\mathbf{K}$ before the
dot product bounds the post-softmax-temperature regime
`[olmo2 §3.1]`. Adopted by OLMo 2, LLaMA 3.1+, and informally widely.

## 3. Variants and lineage

### 3.1 Comparison table

| Model | Norm type | Placement | QK-Norm? | Source |
|---|---|---|---|---|
| Vaswani 2017 | LayerNorm | Post-LN | no | `[vaswani2017 §5.4]` |
| GPT-2 | LayerNorm | Pre-LN | no | `[radford2019-gpt2 §2.3]` |
| BERT | LayerNorm | Post-LN | no | `[devlin2019-bert §3.1]` |
| GPT-3 | LayerNorm | Pre-LN | no | `[brown2020 §2.1]` |
| **LLaMA-1** | **RMSNorm** | **Pre-LN** | no | `[touvron2023 §2]` |
| LLaMA-2/3 | RMSNorm | Pre-LN | no (LLaMA-3.1+ yes) | `[meta-llama3 §3.1]` |
| Mistral 7B | RMSNorm | Pre-LN | no | `[jiang2023 §2]` |
| **OLMo 2** | **RMSNorm** | **Peri-LN** | **yes** | `[olmo2 §3.1]` |
| **Gemma 3** | RMSNorm | **Peri-LN** | yes | `[gemma3 §2.1]` |
| DeepSeek-V3 | RMSNorm | Pre-LN | no | `[deepseek-v3 §2.1]` |
| Qwen3 | RMSNorm | Pre-LN | yes | `[qwen3 §2.1]` |

The 2023–2024 frontier converged on **RMSNorm + Pre-LN**; 2025+ is
splitting between **RMSNorm + Pre-LN + QK-Norm** (DeepSeek, Qwen3) and
**RMSNorm + Peri-LN** (Gemma, OLMo).

### 3.2 RMSNorm vs LayerNorm — what's actually dropped

RMSNorm = LayerNorm with $\boldsymbol{\beta} = 0$ and $\mu = 0$. The
empirical fact is that **dropping the mean-subtraction doesn't hurt**
`[zhang2019 §4; kb/excerpts/zhang2019#sec-4]`. Theoretical
explanations:

- LayerNorm's mean-subtraction projects onto the hyperplane orthogonal
  to $\mathbf{1}$; RMSNorm doesn't, so RMSNorm preserves a specific
  rank-1 component (the mean).
- Geometric work (arXiv 2409.12951) argues LayerNorm is *not*
  invertible (mean-subtraction is rank-deficient) while RMSNorm is.
- Practically: the bias parameter $\boldsymbol{\beta}$ is a learnable
  shift that the model rarely uses non-trivially; dropping it costs
  little.

### 3.3 The Post-LN comeback (Jan 2026)

Chen & Wei 2026 (arXiv 2601.19895) report that **Post-LN can train
stably with new initialization and regularization techniques** —
DeepNorm-style scaled residuals + careful warmup — and offers
expressivity advantages at very deep ($L > 100$) networks. As of
2026-05 this is a single result; not yet adopted at frontier scale.
[CONTRADICTION] Phase 1 sweep §3 lists "Post-LN is unstable" as a
stale prior; this 2026 paper rehabilitates it. The story is now:
Pre-LN trains; Peri-LN is more stable; Post-LN with the right
initialization is competitive again. The earlier Xiong et al.
analysis identified one failure mode; new work addresses it.

### 3.4 DeepNorm (alternative Post-LN stabilizer)

Wang et al. 2022 (DeepNet, arXiv 2203.00555) propose a per-layer
scaling factor $\alpha$ on the residual addition that lets Post-LN
train stably to $L = 1000$ layers `[Tier B reference]`. Used in
some Microsoft models; not in modern open frontier LLMs.

### 3.5 Norm-free and norm-with-gating designs

"Gated Removal of Normalization" (arXiv 2602.10408, 2026) replaces the
norm layer with a learned gating operation that achieves comparable
stability. Position: a 2026 result, not yet widely adopted.

## 4. Intuitions and analogies

[ANALOGY] Normalization is **a ruler the model holds against itself**:
before applying any non-linearity, normalize the input vector to a
known scale, so the non-linearity behaves predictably regardless of
where the residual stream has drifted. The analogy returns to canonical
form via Eq. 4: RMSNorm scales every input vector to RMS = 1 along
the feature axis (then re-applies a learned per-channel scale
$\boldsymbol{\gamma}$). Without it, the residual stream's magnitude
explodes with depth and the activations saturate.

[INTUITION] **Why mean-subtraction was a hold-over from BatchNorm**:
LayerNorm was proposed (Ba et al. 2016) as a layerwise analog of
BatchNorm. BatchNorm's centering matters because batch statistics
include the data mean, which is non-zero for natural images.
LayerNorm's centering normalizes across feature axis within a single
example; the "mean" being subtracted is a feature-axis quantity that's
already arbitrary. RMSNorm dropping it is **stripping a vestigial
operation** that BatchNorm justified but LayerNorm didn't.

[INTUITION] **Pre-LN vs Post-LN as a residual-stream architecture
choice**: Post-LN normalizes the residual stream itself; the stream
"resets" each block. Pre-LN keeps the residual stream raw and only
normalizes the *input to each sublayer*. The Pre-LN stream is therefore
the natural object of study for mechanistic interpretability — you can
tap a layer's residual without worrying about what norm did to it. Post-
LN obscures this. The canonical recurrence is Eq. 5 vs Eq. 6.

[INTUITION] **Peri-LN as best of both**: Pre-LN's residual stream is
unnormalized (so taps are clean), but its sublayer outputs are
unnormalized (so they can pile up). Peri-LN normalizes the sublayer
output before adding to the stream — the stream remains "raw enough"
for interpretability while preventing magnitude blowup. The canonical
form is Eq. 7.

## 5. Frontier and open questions (as of 2026-05)

- **Pre-LN vs Peri-LN at very-deep frontier scale.** Frontier models
  in 2026 split: DeepSeek/Qwen3/LLaMA stay Pre-LN; Gemma 3 and OLMo 2
  switch to Peri-LN. No head-to-head ablation at $\geq 70\text{B}$ has
  been published.
- **QK-Norm's interaction with RoPE.** RoPE rotates $Q, K$ before the
  dot product; applying RMSNorm before the rotation vs after vs both
  is implementation-specific; the trade-offs are not well-characterized.
- **Post-LN rehabilitation transferring to frontier.** arXiv
  2601.19895 (Jan 2026) is encouraging but small-scale. Whether
  frontier labs adopt is open. [CONTRADICTION] Vaswani 2017 used
  Post-LN successfully, the field moved away citing instability,
  and a 2026 paper says it works again with new init — the meta-
  question of "what changed" is itself open.
- **Norm-free architectures.** Gated removal (arXiv 2602.10408) and
  ReZero-style designs explore whether the normalization layer is
  fundamentally necessary; answer is "probably yes but the form can
  vary."
- **Why RMSNorm works as well as LayerNorm.** The geometric
  explanation (arXiv 2409.12951) is partial; a full theory of
  centering's non-role in deep networks doesn't exist.

## 6. See also

- `kb/notes/architecture/transformer-overview.md` — placement of
  norm in the modern decoder-only block.
- `kb/notes/architecture/attention-mechanism.md` — QK-Norm sits inside
  the attention block, before the $QK^\top$ dot product.
- `kb/notes/architecture/ffn.md` — RMSNorm precedes the FFN sublayer
  in Pre-LN; surrounds it in Peri-LN.
- `kb/notes/training/mixed-precision-and-stability.md` —
  normalization is one of several training-stability levers; init
  scaling, gradient clipping, weight decay are complements.
