---
topic: architecture/position-encoding
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - vaswani2017
  - su2021
  - meta-llama4
secondary_sources:
  - peng2023-yarn  # YaRN; not yet excerpted
  - longrope2-2025
  - rope-to-nope-2025
related_topics:
  - architecture/attention-mechanism
  - architecture/long-context
  - architecture/transformer-overview
---

# Position encoding

Attention is permutation-equivariant: with no positional signal, the
attention output for a sequence $(x_1, \ldots, x_N)$ is invariant under
any permutation of positions. The model would treat "dog bites man" and
"man bites dog" identically. Position encoding is the mechanism that
breaks this symmetry. The space of choices is the topic of this note;
since 2021 it has converged on **rotary position embedding (RoPE)** for
decoder-only LLMs, with extension/extrapolation methods (NTK-aware
scaling, YaRN, LongRoPE2, iRoPE) as the current frontier.

## 1. Formal definition — the four families

Let $\boldsymbol{x}_m \in \mathbb{R}^d$ denote the residual-stream
activation at position $m$. A position encoding is a pair of functions
$f_q, f_k: \mathbb{R}^d \times \mathbb{N} \to \mathbb{R}^{d_k}$ such that
the attention dot product

$$\langle f_q(\boldsymbol{x}_m, m), f_k(\boldsymbol{x}_n, n)\rangle$$

depends on the positions $(m, n)$ in some desired way
`[su2021 §3.1 Eq.11; kb/excerpts/su2021#sec-3-1]`.

### 1.1 Sinusoidal absolute encoding (Vaswani 2017)

The original choice is **additive**:

$$f_{\{q,k\}}(\boldsymbol{x}_m, m) = \boldsymbol{W}_{\{q,k\}} (\boldsymbol{x}_m + \boldsymbol{p}_m) \tag{1}$$

with deterministic sinusoidal $\boldsymbol{p}_m \in \mathbb{R}^d$:

$$p_{m, 2t}     = \sin(m / 10000^{2t/d}), \quad p_{m, 2t+1}   = \cos(m / 10000^{2t/d}) \tag{2}$$

`[vaswani2017 §3.5; su2021 §2.2 Eq.4; kb/excerpts/su2021#sec-2-2]`. The
encoding is added once at the input and propagates implicitly through
all layers.

### 1.2 Learned absolute encoding (BERT, GPT-2)

Same form as (1) but $\boldsymbol{p}_m$ is a learned embedding indexed by
$m \in \{0, 1, \ldots, L-1\}$, where $L$ is a hard-coded maximum
sequence length `[su2021 §2.2; kb/excerpts/su2021#sec-2-2]`. Cannot
extrapolate past $L$.

### 1.3 Relative position encoding (Shaw 2018, T5 bias)

The key/value projections receive a per-(query, key)-distance bias term
or vector. T5 uses an additive scalar bias $b_{m-n}$ per attention head,
binned by relative distance:

$$\boldsymbol{q}_m^\top \boldsymbol{k}_n = \boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{W}_k \boldsymbol{x}_n + b_{m-n} \tag{3}$$

`[su2021 §2.3 Eq.8; kb/excerpts/su2021#sec-2-2]`. ALiBi is a parameter-
free instance: $b_{m-n} = -|m-n| \cdot s_h$ with a fixed slope $s_h$ per
head, giving a "linear penalty for distance" with built-in length
extrapolation.

### 1.4 Rotary position embedding (RoPE) — the modern default

The attention dot product is multiplicatively rotated:

$$f_{\{q,k\}}(\boldsymbol{x}_m, m) = \boldsymbol{R}^d_{\Theta, m} \boldsymbol{W}_{\{q,k\}} \boldsymbol{x}_m \tag{4}$$

where $\boldsymbol{R}^d_{\Theta, m}$ is a block-diagonal rotation matrix
with 2×2 blocks $\bigl(\begin{smallmatrix}\cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i\end{smallmatrix}\bigr)$ for $i = 1, \ldots, d/2$, and the
preset frequencies are

$$\Theta = \{\theta_i = b^{-2(i-1)/d}, \; i \in [1, \ldots, d/2]\}, \quad b = 10000 \text{ (default base)} \tag{5}$$

`[su2021 §3.2.2 Eq.14–15; kb/excerpts/su2021#sec-3-2-2]`. The crucial
property — *position information appears only as the relative offset* —
follows from orthogonality:

$$\boldsymbol{q}_m^\top \boldsymbol{k}_n = \boldsymbol{x}_m^\top \boldsymbol{W}_q^\top \boldsymbol{R}^d_{\Theta, n - m} \boldsymbol{W}_k \boldsymbol{x}_n \tag{6}$$

`[su2021 §3.2.2 Eq.16; kb/excerpts/su2021#sec-3-2-2]`. The encoding is
applied **inside attention**, on $Q$ and $K$ only (not $V$), at every
layer.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $d$ | per-head dimension ($d_k$ in attention notation); must be even |
| $b$ | RoPE base frequency, often 10000 in original LLaMA, 500000 in LLaMA-3, 1e6+ in long-context variants |
| $\theta_i$ | $i$-th frequency, geometric over $i$ |
| $m, n$ | absolute token positions (training-time) |
| $\boldsymbol{R}^d_{\Theta, m}$ | the orthogonal $d\times d$ rotation matrix at position $m$ |

## 2. Mechanism — how RoPE is computed in practice

### 2.1 The pair-rotation form (production implementation)

The block-diagonal matrix $\boldsymbol{R}^d_{\Theta, m}$ is sparse —
$d$ nonzeros per row — so the matmul is implemented elementwise. For
$\boldsymbol{x} \in \mathbb{R}^d$, RoPE applies pair-wise rotation:

$$\boldsymbol{R}^d_{\Theta, m} \boldsymbol{x} = \boldsymbol{x} \otimes \cos(m\boldsymbol{\theta}) + \mathrm{rotate\_half}(\boldsymbol{x}) \otimes \sin(m\boldsymbol{\theta}) \tag{7}$$

where $\boldsymbol{\theta}$ is the length-$d$ vector with $\theta_i$
appearing twice (once per element of pair $i$), and
$\mathrm{rotate\_half}(\boldsymbol{x})$ swaps each pair $(x_{2i-1}, x_{2i}) \mapsto (-x_{2i}, x_{2i-1})$
`[su2021 §3.4.2 Eq.34; kb/excerpts/su2021#sec-3-4-2]`. Two element-wise
multiplies and one add per query/key — negligible overhead.

In Hugging Face transformers, this is the
`apply_rotary_pos_emb(q, k, cos, sin)` helper. In llama.cpp, this is
the `ggml_rope_*` family of ops.

### 2.2 The frequency spectrum

The geometric spacing $\theta_i = b^{-2(i-1)/d}$ means dimension-pair $i$
rotates with **wavelength** $\lambda_i = 2\pi / \theta_i = 2\pi b^{2(i-1)/d}$. For $d=128, b=10000$:

- $\lambda_1 = 2\pi$ (one rotation per token — high frequency, encodes
  fine local order)
- $\lambda_{d/2} = 2\pi \cdot 10000^{(d-2)/d} \approx 2\pi \cdot 10000$
  (very long wavelength — encodes coarse global position)

**Long-term decay** (RoPE §3.3): the inner product of rotated query and
key tends to decay as $|m - n|$ grows, because rotations at different
frequencies dephase relative to each other. This matches the intuition
that distant tokens should interact less
`[su2021 §3.3; kb/excerpts/su2021#sec-3-3]`.

## 3. Variants and lineage — RoPE extension and extrapolation

The pre-trained context length $L_{\text{train}}$ is set by training data
and compute. Deploying at $L_{\text{test}} > L_{\text{train}}$ requires
some adjustment because the model has never seen positions $m > L_{\text{train}}$,
so the rotations there are out-of-distribution. The extension methods
fall into four buckets.

### 3.1 Position interpolation (PI, Chen et al. 2023)

For desired ratio $s = L_{\text{test}} / L_{\text{train}} > 1$, scale
positions: replace $m\theta_i$ with $(m/s)\theta_i$ in (4)–(5). This
maps the new range into the training-seen range. Cheap (no retraining
needed for short extensions) but degrades short-context quality
because it compresses the high-frequency dimensions
[FORUM-SIGNAL: amaarora 2025-09-21 https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html].

### 3.2 NTK-aware scaling and YaRN (Peng et al. 2023, arXiv 2309.00071)

The NTK-aware insight: high-frequency dimensions $\theta_i$ (small $i$)
are local — they have many full rotations in $L_{\text{train}}$ — and
should not be interpolated (it harms local fidelity). Low-frequency
dimensions (large $i$) make less than one rotation in $L_{\text{train}}$
and benefit from interpolation. The fix is to scale the **base** $b$
instead of positions:

$$b' = b \cdot s^{d/(d-2)} \tag{8}$$

This compresses primarily the low-frequency dimensions while leaving
high-frequency ones nearly intact. **YaRN** (Yet another RoPE extensioN
method) adds a per-frequency interpolation schedule: dimensions with
wavelength shorter than the training context use no scaling, dimensions
longer than the extended context use full PI scaling, and intermediate
ones use a smooth ramp. YaRN also rescales attention logits by
$1/\sqrt{t}$ where $t$ depends on the scaling factor, to preserve the
softmax temperature regime. YaRN reports ~10× fewer fine-tuning tokens
than PI to extend LLaMA-2 to 128k. [The YaRN PDF was not accessible at
the time of this writing; equations 7–9 above are reconstructed from
secondary sources and the LLaMA-3 implementation. Verify before
propagating.]

### 3.3 LongRoPE / LongRoPE2 (Ding et al. 2024 / 2025)

**LongRoPE**: evolutionary search over per-frequency scaling factors,
allowing non-uniform rescaling that no analytical method can produce.
Reaches 2M tokens.

**LongRoPE2** (arXiv 2502.20082, Feb 2025): claims near-lossless
extension — preserves 98.5% of short-context performance — at 80× fewer
tokens than Meta's earlier LLaMA-3 long-context fine-tuning approach.
[Unverified against PDF in this pass — flagged for follow-up.]

### 3.4 iRoPE (interleaved RoPE + NoPE; LLaMA 4, Jan 2026)

LLaMA 4's position-encoding scheme alternates between RoPE-equipped
attention layers and **NoPE** (no positional encoding) attention layers,
in a 3:1 ratio. Combined with a temperature-scaling factor in the NoPE
layers, the model trained at 256K extrapolates to a claimed 10M-token
inference context — a qualitative regime change, not just better
interpolation. [The Llama 4 PDF was not accessible at the time of this
writing; mechanism described from the Phase 1 sweep and the "Rope to
Nope" companion paper (arXiv 2501.18795). The exact temperature schedule
and the theoretical justification of the 3:1 ratio are open questions.]

[CONTRADICTION] Whether NoPE layers actually improve extrapolation is
contested. The original "NoPE" claim (Kazemnejad et al. 2023, "The
Impact of Positional Encoding on Length Generalization") was that
decoder-only Transformers trained without any positional encoding
generalize better than RoPE/ALiBi to longer sequences on synthetic
length-generalization tasks. The "Rope to Nope and Back Again"
(arXiv 2501.18795) paper qualifies this: hybrid mixing helps on
realistic LLM workloads where pure NoPE underperforms. iRoPE is the
production realization of that mixing.

### 3.5 Comparison table

| Encoding | Form | Where applied | Context-extension cost | Used by |
|---|---|---|---|---|
| Sinusoidal | additive | input | hard (no smooth extension) | original Transformer, BERT |
| Learned absolute | additive | input | none — hard cutoff at $L$ | GPT-2, OPT |
| T5 relative bias | additive | each-layer attention | requires retraining for new bins | T5, mT5 |
| ALiBi | additive | each-layer attention | free (linear extrapolation) | BLOOM, MPT |
| **RoPE** | multiplicative (rotation) | $Q, K$ at each layer | PI / NTK / YaRN / LongRoPE | LLaMA 1–3, Mistral, Qwen, DeepSeek, Gemma, OLMo |
| **iRoPE** | RoPE + NoPE interleave | mixed | trained natively | LLaMA 4 |
| 2D / M-RoPE | RoPE generalized to 2D | image patches in VLMs | n/a | Qwen2-VL, Qwen2.5-VL, Qwen3-VL |

## 4. Intuitions and analogies

[ANALOGY] RoPE is most easily understood as a **clock with $d/2$
hands**, each ticking at a different rate. At position $m$, hand $i$
points to angle $m\theta_i$. A query at position $m$ and a key at
position $n$ "agree" on dimension-pair $i$ when their hands align modulo
$2\pi$ — i.e., when $(m-n)\theta_i \equiv 0$. The dot product $q_m \cdot
k_n$ is a sum across all $d/2$ pairs, weighted by how aligned each pair
is. The analogy returns to the canonical form in §1.4: hand alignment
is exactly the cosine factor $\cos((m-n)\theta_i)$ that emerges from
expanding $\boldsymbol{x}_m^\top \boldsymbol{R}^d_{\Theta, n-m} \boldsymbol{x}_n$
in the canonical 2×2 block basis.

[INTUITION] The reason RoPE extends so much better than learned
absolute embeddings is that the encoding is **defined by the closed-form
rotation matrix**, not by a learned vector indexed by position. The
network does not need to "see" position $m = 100000$ during training to
have a meaningful $\boldsymbol{R}^d_{\Theta, 100000}$ at inference —
the matrix is computable. What can fail is that *attention features
trained with rotations from the small-$m$ regime may not generalize* to
the high-$m$ regime where the dimension-pair phases are
out-of-distribution. PI/NTK/YaRN are all about reshaping the
inference-time frequency spectrum so attention features don't go
out-of-distribution.

[INTUITION] **Why high-frequency dimensions are local.** Dimension-pair
$i = 1$ has $\theta_1 = 1$ (with $b = 10000$, this is roughly 1 radian
per token). Two tokens 1 apart already produce a phase difference of
~57°; tokens 6 apart produce ~360° (full rotation, nearly indistinguishable
from 0). So the highest-frequency dimensions only meaningfully
discriminate within ~6 tokens — they encode local order. The
lowest-frequency dimension has $\theta_{d/2} \approx 1/10000$, so its
phase differences across the whole 4096-token training context are
under 0.4 radians — it encodes coarse position. NTK-aware scaling
exploits this: the local dimensions don't need extension, the global
ones do.

[ANALOGY] **NoPE layers as the "memory of where things were earlier."**
[SPECULATION] One framing of the iRoPE pattern is that RoPE layers
provide explicit local-order constraints, while NoPE layers are forced
to derive position from content alone (e.g., from special tokens, from
linguistic structure, from the residual stream signal accumulated
through prior RoPE layers). At long context, the explicit
$(m-n)\theta_i$ phase difference becomes unreliable due to OOD
frequencies, so the NoPE layers provide a "fallback" channel that
doesn't depend on the rotation. This is speculation — the LLaMA 4
paper's own theoretical justification, if any, is not in this KB yet.

## 5. Frontier and open questions (as of 2026-05)

- **iRoPE generalization.** LLaMA 4 reports 256K-train → 10M-inference
  extrapolation. As of May 2026, no other public model has replicated
  the iRoPE pattern at scale. Whether it transfers to other model
  families and training recipes, or is specific to LLaMA 4's data and
  hyperparameters, is open.
- **RoPE base-frequency tuning.** LLaMA-3 uses base 500000 (vs LLaMA-2's
  10000), Qwen3 uses 1e6 in some variants. The base controls the
  longest "naturally seen" wavelength and thus the implicit context
  ceiling. There is no agreed-upon formula for choosing it as a
  function of target context length; current practice is empirical.
- **2D / M-RoPE for vision.** Qwen2-VL introduced **M-RoPE**: split the
  $d_k$ channel into three thirds and apply separate RoPE rotations on
  the temporal, height, and width axes. Qwen2.5-VL extends to dynamic
  FPS; Qwen3-VL uses **Interleaved-MRoPE** (interleaving the three
  axes per dimension-pair instead of partitioning the channel range).
  Treated in detail in `kb/notes/architecture/multimodal-llm-extensions.md`.
- **Differential position information.** [SPECULATION] Several recent
  papers explore *learned* per-frequency scaling factors as an
  alternative to fixed-base RoPE. Whether the learned approach actually
  generalizes better than the analytic NTK/YaRN family is contested.
- **NoPE theory.** Kazemnejad et al. 2023 proved that NoPE decoder-only
  Transformers can implicitly recover positional information from the
  causal mask alone (the "first token" is uniquely identifiable from
  having empty attention context). The mechanism by which this scales
  to position-aware computation in deep networks is not well
  characterized.

## 6. See also

- `kb/notes/architecture/attention-mechanism.md` — RoPE is applied
  inside the attention block on $Q, K$; the decoupled-RoPE strategy in
  MLA (DeepSeek-V2) is upstream-aware of this note's machinery.
- `kb/notes/architecture/long-context.md` — extension methods (YaRN,
  LongRoPE2, iRoPE) are the substrate for million-token inference;
  position encoding is one of three axes of long-context engineering
  alongside sparse attention and KV-cache compression.
- `kb/notes/architecture/transformer-overview.md` — where position
  encoding sits in the block.
- `kb/notes/architecture/multimodal-llm-extensions.md` — M-RoPE and
  Interleaved-MRoPE for vision tokens.
