---
topic: architecture/attention-mechanism
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - vaswani2017
  - shazeer2019
  - ainslie2023
  - dao2022
  - deepseek-v2
secondary_sources:
  - dao2023  # FlashAttention-2
  - shah2024  # FlashAttention-3
  - yuan2025  # Native Sparse Attention
related_topics:
  - architecture/transformer-overview
  - architecture/position-encoding
  - architecture/long-context
  - inference/kv-cache-management
---

# Attention mechanism

Attention is the core inter-token mixing operation of the Transformer.
Every claim in this note is grounded in a primary source listed in
`theory/kb/index/papers.json`; analogies are tagged and always return to
the canonical symbolic form.

## 1. Formal definition

The **scaled dot-product attention** function takes a matrix of queries
$Q \in \mathbb{R}^{N_q \times d_k}$, a matrix of keys $K \in
\mathbb{R}^{N_k \times d_k}$, and a matrix of values $V \in
\mathbb{R}^{N_k \times d_v}$, and produces an output
$O \in \mathbb{R}^{N_q \times d_v}$:

$$O = \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \tag{1}$$

`[vaswani2017 §3.2.1, Eq.1; kb/excerpts/vaswani2017#sec-3-2-1]`

Symbol glossary for this section:

| Symbol | Meaning |
|---|---|
| $N_q$ | number of query positions (in self-attention, $= N$, the sequence length) |
| $N_k$ | number of key/value positions (also $= N$ for self-attention; $\ne N$ for cross-attention) |
| $d_k$ | per-head key/query dimension |
| $d_v$ | per-head value dimension |
| $d_{\text{model}}$ | residual-stream width (often $d$ or $d_{\text{model}}$ in papers) |
| $h$ | number of heads (Vaswani: $h=8$) |
| $W_i^{Q}, W_i^{K}, W_i^{V}$ | per-head projection matrices, learned |
| $W^O$ | output projection matrix, learned |
| $\sqrt{d_k}$ | scaling factor (variance correction; see §2.2) |

The softmax is applied **row-wise**: for each query position $i$, the
row $\mathrm{softmax}(QK^\top/\sqrt{d_k})_i$ is a probability distribution
over the $N_k$ key positions, and $O_i$ is the convex combination of the
value rows under that distribution
`[dao2022 §2.2; kb/excerpts/dao2022#sec-2-2]`.

In **multi-head attention** the function is applied $h$ times in
parallel on linearly projected inputs and the results are concatenated:

$$\mathrm{MultiHead}(X) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W^O$$
$$\mathrm{head}_i = \mathrm{Attention}(X W_i^Q, X W_i^K, X W_i^V) \tag{2}$$

with $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$,
$W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, $W^O \in
\mathbb{R}^{h d_v \times d_{\text{model}}}$. The Vaswani choice
$d_k = d_v = d_{\text{model}}/h$ keeps total compute roughly equal to
single-head full-width attention `[vaswani2017 §3.2.2;
kb/excerpts/vaswani2017#sec-3-2-2]`.

## 2. Mechanism — how the function computes

### 2.1 Step-by-step with tensor shapes

For self-attention on a residual-stream activation $X \in \mathbb{R}^{B
\times N \times d_{\text{model}}}$ (batch $B$, sequence $N$, width
$d_{\text{model}}$):

1. **Project** $Q = X W^Q$, $K = X W^K$, $V = X W^V$. Each is
   $(B, N, h \cdot d_k)$; reshape to $(B, h, N, d_k)$.
2. **Score** $S = QK^\top / \sqrt{d_k}$. Shape $(B, h, N, N)$. This is
   the $N \times N$ matrix that FlashAttention is designed never to
   materialize in HBM.
3. **Mask** (decoder/causal): set $S_{ij} \leftarrow -\infty$ for $j > i$
   so that $\mathrm{softmax}(S)_{ij} = 0$ above the diagonal `[vaswani2017
   §3.2.3]`.
4. **Normalize** $P = \mathrm{softmax}(S)$ row-wise. Shape $(B, h, N, N)$.
5. **Aggregate** $O = PV$. Shape $(B, h, N, d_v)$, reshape to $(B, N,
   h \cdot d_v)$.
6. **Project out** $Y = O W^O$, shape $(B, N, d_{\text{model}})$, returned
   to the residual stream.

Compute cost: $O(B \cdot h \cdot N^2 \cdot d_k)$ FLOPs in steps 2 and 5
combined — the well-known **quadratic in $N$** cost. Memory cost in the
naive implementation: $O(B \cdot h \cdot N^2)$ for $S$ and $P$.

### 2.2 Why the $1/\sqrt{d_k}$ scaling

If the components of $q$ and $k$ are independent with mean 0 and
variance 1, the dot product $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ has
mean 0 and **variance $d_k$**. As $d_k$ grows the pre-softmax logits
spread out, pushing softmax into a regime of vanishing gradients. The
fix is to divide by $\sqrt{d_k}$, restoring unit variance `[vaswani2017
§3.2.1, fn.4; kb/excerpts/vaswani2017#sec-3-2-1]`.

[INTUITION] The softmax is sharp where logits are spread and flat where
they are squashed; in the sharp regime, almost all gradient flows
through the maximum element only. Scaling by $\sqrt{d_k}$ keeps the
distribution "warm" enough at initialization that all positions still
receive non-negligible gradient. This is intuition, not a separate
claim — the canonical quantitative form is the $\mathrm{Var}(q \cdot
k) = d_k$ result above.

### 2.3 Three structural uses

The same multi-head attention block, with different sources for $Q$,
$K$, $V$, plays three roles `[vaswani2017 §3.2.3;
kb/excerpts/vaswani2017#sec-3-2-3]`:

1. **Encoder self-attention:** $Q, K, V$ all come from the previous
   encoder layer; no mask. Every position attends to every position.
2. **Decoder masked self-attention:** $Q, K, V$ from the previous
   decoder layer; causal mask blocks attention from position $i$ to any
   $j > i$, preserving the autoregressive property.
3. **Encoder–decoder cross-attention:** $Q$ from the previous decoder
   layer; $K, V$ from the encoder output. This is how the decoder reads
   the source. Decoder-only LLMs (GPT family, LLaMA, etc.) drop this
   role and keep only role 2.

## 3. Variants and lineage

The mechanism in §1 is fixed; what varies across modern LLMs is
**(a)** how heads share K/V across themselves, and **(b)** how the
function is implemented on hardware. These are orthogonal axes.

### 3.1 The KV-sharing axis: MHA → MQA → GQA → MLA

The KV cache during autoregressive decoding stores one $K_t, V_t$ per
generated token per layer per head. Reloading it each step is the
dominant memory-bandwidth cost
`[shazeer2019 §1; kb/excerpts/shazeer2019#sec-1]`.

Shazeer 2019 quantifies it: the memory/compute ratio in incremental
decoding is $\Theta(n/d + 1/b)$ (for sequence length $n$, model width
$d$, batch $b$). When $n \approx d$ or $b \approx 1$, the ratio
approaches 1 and the layer is bandwidth-bound on modern accelerators
`[shazeer2019 §2.4; kb/excerpts/shazeer2019#sec-2-4]`.

The interventions all share-or-compress the K/V projections across
heads:

| Variant | $K, V$ shape per layer | KV cache / token / layer | Capability |
|---|---|---|---|
| **MHA** (Vaswani 2017) | per-head, $h$ keys & $h$ values | $2 n_h d_h$ | strong |
| **MQA** (Shazeer 2019) | shared, 1 key & 1 value across heads | $2 d_h$ | weak |
| **GQA-$g$** (Ainslie 2023) | $g$ groups, each shared | $2 n_g d_h$ | moderate–strong |
| **MLA** (DeepSeek-V2 2024) | latent $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ + shared RoPE key | $(d_c + d_h^R) \approx \tfrac{9}{2} d_h$ | stronger than MHA |

`[deepseek-v2 §2.1.4 Table 1; kb/excerpts/deepseek-v2#sec-2-1-4]`

**MQA** drops the $h$ dimension from $W^K$ and $W^V$: only queries are
multi-headed, one $K, V$ per layer
`[shazeer2019 §3; kb/excerpts/shazeer2019#mqa-structure]`. The KV cache
shrinks by a factor of $h$. Capacity loss is real and shows up as
training instability and quality regression in larger models
`[ainslie2023 §1; kb/excerpts/ainslie2023#sec-1]`.

**GQA-$g$** divides the $h$ query heads into $g$ groups, each group
sharing one $K, V$ head. GQA-1 = MQA, GQA-$h$ = MHA, intermediate $g$
interpolates
`[ainslie2023 §2.2; kb/excerpts/ainslie2023#sec-2-2]`. Ainslie also
introduce **uptraining**: convert an existing MHA checkpoint to GQA by
mean-pooling K/V projections within each group, then continue
pre-training for $\alpha \approx 5\%$ of the original budget
`[ainslie2023 §2.1; kb/excerpts/ainslie2023#sec-2-1]`. This is the
recipe LLaMA-2/3, Mistral, Qwen, and most decoder-only LLMs from 2023
onward inherited.

**MLA** in DeepSeek-V2 is a structurally different idea: K and V are
not selected from a smaller set of heads, they are **reconstructed from
a low-rank latent**. The cache holds only $\mathbf{c}_t^{KV} = W^{DKV}
\mathbf{h}_t \in \mathbb{R}^{d_c}$ with $d_c \ll d_h n_h$
`[deepseek-v2 §2.1.2 Eq.9; kb/excerpts/deepseek-v2#sec-2-1-2]`. Per-head
$\mathbf{k}_{t,i}^C, \mathbf{v}_{t,i}^C$ are produced at use time by
up-projection $W^{UK}, W^{UV}$, and during inference $W^{UK}$ can be
pre-multiplied (absorbed) into $W^Q$ and $W^{UV}$ into $W^O$, so K and
V never need to be materialized in the forward pass at all
`[deepseek-v2 §2.1.2; kb/excerpts/deepseek-v2#sec-2-1-2]`.

The remaining wrinkle is **decoupled RoPE**. Standard RoPE inserts a
position-dependent rotation between $Q$ and $K$ that prevents the $W^{UK}$-
into-$W^Q$ absorption. DeepSeek-V2's fix is to split each query and key
into two parts: a content part computed from the latent (no RoPE), and
a small per-head $\mathbf{q}_{t,i}^R$ + a single shared
$\mathbf{k}_t^R$ that carry RoPE
`[deepseek-v2 §2.1.3 Eq.14–19; kb/excerpts/deepseek-v2#sec-2-1-3]`.
Total cache: $(d_c + d_h^R) l$ elements per token, comparable to GQA-2
in size but with quality matching or beating MHA in DeepSeek-V2's
ablation `[deepseek-v2 §2.1.4]`.

[CONTRADICTION] DeepSeek-V2 reports MLA quality > MHA quality at
matched cache size; reproduction outside DeepSeek (independent ablations
on other model scales) is sparse as of 2025 and the comparison
methodology is not yet community-standardized. Treat the "stronger than
MHA" claim as DeepSeek-internal until cross-replicated.

### 3.2 The hardware-implementation axis: standard → FlashAttention 1/2/3

Independently of how K/V are shared, the function in Eq. (1) can be
computed in different ways. The standard implementation materializes
$S = QK^\top$ and $P = \mathrm{softmax}(S)$ as $(N \times N)$ matrices in
GPU HBM, costing $O(N^2)$ memory traffic
`[dao2022 §2.2; kb/excerpts/dao2022#sec-2-2]`.

**FlashAttention** (Dao et al. 2022) computes Eq. (1) **exactly** —
same output, no approximation — using two ideas
`[dao2022 §1; kb/excerpts/dao2022#sec-1]`:

1. **Tiling**: process $K, V$ in blocks small enough to fit in on-chip
   SRAM. Use the **online softmax** decomposition to combine block
   results without materializing the full attention matrix
   `[dao2022 §3.1; kb/excerpts/dao2022#sec-3-1]`. The arithmetic is the
   same; the order is reorganized so that the $N \times N$ matrix is
   never written to HBM.
2. **Recomputation**: for the backward pass, store only $O$ and the
   softmax stats $(m, \ell)$ from the forward pass; recompute $S, P$ on
   the fly in SRAM. Saves $O(N^2)$ activations
   `[dao2022 §3.1; kb/excerpts/dao2022#sec-3-1-recompute]`.

The IO complexity is $\Theta(N^2 d^2 M^{-1})$ HBM accesses, where $M$
is SRAM size, vs.\ $\Theta(Nd + N^2)$ for standard attention. For
typical $d \in [64, 128]$ and $M \sim 100$KB, this is many-times fewer
HBM accesses
`[dao2022 §3.2 Theorem 2; kb/excerpts/dao2022#sec-3-2]`. A matching
lower bound (Proposition 3) shows this is optimal among exact
algorithms `[dao2022 §3.2; kb/excerpts/dao2022#sec-3-2-lowerbound]`.

**FlashAttention-2** (Dao 2023, arXiv 2307.08691) re-partitions the
work so that more of the algorithm runs as matmul ops (which utilize
tensor cores well) and reduces non-matmul FLOPs. ~2× over
FlashAttention-1 on A100.

**FlashAttention-3** (Shah et al. 2024, arXiv 2407.08608) targets H100
specifically: asynchronous warp-specialized scheduling, FP8 support,
and use of the H100 TMA engine. ~1.5–2× over FlashAttention-2 on H100,
with FP8 quality preserved by block-quantization.

### 3.3 Other lineage points

- **Sparse / windowed attention** (e.g., Longformer, BigBird) restricts
  $S_{ij}$ to a sparsity pattern. The ratio of dense to sparse changed
  in 2025: DeepSeek's **Native Sparse Attention** (NSA, Yuan et al.
  2025, arXiv 2502.11089) introduces a hardware-aligned natively sparse
  attention with end-to-end training (sparsity is part of pre-training,
  not retrofitted). Treated in `kb/notes/architecture/long-context.md`
  rather than here.
- **Linear / kernel attention** (Performer, Linear Transformer) replaces
  the softmax with a kernel feature map so that
  $\mathrm{softmax}(QK^\top)V$ becomes $\phi(Q)(\phi(K)^\top V)$,
  reducing cost to $O(N)$. Quality has trailed standard attention; not
  used at the frontier as of 2025.
- **State-space alternatives** (Mamba 2024, RWKV) replace attention
  entirely. Treated in `kb/notes/architecture/state-space-models.md`.

## 4. Intuitions and analogies

[ANALOGY] Attention is often described as a **soft dictionary lookup**:
queries are search keys, keys are dictionary keys, values are dictionary
values, and softmax produces a soft argmax over keys that returns a
weighted blend of values. The analogy returns to canonical form in §1:
the "softness" is exactly the row-wise softmax in Eq. (1), and the
"blend" is exactly the $PV$ product. The dictionary picture is helpful
for orienting students; it is not a separate claim about what attention
"is."

[INTUITION] Multi-head attention is not just "more attention." Each
head has its own $W_i^Q, W_i^K, W_i^V$, so heads can specialize on
different relations — syntactic agreement, coreference, copy patterns,
etc. The Vaswani motivation is that "averaging inhibits this" with a
single head `[vaswani2017 §3.2.2; kb/excerpts/vaswani2017#sec-3-2-2-motivation]`.
The canonical math doesn't say what each head learns; that empirical
question is the subject of mechanistic interpretability (see
`kb/notes/interpretability/mechanistic-interpretability.md`).

[ANALOGY] The KV cache is the model's **scratchpad of past keys and
values**. During training all $N$ tokens are computed in parallel; the
"cache" is fictitious. During autoregressive decoding the model writes
one new $(K_t, V_t)$ per token per layer per head and reads the entire
prefix at every step. MQA/GQA/MLA are all answers to the question
"what's the smallest scratchpad that still preserves quality?"

[INTUITION] FlashAttention is **not faster math**; it is the **same
math, reordered for the memory hierarchy**. The FLOP count of Eq. (1)
is unchanged (in fact slightly higher, due to recomputation in the
backward pass). The wall-clock win comes from never paying the HBM
round-trip on the $N \times N$ matrix. This is why FlashAttention is
called *exact*: every output is bit-equivalent to a standard
implementation up to floating-point reduction order.

## 5. Frontier and open questions (as of 2026-05)

- **MLA generalization beyond DeepSeek.** As of 2026, DeepSeek-V2/V3,
  DeepSeek-R1, and a growing number of Chinese labs use MLA. Western
  frontier labs (Anthropic, OpenAI, Google) do not publicly disclose
  attention variants but external GGUF/inference-stack work suggests
  GQA remains dominant. [CONTRADICTION] Whether MLA-style latent
  compression generalizes beyond the DeepSeek training recipe is an
  open question; the original ablation is single-source.
- **NSA vs.\ dense at long context.** Yuan et al.\ 2025 argue that
  pre-training with native sparse attention can match or exceed dense
  attention quality at 64k+ contexts while reducing compute. This is
  recent and the reproductions are still emerging.
- **Differential Transformer.** Ye et al.\ 2024 (Microsoft) propose
  computing attention as the difference of two softmax maps, claiming
  reduced "attention noise" and improved long-context retrieval.
  Status: promising single-paper result; not yet adopted at frontier.
- **Linear attention re-emergence.** RetNet, Mamba-2, and gated linear
  attention variants are blurring the line between attention and
  state-space models. Hybrid (attention + linear) architectures are
  active in 2025–2026. See `kb/notes/architecture/state-space-models.md`.
- **Information-flow analysis.** Mechanistic interpretability work on
  induction heads, copy heads, and circuit-level attention patterns
  continues; see `kb/notes/interpretability/mechanistic-interpretability.md`.

## 6. See also

- `kb/notes/architecture/transformer-overview.md` — where attention sits
  in the block.
- `kb/notes/architecture/position-encoding.md` — how positions are
  injected into attention; RoPE is the substrate that MLA's decoupled-
  RoPE strategy must work around.
- `kb/notes/architecture/long-context.md` — sparse, sliding, and native-
  sparse variants for $N \gg 4096$.
- `kb/notes/inference/kv-cache-management.md` — paging, eviction,
  speculative-decode interaction with the KV cache. The MQA/GQA/MLA
  story above is upstream of all of this.
- `kb/notes/interpretability/mechanistic-interpretability.md` — what
  individual heads learn, induction heads, attention circuits.
