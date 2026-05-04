---
topic: architecture/long-context
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - yuan2025
  - ring-attention-2023
  - longrope2-2025
  - meta-llama4
secondary_sources:
  - dao2022  # FlashAttention is the prerequisite kernel
  - infini-attention-2024
  - su2021  # RoPE; iRoPE / YaRN extension methods
related_topics:
  - architecture/attention-mechanism
  - architecture/position-encoding
  - inference/kv-cache-management
---

# Long context

"Long context" refers to extending the inference-time sequence length
$N_{\text{infer}}$ well beyond the pretraining length $N_{\text{train}}$.
As of mid-2026 the frontier-model norms are roughly:

| Model | Native train ctx | Claimed inference ctx |
|---|---|---|
| GPT-4o, Claude 3.5+ | undisclosed | 128K–200K |
| Gemini 2.5 Pro | undisclosed | 1M (some users 2M) |
| LLaMA 4 Scout / Maverick | 256K | 10M (Scout) |
| DeepSeek-V3.2 | 128K | 128K (full attention; sparse via DSA) |
| Qwen3 | 32K + YaRN | 256K |

`[Phase 1 sweep §2 long-context; theory/plans/landscape-sweep/architecture.md]`

The cost barriers to long context are three: (i) attention's $N^2$ FLOP
cost; (ii) attention's $N$-linear KV-cache memory cost during decode;
(iii) position-encoding distribution-shift past $N_{\text{train}}$.
Modern long-context stacks attack all three.

## 1. The cost equations

For a self-attention layer with $h$ heads, head dim $d$, and sequence
length $N$:

- **Attention compute (training, prefill):** $O(B h N^2 d)$ FLOPs.
  Dominates the transformer compute past $N \sim 2048$.
- **Attention KV memory (decode):** $2 N h d \cdot \text{bytes}$ per
  layer per batch element, times $L$ layers. For LLaMA-2-70B at $N=128$K,
  this is $\sim 50$ GB just for KV cache (at FP16, 8 GQA heads, 128 layers,
  per request).
- **Position OOD:** RoPE phases $m\theta_i$ at $m > N_{\text{train}}$ are
  unseen during training; attention features may not generalize. See
  `kb/notes/architecture/position-encoding.md`.

The `kb/notes/architecture/attention-mechanism.md` note covers the
exact-attention KV-sharing axis (MHA → MQA → GQA → MLA) which addresses
(ii) and the FlashAttention family which addresses (i)'s constant
factor. This note covers four other attack vectors.

## 2. Mechanism — four classes of long-context technique

### 2.1 Sparse attention (training-time and inference-time)

Replace dense attention with a structured-sparse mask, reducing FLOPs.
Three sub-families:

**Sliding-window attention (Longformer 2020, Mistral 2023):** each
position attends to the previous $w$ tokens. Compute drops to $O(N w d)$.
Quality is preserved if $w$ is chosen so that important long-range
dependencies fit in one window's "receptive field" through stacked
layers (effective receptive field grows linearly with depth).

**Block-sparse + global tokens (BigBird 2020):** small set of "global"
tokens attend everywhere, plus local windows, plus random sparse links.
Theoretical receptive-field arguments preserve approximation power.

**Native Sparse Attention (Yuan et al. 2025, arXiv 2502.11089):** the
2025 frontier. Three parallel attention branches per layer: a
**compressed** branch (block-summarizing keys/values via a learned MLP,
giving coarse global attention), a **selected** branch (top-$n$
importance-ranked KV blocks, fine-grained), and a **sliding-window**
branch (local context). Outputs are gated:

$$\boldsymbol{o}_t^* = \sum_{c \in \{\mathrm{cmp}, \mathrm{slc}, \mathrm{win}\}} g_t^c \cdot \mathrm{Attn}(\boldsymbol{q}_t, \tilde{K}_t^c, \tilde{V}_t^c)$$

`[yuan2025 §3.2 Eq.5; kb/excerpts/yuan2025#sec-3-2]`. The gates $g_t^c$
are an MLP+sigmoid on the input. The crucial design rationale:
**training with native sparsity** — the model learns to use the sparse
branches from pretraining, not retrofit at inference.

> "Applying sparsity post-hoc forces models to deviate from their
> pretrained optimization trajectory. As demonstrated by Chen et al.
> (2024b), top 20% attention can only cover 70% of the total attention
> scores, rendering structures like retrieval heads in pretrained
> models vulnerable to pruning during inference."
> `[yuan2025 §2.2; kb/excerpts/yuan2025#sec-2-2]`

NSA reports **11.6× decode**, **9.0× forward**, **6.0× backward**
speedup at 64K seq vs full attention, while matching/exceeding full-
attention quality on general / LongBench / reasoning evals
`[yuan2025 §1; kb/excerpts/yuan2025#sec-1]`.

The kernel-level innovation: where FlashAttention loads contiguous
*query blocks* into SRAM, NSA loads all *query heads of one GQA group at
one position* into SRAM, paired with that position's selected sparse
KV blocks. The GQA-group-shared selection
(`[yuan2025 §3.3.2 Eq.10; kb/excerpts/yuan2025#sec-3-3-2]`) is what
makes block-coalesced loading work despite sparsity.

**DeepSeek Sparse Attention (DSA, V3.2 Dec 2025)** is a follow-on built
on MLA: shared latent vectors across query heads enable block-level
sparse selection that is compatible with MLA's cache-absorbed forward.
[DSA PDF arXiv:2512.02556 not accessible in this pass — flagged as
open question.]

### 2.2 Linear / kernel attention (compute reduction by approximation)

Replace $\mathrm{softmax}(QK^\top)V$ with a kernelized form
$\phi(Q)(\phi(K)^\top V)$ where $\phi$ is a feature map. The
right-associative product is $O(N d^2)$ vs $O(N^2 d)$. Performer
(Choromanski 2020), Linear Transformer (Katharopoulos 2020) are early
examples. RWKV-7 (Mar 2025) and Mamba-2 SSD framing
(`kb/notes/architecture/state-space-models.md`) generalize this
direction. Pure-linear models lag dense attention on recall-heavy
benchmarks; **hybrid** designs (Jamba, Hymba, Zamba) interleave linear
and full attention layers.

### 2.3 Compressive memory — Infini-Attention (2024)

Munkhdalai et al. (Apr 2024, arXiv 2404.07143) proposes **Infini-
Attention**: each block contains both standard attention (over a finite
local window) and a compressive memory $M_s \in \mathbb{R}^{d \times d}$
that is updated linearly (via outer-product accumulation, like a linear
transformer) and read via cross-attention from local queries. The local
window stays bounded; the compressive memory accumulates a lossy
summary of all prior tokens. Reports 1M-token passkey retrieval.

[CONTRADICTION] Infini-Attention's reported quality is high but
reproductions outside Google have been limited; the loss in
information from a $d \times d$ compressive memory is not
characterized at frontier scale. As of mid-2026, no production-scale
LLM publicly uses Infini-Attention.

### 2.4 Distributed attention — Ring Attention (2023)

Ring Attention (Liu, Zaharia, Abbeel 2023, arXiv 2310.01889): partition
the sequence across $D$ devices, each holding $N/D$ contiguous tokens'
$Q, K, V$. Compute attention by **rotating $K, V$ blocks around the
ring** — each device's queries attend to its local KVs, then to the
next device's, etc., while overlapping communication with computation.
Total compute is unchanged ($N^2 d$); per-device memory is $O(N/D)$
for the activations and $O(N/D)$ for KV.

The win is that the maximum sequence length scales with $D$ (the device
count) rather than per-device memory. Combined with FlashAttention
(used as the per-tile kernel), Ring Attention enables **device-count
× longer-context** scaling. LLaMA 4's 10M-context inference and
Gemini's 1M-context training rely on this pattern (under various names:
context parallelism, sequence parallelism). [Ring Attention PDF
arXiv:2310.01889 not accessible in this pass — equations and reported
mechanism reflect well-established secondary references.]

**Context parallelism for inference** (Yang et al. 2024, arXiv 2411.01783)
reports 93% parallelization efficiency at 1M context, LLaMA 405B, on
128 H100s — production-grade demonstration of Ring-style distributed
attention for serving.

### 2.5 Position-encoding extension

Covered in `kb/notes/architecture/position-encoding.md`. Brief recap:

- **Position interpolation** (PI): scale positions $m \to m/s$.
- **NTK-aware scaling, YaRN**: scale RoPE base $b$, with per-frequency
  schedules. ~10× more token-efficient than PI for fine-tuning to long
  context.
- **LongRoPE2** (Feb 2025): evolutionary search over per-frequency
  scaling factors; near-lossless extension at 80× fewer tokens than
  Meta's LLaMA-3 long-context fine-tuning.
- **iRoPE** (LLaMA 4, Jan 2026): interleave RoPE attention layers with
  NoPE attention layers (3:1 ratio) + temperature scaling in NoPE
  layers. Trained natively at 256K, claimed 10M-token extrapolation.

## 3. Variants and lineage

| Method | Class | Compute | KV | Quality vs full | Year |
|---|---|---|---|---|---|
| Sliding-window | Sparse | $O(N w d)$ | $O(N)$ | strong locally | 2020+ |
| BigBird | Sparse + global | $O(N(w+g)d)$ | $O(N)$ | strong | 2020 |
| Performer | Linear | $O(N d^2)$ | $O(d^2)$ | weaker on recall | 2020 |
| Mamba / Mamba-2 | SSM | $O(N d^2)$ | $O(d^2)$ | strong on dense, weaker on copy | 2023+ |
| Infini-Attention | Compressive memory | $O(N w d)$ | $O(d^2 + w)$ | reported 1M passkey | 2024 |
| Ring Attention | Distributed exact | $O(N^2 d)$ but $\Theta(N/D)$ per device | $O(N/D)$ | exact (no approx) | 2023 |
| **NSA** | Native sparse, trained | $O(N (n + w + N/l) d)$ | $O(N)$ | matches/exceeds | 2025 |
| **iRoPE** | Position extension | (just RoPE swap) | unchanged | extrapolates 40× | 2026 |
| YaRN | Position extension | $O(1)$ | unchanged | preserved | 2023 |

The compute column for NSA is approximate: $n$ selected blocks + $w$
sliding window + $N/l$ compressed blocks (where $l$ is block length).
For 64K seq with $n = 16$, $w = 512$, $l = 64$: NSA computes attention
over $\sim$1500 effective KV positions per query, vs 64K — a $\sim$40×
sparsity ratio that translates to the reported 9× forward speedup
(constant-factor overhead for three branches, MLP compression, gate).

## 4. Intuitions and analogies

[ANALOGY] Long context is a **memory hierarchy problem**, not a single
algorithmic choice. The full prefix is too large for fast access; the
question is which compressed/sampled views to keep at which speed
tier:

- **Sparse attention** = "keep only important keys, blockwise."
- **Linear attention / SSM** = "keep a fixed-size summary state."
- **Infini-Attention** = "small-window full-fidelity + larger compressed
  state."
- **Ring attention** = "shard the prefix across devices, exact attention."
- **Position extension** = "make sure the addressing function still
  works at long range."

The four are largely orthogonal — production stacks combine them.
LLaMA 4 uses iRoPE + (some sparse?) + early-fusion + Ring-style context
parallelism for the 10M-token inference. DeepSeek-V3.2 uses MLA + DSA
(sparse on top of MLA's compressed cache). The analogy returns to
canonical form via the compute-and-memory equations of §1.

[INTUITION] **Why pretraining at long context is hard.** The
constituting tokens of a long-range dependency need to co-occur in the
training data, which is rare beyond a few thousand tokens (most natural
documents are short). One mitigation: synthetic long-context data
(concatenated documents, "needle in a haystack" style training). NSA's
"native" emphasis means the sparse pattern is itself learned during
pretraining, not retrofit — but the model still needs *something* to
attend to at long range during pretraining for the sparse branches to
develop useful behavior
`[yuan2025 §2.2; kb/excerpts/yuan2025#sec-2-2]`.

[INTUITION] **The recall-vs-compute tradeoff.** Linear attention and
SSMs lose information per token (their state is fixed-size, not
$N$-linear). Empirically this manifests as weak performance on
**copy/needle-in-haystack** tasks where the model must retrieve a
specific token from far back. Sparse attention preserves selected
tokens exactly, so it doesn't suffer the fixed-state bottleneck.
Hybrids (`kb/notes/architecture/state-space-models.md`) are an attempt
to keep linear-attention's compute scaling for "background" while
keeping a few full-attention layers for needle-recall.

[CONTRADICTION] **What "1M-token context" actually means.** The
needle-in-a-haystack benchmark family, where the model must retrieve a
short fact buried in long noise, produces near-100% accuracy claims
from many vendors at 1M+ context. The **harder** evaluations — RULER,
HELMET — show steeper degradation past 32K. Practical long-context
performance varies by task type (retrieval vs reasoning vs aggregation)
in ways that simple "context length" numbers obscure. Belongs in
`kb/notes/evaluation/eval-methodology.md`.

## 5. Frontier and open questions (as of 2026-05)

- **NSA generalization beyond DeepSeek's training stack.** NSA is the
  highest-quality public claim for trainable sparse attention. Whether
  the three-branch + GQA-group-aligned selection works as well outside
  DeepSeek-AI's training recipe is open.
- **iRoPE theoretical justification.** Why does interleaving NoPE
  layers help extrapolation? The "Rope to Nope and Back Again" paper
  (arXiv 2501.18795) gives empirical evidence; a theoretical account of
  the 3:1 ratio and the temperature-scale schedule is missing.
- **DSA × MLA interaction.** DeepSeek-V3.2 (Dec 2025) layers DSA on
  MLA. The compatibility — sparse selection over a low-rank latent
  cache — is non-obvious and the paper's mechanics need a Phase 2 deep
  dive.
- **Ring attention for inference at scale.** Context parallelism is
  well-established in training; inference-time deployment patterns
  (KV-cache sharding, multi-request batching with different context
  lengths on the same ring) are still maturing. Belongs in
  `kb/notes/inference/serving-systems.md`.
- **Compressive memory at frontier scale.** Infini-Attention has not
  seen frontier replication. Whether a $d \times d$ compressive state
  is sufficient for trillion-parameter-scale long-context recall is
  open.
- **Long-context evaluation rigor.** [CONTRADICTION] As of 2026-05,
  vendor-reported context lengths are not directly comparable. NIAH
  saturates; RULER and HELMET show meaningful degradation. The "real"
  frontier — useful reasoning over 1M+ tokens — is still empirically
  poorly characterized.

## 6. See also

- `kb/notes/architecture/attention-mechanism.md` — exact-attention
  algorithms (FlashAttention, MLA) that long-context techniques compose
  with.
- `kb/notes/architecture/position-encoding.md` — RoPE / YaRN / iRoPE,
  the position-axis of long-context engineering.
- `kb/notes/architecture/state-space-models.md` — linear-attention and
  SSM-hybrid alternatives.
- `kb/notes/inference/kv-cache-management.md` — paging, eviction,
  quantization of the KV cache; orthogonal compression axis.
- `kb/notes/evaluation/eval-methodology.md` — long-context benchmarks
  (NIAH, RULER, HELMET) and what they actually measure.
