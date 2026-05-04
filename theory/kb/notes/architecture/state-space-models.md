---
topic: architecture/state-space-models
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - gu2023-mamba
  - mamba2
  - lieber2024-jamba
secondary_sources:
  - rwkv7-2025
  - vaswani2017  # contrast point
related_topics:
  - architecture/attention-mechanism
  - architecture/long-context
  - architecture/transformer-overview
---

# State-space models (SSMs) and SSM/Transformer hybrids

A state-space model processes a sequence with a **recurrent state**
$\boldsymbol{h}_t$ rather than attention over a growing prefix. Per-token
compute is $O(d^2)$ (constant in sequence length); memory is $O(d^2)$
for the state. This makes SSMs structurally cheaper than attention at
long sequences but at the cost of fixed-state information bottleneck.
The frontier as of 2026 is **hybrids**: serial or parallel
interleavings of SSM blocks and attention blocks.

The Mamba/Mamba-2/Jamba PDFs were not accessible during this Phase 2
pass; equations and reported numbers below reflect well-established
forms from secondary references. Verify before propagating to LaTeX.

## 1. Formal definition — the linear state-space recurrence

A continuous-time linear state-space model of dimension $d$ on input
$u(t) \in \mathbb{R}$:

$$\frac{d \boldsymbol{h}(t)}{d t} = A \boldsymbol{h}(t) + B u(t), \quad y(t) = C \boldsymbol{h}(t) + D u(t) \tag{1}$$

with $A \in \mathbb{R}^{d \times d}$, $B \in \mathbb{R}^{d \times 1}$,
$C \in \mathbb{R}^{1 \times d}$, $D \in \mathbb{R}$. Discretizing with
step size $\Delta$ via zero-order hold gives the discrete recurrence:

$$\boldsymbol{h}_t = \bar{A} \boldsymbol{h}_{t-1} + \bar{B} u_t, \quad y_t = C \boldsymbol{h}_t + D u_t \tag{2}$$

with $\bar{A} = \exp(\Delta A)$, $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$
[Gu et al. 2021 S4; classical control-theory derivation].

For a sequence input $u_{1:N}$, the recurrence in (2) computes
$y_{1:N}$ in $O(N d^2)$ — linear in $N$, vs $O(N^2 d)$ for attention.

### 1.1 S4 — structured state-space models (Gu, Goel, Ré 2022)

S4 chooses $A$ to have **HiPPO** structure: a specific dense matrix that
"optimally compresses" the input history under a polynomial basis
(orthogonal Legendre polynomials). The resulting recurrence has a
**convolutional dual**: for fixed $\bar{A}, \bar{B}, C$, the output is
$y = \bar{K} * u$ with kernel

$$\bar{K} = (C \bar{B}, C \bar{A} \bar{B}, C \bar{A}^2 \bar{B}, \ldots, C \bar{A}^{N-1} \bar{B}) \tag{3}$$

S4's key trick: compute $\bar{K}$ efficiently via FFT in $O(N \log N)$
(rather than $O(N d^2)$ recurrence) during training, but use the
recurrent form during autoregressive inference for $O(d^2)$ per step.
This gives **dual-form** SSMs: parallelizable like a CNN at training
time, streamable like an RNN at inference time.

### 1.2 Mamba — selective state spaces (Gu & Dao 2023, arXiv 2312.00752)

Mamba's innovation: make $\bar{A}, \bar{B}, C$ **input-dependent**.
Specifically, $\bar{B}_t = s_B(u_t)$, $C_t = s_C(u_t)$, and $\Delta_t =
s_\Delta(u_t)$ are produced by linear projections of the current input;
$A$ remains a learned diagonal matrix. The discretized recurrence with
input-dependent parameters becomes:

$$\boldsymbol{h}_t = \bar{A}_t \boldsymbol{h}_{t-1} + \bar{B}_t u_t \tag{4}$$

The **selection** mechanism: $\Delta_t$ controls how much of the new
input is incorporated vs. how much state is retained, per token. This
breaks the convolutional dual (because $\bar{A}_t$ now varies per
position), but Mamba implements a **parallel scan** in $O(N \log N)$
to recover training-time parallelism on GPU.

The Mamba block: applies the selective SSM to a per-channel
1-dimensional input (each of $d$ "channels" has its own SSM with shared
$A$ but per-channel $B, C, \Delta$). The block also includes a gated
nonlinearity à la GLU and a residual connection.

[The Mamba PDF was not accessible in this pass. Equation forms above
are the canonical forms from textbook treatments (Gu's PhD thesis;
Goombalab blog https://goombalab.github.io/blog/2024/mamba2-part1-model/).
Verify Mamba-1 specifics before formal output.]

### 1.3 Mamba-2 — Structured State Space Duality (Dao & Gu 2024, arXiv 2405.21060)

Mamba-2's theoretical contribution: **State Space Duality (SSD)**, a
framework showing that selective SSMs and a restricted form of
attention compute the same function. Specifically, Mamba-2 shows that:

$$\mathrm{Mamba\text{-}2}(\boldsymbol{u}) = M \boldsymbol{u}$$

where $M$ is a **lower-triangular semi-separable matrix** of rank $\le d$
(the SSM state dimension). This semi-separable structure both (i)
recovers efficient $O(N)$-flop training via structured matmul, and (ii)
gives a quadratic-form expression that runs FlashAttention-style on
modern hardware (achieving 2–8× speedup over Mamba-1).

The SSD framing makes Mamba-2 an attention variant with a structurally
restricted attention matrix — the "transformers are SSMs" subtitle
literalizes this.

[Mamba-2 PDF not accessible. SSD framing reflects Dao's tridao.me blog
post and the Goombalab series. Verify exact theorem statements.]

### 1.4 RWKV-7 (Peng et al. 2025, arXiv 2503.14456)

RWKV is a parallel SSM/attention-hybrid lineage. RWKV-7 "Goose"
introduces a **generalized delta rule** with vector-valued gating and
claims to recognize all regular languages and perform state tracking —
a complexity-class claim placing RWKV-7 above pure-Transformer
expressiveness under standard conjectures (TC⁰).

[CONTRADICTION] RWKV-7's expressiveness claims are strong and as of
mid-2026 not yet independently replicated on formal-language benchmarks.
The empirical LM quality at 7B+ is competitive with similar-size
Transformers; whether the theoretical-expressiveness claim translates
to practical advantage is open.

## 2. Mechanism — why SSMs are structurally cheaper but quality-tradeoff'd

### 2.1 Compute and memory

For a sequence of length $N$ with state dimension $d$:

| Operation | Attention (vanilla) | SSM (Mamba) |
|---|---|---|
| Train forward FLOPs | $O(N^2 d)$ | $O(N d^2)$ |
| Decode FLOPs / token | $O(N d)$ | $O(d^2)$ |
| State / KV memory | $O(N d)$ growing | $O(d^2)$ fixed |
| Recall | exact (each prior token addressable) | lossy (state is summary) |

The crossover point where SSM's $O(N d^2)$ training cost beats
attention's $O(N^2 d)$ is at $N \sim d$. For Mamba-1's $d \approx 16$
(state dim) and typical $N \in [4096, 32768]$, the SSM is orders of
magnitude cheaper. For decode the gap is even larger: SSMs have
**constant per-token cost**.

### 2.2 The recall bottleneck

A pure SSM compresses the entire prefix into a state of fixed size
$d^2$ (or $d$ for diagonal $A$). Tasks that require **exact retrieval**
of a specific prior token — copying, hashing, needle-in-haystack —
expose this bottleneck: the relevant token's information may have been
overwritten in the state by intervening updates. Empirically, pure
Mamba models are weaker than Transformers on associative-recall tasks
at matched compute [INTUITION; no specific paper citation in this pass —
this is well-established from the Mamba-2 / Jamba papers' ablations
which are not yet excerpted].

This is the first-principles argument for **hybrid** architectures.

## 3. Variants and lineage — the hybrid families

### 3.1 Pure SSM lineage

- **S4** (Gu, Goel, Ré 2022) — HiPPO-structured $A$, FFT-based training
  convolution, recurrence-based inference.
- **S5** (Smith, Warrington, Linderman 2023) — diagonal-plus-low-rank
  $A$, simpler than S4's HiPPO-NPLR.
- **Mamba** (Gu & Dao 2023) — selective (input-dependent) SSM; parallel
  scan; per-channel SSMs.
- **Mamba-2** (Dao & Gu 2024) — SSD: rank-restricted attention dual.
- **RWKV-4–7** (Peng et al.) — RNN-style with attention-derived
  parallelism.

### 3.2 Hybrid lineage

| Model | Backbone | Attention pattern | Year |
|---|---|---|---|
| **Jamba** (AI21 2024) | serial: 1 attn block per 7 Mamba blocks | full-attention, GQA | 2024 |
| **Jamba-1.5** (AI21 Aug 2024) | same pattern, scaled to 94B active | full-attention | 2024 |
| **Zamba** (Glorioso et al. 2024) | Mamba-1 backbone + 1 shared attention block | shared attn applied periodically | 2024 |
| **Zamba2** (Glorioso et al. Nov 2024) | Mamba-2 backbone + 2 shared attn blocks | 6× KV cache reduction vs Zamba1 | 2024 |
| **Hymba** (NVIDIA Nov 2024) | parallel: SSM + attention heads in same block | attention and SSM heads side-by-side | 2024 |
| **Falcon-Mamba** (TII 2024) | pure Mamba-1 at 7B scale | none | 2024 |

The **interleaving ratio** is the key design knob. AI21 reports
attention layers at 1:7 vs Mamba layers; this is enough attention to
cover recall-style needs while keeping the cheap SSM substrate
dominant. Hymba's parallel design avoids the serial attn-bottleneck:
each block has both attention heads and SSM heads operating on the
same input in parallel.

Pure-Mamba architectures **at frontier scale (>30B) have not
materialized as of mid-2026**. The deployed long-context architectures
are all attention-based (LLaMA 4, Qwen3, DeepSeek-V3) or
SSM-attention hybrids (Jamba-1.5).

### 3.3 Why hybrids and not pure SSM

[INTUITION] Hybrids capture two distinct mechanism types:

- The SSM layers act as a **compressed-history channel** — relatively
  cheap, content-summarizing, weak at exact recall.
- The attention layers act as **exact-retrieval gates** — recall any
  prior token by content match, expensive.

Stacking them gives the model both modalities. The 1:7 ratio (Jamba)
suggests attention is rare-but-essential: the model needs occasional
exact-recall steps but most computation is routine. [SPECULATION] The
optimal ratio likely depends on the task distribution: code (lots of
exact-name lookup) probably benefits from more attention, whereas
narrative summarization (lots of compression) benefits from more SSM.

### 3.4 SSD as a unifying lens

Mamba-2's SSD framing (§1.3) is theoretically attractive: it suggests
that the practical distinction between attention and SSM is not
"recurrent vs parallel" but rather "what structural restriction on the
attention matrix." Vanilla attention is unrestricted; sliding-window
attention is band-banded; Mamba-2 is rank-restricted (semi-separable).
This puts the design space on a continuum rather than a binary.
[INTUITION] Whether this lens leads to *new* useful constraints (e.g.,
"mid-rank semi-separable plus full-attention residual") or just
re-explains existing methods is an active research question.

## 4. Intuitions and analogies

[ANALOGY] An SSM is a **compression buffer that updates per token**.
The state $\boldsymbol{h}_t$ is a fixed-size summary of the prefix
$u_{1:t}$; each new $u_t$ rewrites part of it. By contrast, attention
**keeps every prior token in addressable form** (the KV cache). The
fixed-size buffer is cheap; the addressable cache is expensive but
exact. The analogy returns to canonical form via the cost equations of
§2.1.

[ANALOGY] **Mamba-2 SSD as "attention with a bandwidth budget."** The
attention matrix $\mathrm{softmax}(QK^\top)$ in vanilla Transformers is
a dense $N \times N$ matrix with no structural constraint. Mamba-2's
matrix $M$ is constrained to be lower-triangular and semi-separable
with rank $\le d$ — the model can only express interactions with
"bandwidth" $d$. This is exactly the recurrent state's information
bottleneck written in attention notation. The analogy returns to the
SSD theorem's matrix decomposition.

[INTUITION] **Why the 1:7 hybrid ratio is roughly right.** [SPECULATION]
A vanilla Transformer's per-token attention computation costs roughly
$N \cdot d_h$ operations vs Mamba's $d^2$ (state-update). At $N = 8K$,
$d_h = 128$: attention is $\sim 10^6$ ops/token; Mamba (state $d \sim
16$) is $\sim 256$ ops/token. The ratio is $\sim 4000\times$. Replacing
1 attention layer with 1 Mamba layer at fixed total layer count saves
on average that ratio in compute on that layer. So 1:7 hybrids reduce
attention-layer compute to $\tfrac{1}{8}$ of an all-attention model,
giving most of the gain while preserving recall. The right ratio
depends on $N$; at very long $N$, SSM's advantage grows and the
optimal hybrid pushes toward more SSM.

## 5. Frontier and open questions (as of 2026-05)

- **Pure SSM at 30B+.** No public model. Whether the recall bottleneck
  scales away with parameter count, or remains fundamental, is open.
- **RWKV-7 expressiveness claims.** Claims state-tracking and
  regular-language recognition beyond Transformer capability under
  TC⁰. Independent verification on formal-language benchmarks
  (Dyck-$n$, parity, etc.) at scale is needed.
- **Optimal hybrid ratio and pattern.** Jamba's 1:7 vs Zamba's
  shared-attention-block vs Hymba's parallel design — no controlled
  comparison at matched params/compute exists publicly.
- **SSD generalizations.** Are there useful SSD-like decompositions
  beyond rank-restricted (semi-separable)? Block-sparse,
  banded-plus-low-rank, etc. — open theoretical territory.
- **Mamba-3 and DeepSeek-style hybrid.** Rumored / forthcoming. As of
  2026-05 no public release.
- **SSM × MoE.** Most hybrid papers are dense; the interaction with
  MoE routing is largely unexplored. [SPECULATION] An SSM-routed MoE
  could in principle pick the SSM "summary" view of the prefix to
  dispatch to an expert.

## 6. See also

- `kb/notes/architecture/attention-mechanism.md` — the dense-attention
  baseline; FlashAttention as the backbone the hybrids' attention
  layers use.
- `kb/notes/architecture/long-context.md` — SSMs are one of four
  long-context attack vectors; the recall-vs-compute tradeoff is
  contextualized there.
- `kb/notes/architecture/transformer-overview.md` — block-diagram
  comparisons across SSM, hybrid, and attention-only architectures.
- `kb/notes/architecture/normalization.md` — Mamba and Jamba use RMSNorm
  with various pre/post placements; distinct considerations.
