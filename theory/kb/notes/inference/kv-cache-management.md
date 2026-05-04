---
topic: inference/kv-cache-management
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - kwon2023            # PagedAttention / vLLM
  - shazeer2019         # MQA — quantifies why KV bandwidth dominates
  - ainslie2023         # GQA
  - deepseek-v2         # MLA — latent KV
  - zheng2024-sglang    # RadixAttention / cross-request prefix reuse
secondary_sources:
  - kvquant2024         # per-channel + nuq2 quantization for KV
related_topics:
  - architecture/attention-mechanism
  - inference/serving-systems
  - inference/quantization
---

# KV cache management

The autoregressive decoding loop reads the K/V projections of every
prior token at every layer at every step. The size, layout, and reuse
of this state — the **KV cache** — is the central memory-bandwidth
problem in modern LLM inference. The interventions stack along three
mostly-orthogonal axes: (1) **architectural compression** (MQA / GQA /
MLA — done at training time, treated upstream in
`kb/notes/architecture/attention-mechanism.md`), (2) **layout and
allocation** (paging, continuous batching, copy-on-write), and (3)
**numerical compression** (quantization to 4 / 2 bits, layer-wise
mixed precision). This note focuses on (2) and (3); (1) is referenced
where it is upstream of a layout decision.

## 1. The size and structure of the cache

Per layer, per head, per token, the cache stores one $K_t \in
\mathbb{R}^{d_h}$ and one $V_t \in \mathbb{R}^{d_h}$. For a sequence of
length $n$ on a model with $L$ layers and $n_h$ heads of dimension
$d_h$, the cache is

$$|\text{KV}| = 2 \cdot L \cdot n_h \cdot d_h \cdot n \cdot b \quad \text{bytes}, \tag{1}$$

with $b$ bytes per element ($b=2$ for FP16/BF16, $b=1$ for FP8 etc.).
Kwon et al. give the OPT-13B example: $2 \cdot 5120 \cdot 40 \cdot 2 =
800$ KB per token, so a single 2048-token request demands $\approx
1.6$ GB of KV cache `[kwon2023 §3; kb/excerpts/kwon2023#sec-3]`.

For decoding on modern accelerators, the bottleneck is *not* the FLOPs
that go into producing $K_t, V_t$; it is the bandwidth of *re-reading*
the entire cached prefix at every step. Shazeer's analysis gives the
governing memory/compute ratio in incremental decoding as $\Theta(n/d
+ 1/b_{\text{batch}})$ for sequence length $n$, model width $d$, and
batch size $b_{\text{batch}}$
`[shazeer2019 §2.4; kb/excerpts/shazeer2019#sec-2-4]`. When $n
\approx d$ or $b_{\text{batch}} \approx 1$, the ratio approaches 1
and the operation is HBM-bandwidth-bound. This is the *reason* every
intervention below exists.

## 2. Architectural compression of K/V — recap

Treated in detail in `kb/notes/architecture/attention-mechanism.md §3.1`.
Headline shape per token per layer:

| Variant | Cache per token / layer | Source |
|---|---|---|
| MHA (Vaswani 2017) | $2 n_h d_h$ | `[vaswani2017]` |
| MQA (Shazeer 2019) | $2 d_h$ | `[shazeer2019 §3; kb/excerpts/shazeer2019#mqa-structure]` |
| GQA-$g$ (Ainslie 2023) | $2 g d_h$ | `[ainslie2023 §2.2; kb/excerpts/ainslie2023#sec-2-2]` |
| MLA (DeepSeek-V2 2024) | $(d_c + d_h^R) \approx \tfrac{9}{2} d_h$ | `[deepseek-v2 §2.1.4]` |

These are the *training-time* options. They cannot be retrofitted
freely (TransMLA `[transmla2025]` claims 0.3–0.6% data is enough to
convert GQA → MLA, which is a 2025 result; the canonical assumption
remains that MQA/GQA/MLA are chosen at pre-training and inherited by
the inference stack).

## 3. Memory layout — PagedAttention / vLLM

### 3.1 The waste taxonomy

Kwon et al. identify three sources of memory waste in the
contiguous-allocation regime that pre-2023 serving stacks (Orca,
FasterTransformer) used `[kwon2023 §3; kb/excerpts/kwon2023#sec-3]`:

1. **Reservation** — slots reserved for *future* generated tokens are
   unused right now but cannot be loaned out. Pure carrying cost.
2. **Internal fragmentation** — the slack between a request's actual
   final length and the maximum length for which space was
   pre-allocated. Realized as waste only after the request finishes.
3. **External fragmentation** — buddy-allocator-style holes between
   pre-allocated chunks; never used for any generation.

Empirical impact: in their measurements, only 20.4–38.2% of allocated
KV cache was actual token state in the Orca variants; vLLM brings
this to 96.3% `[kwon2023 §3 / Fig.2; kb/excerpts/kwon2023#sec-3]`.

### 3.2 The PagedAttention algorithm

The KV cache of each sequence is partitioned into **KV blocks** of
fixed size $B$ tokens (typical $B=16$). With $K_j = (k_{(j-1)B+1},
\ldots, k_{jB})$ and $V_j = (v_{(j-1)B+1}, \ldots, v_{jB})$, the
attention output at query position $i$ becomes

$$A_{ij} = \frac{\exp(q_i^\top K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil} \exp(q_i^\top K_t \mathbf{1} / \sqrt{d})}, \quad o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^\top, \tag{2}$$

with the kernel reading each $K_j, V_j$ block from a possibly
non-contiguous physical address `[kwon2023 §4.1 Eq.4;
kb/excerpts/kwon2023#sec-4-1]`. The math is exactly Eq.(1) of attention,
just block-decomposed; the contribution is layout, not arithmetic.

The serving system maintains **block tables** mapping each request's
*logical KV blocks* to *physical KV blocks*, exactly analogous to OS
virtual memory page tables `[kwon2023 §4.2; kb/excerpts/kwon2023#sec-4-2]`.
Three operating-system idioms transfer one-to-one:

- **Allocate-on-demand:** physical blocks are claimed only as new
  tokens are generated; no maximum-length pre-reservation.
- **Copy-on-write** for shared prefixes: when two sequences share a
  prompt block and one is about to write into it, the block is copied
  and its reference count decremented `[kwon2023 §4.4;
  kb/excerpts/kwon2023#sec-4-4]`. Same pattern as `fork()` in Unix.
- **Swap and recompute** for preemption: evicted blocks can be moved
  to CPU RAM (swap) or simply recomputed from the prompt + already-
  generated tokens in one prefill iteration `[kwon2023 §4.5;
  kb/excerpts/kwon2023#sec-4-5]`. The swap-space size is bounded by
  the GPU KV space.

### 3.3 Why this matters for sampling-heavy workloads

The big win of paging is in workloads that share prefixes:

- **Parallel sampling** ($n$ samples from one prompt): the prompt's KV
  cache is stored once, with reference count $n$. Generation diverges
  per sample with copy-on-write at the block boundary
  `[kwon2023 §4.4; kb/excerpts/kwon2023#sec-4-4]`.
- **Beam search** of width $k$: most blocks are shared across beams;
  only the latest divergence creates a new block per beam.
- **System prompts:** "predefined shared prefixes" can be cached
  once across all requests
  `[kwon2023 §4.4; kb/excerpts/kwon2023#sec-4-4]`.

Reported headline: 2–4× throughput vs. FasterTransformer and Orca on
OPT-13B/66B/175B at matched latency
`[kwon2023 abstract / §6.1; kb/excerpts/kwon2023#abstract]`.

## 4. Cross-request reuse — RadixAttention

PagedAttention's sharing is *intra-request* and within deliberately-
declared shared prefixes. SGLang's **RadixAttention** generalizes this
to *fully automatic cross-request prefix reuse* `[zheng2024-sglang §3;
kb/excerpts/zheng2024-sglang#sec-3]`:

- All prompts and their generations are inserted into a **radix tree**
  whose edges are token sequences and whose nodes hold pointers to KV
  cache pages.
- A new request hits the longest matching prefix; its KV cache is
  reused up to the divergence point.
- Eviction is **LRU on leaves** with reference counting on the
  in-flight batch. A node is evictable iff its reference count is zero
  and no child is in-flight `[zheng2024-sglang §3;
  kb/excerpts/zheng2024-sglang#sec-3]`.
- A **cache-aware scheduler** orders the waiting queue by
  longest-shared-prefix-first; Zheng et al.'s Theorem 3.1 shows this is
  equivalent to depth-first traversal of the radix tree, which is
  cache-hit-rate-optimal `[zheng2024-sglang §3 Theorem 3.1;
  kb/excerpts/zheng2024-sglang#sec-3-cache-aware]`.

RadixAttention is *compatible* with PagedAttention; in deployment,
SGLang's pages are typically token-sized (page = 1 token) which makes
prefix matching exact `[zheng2024-sglang §3;
kb/excerpts/zheng2024-sglang#sec-3]`. vLLM v0.4+ shipped its own
"automatic prefix caching" feature that instantiates the same idea on
PagedAttention pages.

[INTUITION] RadixAttention is "memoization for the prefill phase."
Given a long system prompt + few-shot exemplars + new task input, the
*prefill* of the first two parts is identical across requests; the
radix tree turns this from $O(\text{shared\_tokens})$ recomputation per
request into $O(1)$ pointer lookup. The asymptotic-cost picture is the
same as ordinary prefix-tree memoization; the engineering challenge is
LRU eviction under reference counts and continuous batching.

## 5. Numerical compression — KV-cache quantization

Storage per element is the other axis. Going from FP16 to INT4 cuts
the cache by 4× without changing layout. The tension is that quality
degradation appears asymmetrically across keys vs. values and across
layers.

### 5.1 KVQuant — per-channel keys, per-token values, plus nuq

KVQuant `[kvquant2024]` (Hooper et al., NeurIPS 2024) reports that
the key cache and the value cache need *different* quantization
geometries:

- **Per-channel quantization for keys**: outliers are concentrated in
  specific channels (analogous to activation outliers in SmoothQuant
  `[xiao2023-smoothquant §3; kb/excerpts/xiao2023-smoothquant#sec-3]`),
  so per-channel scales preserve the outlier resolution.
- **Per-token quantization for values**: outliers in values are
  spread per-token rather than per-channel.
- **nuq2** — a non-uniform 2-bit quantizer fit to the value
  distribution.

Reported headline: 8× compression of the KV cache enabling 1M-token
contexts on a single A100. [The full quantitative claim and ablation
need cross-checking against the KVQuant paper itself; this note's
secondary citation reflects that KVQuant is referenced via the Phase 1
sweep, not directly via a verbatim excerpt.]

### 5.2 Layer-wise mixed precision

KVTuner `[2502.04420]` and similar 2025 work argue that not all
layers' KV caches are equally sensitivity-bound, so layer-wise mixed
precision (e.g. INT4 in early layers, INT8 in late layers) Pareto-
dominates uniform quantization. [SPECULATION about which specific
schedule wins; KVTuner is cited from Phase 1 as a representative of
the technique class — leaf-level treatment requires the paper's
verbatim ablation, which is not yet in this KB.]

## 6. The MLA-vs-quantization interaction (open question)

[CONTRADICTION] Multi-Head Latent Attention `[deepseek-v2 §2.1.2;
kb/excerpts/deepseek-v2#sec-2-1-2]` stores a *latent* $\mathbf{c}_t^{KV}
\in \mathbb{R}^{d_c}$ rather than per-head K/V. The latent is dense
across all heads, so per-head outlier patterns that motivate KVQuant's
per-channel-key scheme do not obviously apply. As of 2026-05, the
literature treats KV-quantization and MLA as independent options; a
principled integration (e.g. quantization-aware MLA training, or
post-hoc latent quantization) is not in primary published form. This
is flagged in Phase 1's open-questions list.

## 7. Frontier and open questions (as of 2026-05)

- **Cross-request KV sharing in production.** Both vLLM (automatic
  prefix caching) and SGLang (RadixAttention) ship this. Reported hit
  rates of 85–95% on agentic workloads `[FORUM-SIGNAL: vLLM blog
  posts]` need primary-source numbers in the KB. The 15–25% on vLLM's
  prior versions is the contrast point.
- **Prefill / decode disaggregation.** A 2025 architectural shift:
  separate "prefill workers" and "decode workers," moving KV across the
  cluster via NVLink/InfiniBand. Reduces tail latency by removing the
  prefill-vs-decode head-of-line blocking. vLLM has experimental
  support; impact on aggregate throughput is contested
  `[FORUM-SIGNAL: vllm.ai blog]`. Treated in
  `kb/notes/inference/serving-systems.md`.
- **KV sharing across requests with privacy boundaries.** RadixAttention
  is correctness-safe (the cache is just a memoization), but in
  multi-tenant settings the question of whether prefix-cache hits leak
  *information* about other tenants (timing channels) is not addressed
  in the SGLang paper.
- **Long-context KV cost vs. window-attention / NSA.** As contexts
  push 1M+ tokens, even paged FP16 KV is memory-prohibitive on a
  single node. The architectural alternatives (sliding-window
  attention, Native Sparse Attention `[yuan2025]`, MLA latents) are
  the more aggressive lever; quantization is the cheaper retrofit.
  Treated in `kb/notes/architecture/long-context.md`.
- **Recompute vs. swap.** Kwon et al.'s §7.3 reports that for OPT
  sizes the breakeven depends on whether the prefix is short
  (recompute wins) or long (swap wins). The recompute regime gets
  more attractive as FlashAttention-3 / FP8 prefill speeds rise.

## 8. See also

- `kb/notes/architecture/attention-mechanism.md §3.1` — the MQA→GQA→MLA
  KV-shape evolution. Upstream of every layout decision.
- `kb/notes/inference/serving-systems.md` — vLLM, SGLang, TensorRT-LLM
  as full systems. PagedAttention is one component of vLLM; this note
  treats it as the algorithm.
- `kb/notes/inference/quantization.md` — weight and activation
  quantization. KV-cache quantization shares the outlier-channel
  intuition with SmoothQuant / AWQ but applies it to a different
  tensor.
- `kb/notes/inference/speculative-decoding.md` — speculative methods
  amplify the KV-bandwidth problem (each draft step still touches the
  cache) but verification is parallel; vLLM and SGLang both interact
  with speculative decoding via the block layout.
