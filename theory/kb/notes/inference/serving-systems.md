---
topic: inference/serving-systems
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - kwon2023            # vLLM / PagedAttention (SOSP 2023)
  - zheng2024-sglang    # SGLang / RadixAttention (NeurIPS 2024)
secondary_sources:
  - dao2022             # FlashAttention as the kernel substrate
  - dao2023             # FlashAttention-2
  - shah2024            # FlashAttention-3
  - dong2024-xgrammar   # structured output as serving feature
related_topics:
  - inference/kv-cache-management
  - inference/speculative-decoding
  - inference/structured-output
  - architecture/attention-mechanism
---

# Serving systems

A "serving system" is the orchestration layer between requests and the
GPU. It owns: request scheduling, KV cache memory allocation, batch
formation, kernel dispatch, and (increasingly) constrained-decoding
plus speculative drafts. By 2026, three open systems dominate
production: **vLLM** (Berkeley, generic-purpose), **SGLang**
(LMSYS/Berkeley, programmatic LM workflows), and **TensorRT-LLM**
(NVIDIA, latency-optimized, vendor-locked). On the closed-source side,
the production stacks at OpenAI / Anthropic / Google are not publicly
described in detail.

This note treats vLLM and SGLang as primary case studies — both have
peer-reviewed publications. TensorRT-LLM is referenced via vendor docs
and is treated as Tier-B; llama.cpp is Tier-B/C (no canonical paper).

## 1. The serving stack — anatomy

A modern LLM serving system is a layered loop:

1. **Request intake.** Tokenize input; assign a sequence ID; place in
   the waiting queue.
2. **Scheduler.** Decide which sequences to advance this iteration.
   Modern stacks use **iteration-level (continuous) batching** — a
   batch is reformed every step rather than every request — so the
   batch shape changes per iteration `[kwon2023 §2.3;
   kb/excerpts/kwon2023#sec-2]`.
3. **KV cache manager.** Allocate / map / evict KV blocks. PagedAttention
   `[kwon2023 §4]` for vLLM; RadixAttention `[zheng2024-sglang §3]` for
   SGLang. See `kb/notes/inference/kv-cache-management.md`.
4. **Kernel dispatch.** For each layer: attention kernel
   (FlashAttention-2/3, PagedAttention's block-aware variant), GEMM
   for projections, normalization, FFN. Substrate is FlashAttention
   `[dao2022; dao2023; shah2024;
   kb/excerpts/dao2022#sec-1-headline]`.
5. **Sampling / verification.** Sampling for normal decode; rejection
   sampling for speculative; FSM-constrained for structured output.
6. **Output streaming.** Token IDs back to the client over WebSocket
   / SSE.

The architectural debate over the past three years has been: *which of
these layers should be co-designed?* PagedAttention co-designs
attention kernel + memory manager. RadixAttention adds frontend +
memory manager. SGLang's compressed-FSM co-designs constraint
specification + kernel. The pattern: cross-layer co-design wins,
within-layer optimization plateaus.

## 2. Continuous batching — the universal substrate

**Continuous batching** (also "iteration-level scheduling," due to Yu
et al., Orca, OSDI 2022) is the precondition for everything else. The
key idea `[kwon2023 §2.3; kb/excerpts/kwon2023#sec-2]`:

> Fine-grained batching mechanisms, such as cellular batching and
> iteration-level scheduling, have been proposed. Unlike traditional
> methods that work at the request level, these techniques operate at
> the iteration level. After each iteration, completed requests are
> removed from the batch, and new ones are added. Therefore, a new
> request can be processed after waiting for a single iteration, not
> waiting for the entire batch to complete. Moreover, with special
> GPU kernels, these techniques eliminate the need to pad the inputs
> and outputs.

Without continuous batching, a single long sequence stalls the entire
batch until completion (head-of-line blocking). With it, completed
sequences free their slots immediately and pending requests join.
This is the SOSP'23 / OSDI'22 era's core insight; vLLM, SGLang,
TensorRT-LLM, llama.cpp all adopted it.

[INTUITION] Continuous batching turns the batch from "a frozen rectangle
of tokens" into "a streaming pool of in-flight sequences," exactly
analogous to the OS shifting from batch processing to time-shared
multitasking. Once you have it, the natural follow-on questions —
*how to lay out memory across heterogeneous-length sequences* (paged
KV) and *how to share state across same-prefix sequences* (radix
trees) — both become obvious.

## 3. vLLM — PagedAttention as the load-bearing primitive

vLLM is built around **PagedAttention** `[kwon2023]`, treated in detail
in `kb/notes/inference/kv-cache-management.md §3`. The system design
contributions on top of the algorithm are:

- **Centralized scheduler + distributed workers.** A single scheduler
  process (Python) dispatches block-table updates to GPU workers
  (C++/CUDA) `[kwon2023 §4.6;
  kb/excerpts/kwon2023#sec-4-2]`. Workers see only physical block IDs;
  they do not coordinate on memory management directly.
- **First-come-first-served preemption.** Under capacity pressure the
  *latest-arrived* requests are preempted; their KV is either swapped
  to CPU RAM or freed and recomputed on resume `[kwon2023 §4.5;
  kb/excerpts/kwon2023#sec-4-5]`.
- **Co-designed kernels.** Three custom CUDA kernels: fused
  reshape+block-write (after each layer's K/V projection), block-aware
  attention (reads non-contiguous KV via the block table), and fused
  block copy (for copy-on-write at block granularity)
  `[kwon2023 §5.1; kb/excerpts/kwon2023#sec-4-1]`.
- **Sequence operations as the API.** vLLM exposes `fork`, `append`,
  `free` to higher layers, mirroring OS process operations. Beam
  search and parallel sampling are implemented in terms of these
  primitives, not as built-in modes
  `[kwon2023 §5.2; kb/excerpts/kwon2023#sec-4-1]`.

Reported headline: 2–4× over FasterTransformer and Orca on OPT-13B
through OPT-175B, more pronounced on long sequences, large models, and
sampling-heavy workloads `[kwon2023 §6.1;
kb/excerpts/kwon2023#sec-6-1-headline]`.

By 2026, vLLM has accreted: FlashAttention-2/3 backends, AWQ/GPTQ/FP8
weight loading, EAGLE-3 speculative decoding, XGrammar structured
output, multi-LoRA serving, and disaggregated prefill/decode
(experimental). The OSS engine is the de-facto baseline that other
papers compare against.

## 4. SGLang — programmable structure + RadixAttention

SGLang `[zheng2024-sglang]` enters the design space from a different
direction: rather than starting from "make request-level decoding
faster," it starts from "LM programs are not single requests — they
are *programs* with shared state, control flow, and structured
outputs."

### 4.1 The frontend DSL

The SGLang frontend is a Python-embedded DSL with primitives for
generation (`gen`, `select`), parallelism (`fork`, `join`), and
multi-modal input (`image`, `video`)
`[zheng2024-sglang §2; kb/excerpts/zheng2024-sglang#sec-1-distributed]`.
A single SGLang program may contain dozens of LLM calls with
shared system prompts and few-shot examples.

The runtime contribution that makes this fast is **automatic KV
sharing** — see §4.2 — which exploits the fact that calls within an
SGLang program share huge fractions of their token sequences.

### 4.2 RadixAttention

Detailed in `kb/notes/inference/kv-cache-management.md §4`. Headline:

- A radix tree over all in-flight and recently-completed token
  sequences indexes the KV cache.
- New requests automatically reuse the longest prefix match.
- LRU eviction with reference counts; cache-aware scheduler maximizes
  hit rate by depth-first traversal order
  `[zheng2024-sglang §3 Theorem 3.1;
  kb/excerpts/zheng2024-sglang#sec-3-cache-aware]`.

The crucial system property is **compatibility**: RadixAttention sits
on top of paged KV (page = 1 token) and is orthogonal to continuous
batching, tensor parallelism, and quantized weights
`[zheng2024-sglang §3; kb/excerpts/zheng2024-sglang#sec-3]`.

### 4.3 Compressed-FSM constrained decoding

For structured output (JSON schemas, regex), SGLang compiles the
constraint to a finite state machine and *compresses* singular-
transition chains into single edges. Multiple tokens that the FSM
forces deterministically can be decoded in one forward pass instead
of $k$ separate ones
`[zheng2024-sglang §4; kb/excerpts/zheng2024-sglang#sec-4]`. Treated
in detail in `kb/notes/inference/structured-output.md`.

### 4.4 Reported throughput

SGLang reports **up to 6.4×** over vLLM/Guidance/LMQL on workloads
with shared prefixes (multi-turn chat, agent control, RAG)
`[zheng2024-sglang abstract; kb/excerpts/zheng2024-sglang#abstract]`.
The 6.4× headline is on workloads that *exercise* RadixAttention and
compressed-FSM together; on single-turn generation without prefix
sharing the gap is much smaller.

[CONTRADICTION] Production benchmarks (`[FORUM-SIGNAL: 2026 vendor
comparisons]`) report SGLang 29% higher throughput than vLLM on models
≤70B and roughly parity at 70B+. The 6.4× number applies to a
specific workload class, not aggregate serving. Both are accurate
within their measurement scopes.

## 5. The 2025–2026 architectural shift — prefill / decode disaggregation

The prefill phase (processing the full prompt to produce the first
token's KV) is **compute-bound**: parallel matrix-matrix
multiplication scaling with $N \cdot d^2$. The decode phase
(generating subsequent tokens) is **memory-bandwidth-bound**: per-step
matrix-vector multiplication scaling with $d^2$.

Running both on the same GPU forces interference: a long prompt's
prefill stalls all in-flight decodes; conversely, latency-sensitive
decodes are starved by chunked-prefill scheduling.

The **disaggregated prefill/decode** architecture splits them onto
different GPU pools. The KV cache is migrated from prefill-pool to
decode-pool over the cluster fabric (NVLink-Switch, InfiniBand RDMA).

vLLM's docs (`[FORUM-SIGNAL: docs.vllm.ai disagg_prefill]`) note this
is **not** a throughput optimization but a tail-latency control
mechanism. The aggregate throughput is roughly conserved (you spent
the same FLOPs); what changes is that the prefill and decode
*latency distributions* decouple. Disaggregation also enables
heterogeneous hardware — H100 for prefill, A100 for decode, etc.

[CONTRADICTION] Some 2025 vendor blogs and academic papers claim
disaggregation *also* improves throughput in specific configurations
(e.g. when prefill batch size is mismatched to decode batch size in a
single-pool setup). vLLM's own documentation hedges. The actual
throughput-vs-latency Pareto surface for PD disaggregation is not
well-characterized empirically as of 2026-05; this is on Phase 1's
open-questions list.

## 6. The kernel substrate — FlashAttention

Every modern serving system uses FlashAttention-2/3 as the attention
kernel. Treated in detail in
`kb/notes/architecture/attention-mechanism.md §3.2`. Brief recap:

- FA-1 `[dao2022]` — IO-aware tiling + recomputation; same math as
  Eq.(1) of attention, $O(N^2 d^2 / M)$ HBM accesses with $M$ = SRAM.
- FA-2 `[dao2023]` — re-partitioned work, more matmul-bound, ~2× over
  FA-1 on A100.
- FA-3 `[shah2024]` — Hopper async warp-spec + FP8, ~1.5–2× over FA-2
  on H100.

PagedAttention's kernel `[kwon2023 §5.1]` is FA-2-style but with
block-aware K/V reads from the block table. SGLang uses standard
FlashAttention with the radix-tree-page-pool layout.

## 7. The performance Pareto, 2026

[FORUM-SIGNAL: 2026 vendor benchmarks; not a primary citation]

| Engine | Strength | Weakness |
|---|---|---|
| **vLLM** | Most compatible (fastest model upstreaming), highest throughput at very high concurrency, biggest community | Mid-pack on small-batch latency |
| **SGLang** | Highest throughput ≤70B, lowest latency on prefix-sharing workloads, best DSL ergonomics | Smaller community, fewer model variants |
| **TensorRT-LLM** | Lowest p50 / p95 TTFT (first-token latency); kernel-level NVIDIA optimization | NVIDIA-only, less flexible, slower model onboarding |
| **llama.cpp / ggml** | CPU + consumer-GPU + Metal + RPC; the GGUF ecosystem | Not throughput-optimized for datacenter |

Choice is workload-dependent. For chat with long shared system
prompts, SGLang. For raw datacenter throughput on large MoE models,
vLLM (since it inherits day-one DeepSeek-V3 / Llama-4 support).
For minimum first-token latency, TensorRT-LLM.

## 8. Frontier and open questions (as of 2026-05)

- **PD disaggregation tradeoff surface.** The throughput-vs-latency
  Pareto for disaggregation under realistic mixed workloads is open.
  vLLM's blog notes the feature is experimental as of v0.7.
- **Multi-tenant cache sharing.** RadixAttention shares cache *within*
  a tenant's request stream by default. Sharing across tenants is
  correctness-safe (the cache is just memoized prefills) but leaks
  timing information about other tenants' prompts. The privacy model
  is not addressed in primary publications.
- **Heterogeneous GPU pools.** With H100 + A100 + L40S in the same
  cluster, optimal request-to-pool assignment becomes an online
  scheduling problem. Active in 2025–2026 research.
- **Long-context serving.** 1M+ context windows (Gemini 2.5,
  Llama-4) push the KV cache outside single-node memory. Ring-attention
  `[ring-attention-2023]` was originally a *training* technique;
  serving-side sharded KV is an emerging area.
- **Dynamic LoRA + multi-tenant.** Serving thousands of fine-tuned
  LoRAs concurrently (S-LoRA, Punica) is a 2024 contribution that
  vLLM and SGLang have integrated. The cache implications are
  non-trivial.

## 9. See also

- `kb/notes/inference/kv-cache-management.md` — PagedAttention and
  RadixAttention details. Upstream of the serving design.
- `kb/notes/inference/speculative-decoding.md` — vLLM and SGLang both
  integrate EAGLE/Medusa as scheduler-level features.
- `kb/notes/inference/structured-output.md` — XGrammar as the modern
  structured-output substrate; backend-default in vLLM/SGLang/TRT-LLM.
- `kb/notes/architecture/attention-mechanism.md §3.2` — FlashAttention
  family as the kernel substrate.
