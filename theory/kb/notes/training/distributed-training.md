---
topic_key: distributed-training
area: training
status: draft
last_updated: 2026-05-04
primary_sources: [torchtitan2024, deepseek-v3]
secondary_sources: [muon-moonlight2025]
related_topics: [optimization, mixed-precision-and-stability, mixture-of-experts]
---

# Distributed training

## Formal definition {#def}

A distributed training run partitions the *parameters*, *optimizer state*,
*activations*, and *data* of one logical model across $G$ GPU workers, so
that each step's mathematical effect on parameters is identical to a
single-GPU run with the same global batch. The partition is described by
a tuple of *parallelism axes*; the modern open practice uses up to five
simultaneously `[torchtitan2024 §1; kb/excerpts/torchtitan2024#sec-1-axes]`:

| Axis | Splits | Notation here |
|---|---|---|
| Data Parallel (DP) | per-batch (with full-replica or sharded params) | $d$ |
| Tensor Parallel (TP) | within a layer (matmul split column-/row-wise) | $t$ |
| Sequence/Context Parallel (SP/CP) | along the token-position axis | $c$ |
| Pipeline Parallel (PP) | across layer ranges (stages) | $p$ |
| Expert Parallel (EP) | across MoE experts (one expert subset per worker) | $e$ |

A configuration is then $G = d \cdot t \cdot c \cdot p \cdot e$ (subset
of axes used). The design problem is to pick the axis sizes — call this
a *parallelism plan* — that maximizes Model FLOP Utilization (MFU) under
the constraints of memory, interconnect bandwidth, and the model's
shape. *FSDP* is the most common DP variant: a sharded-parameter ZeRO
implementation
`[torchtitan2024 §2.1.2; kb/excerpts/torchtitan2024#sec-2-1-2-fsdp2]`.

## Mechanism — what each axis costs {#mechanism}

**DP (FSDP / ZeRO).** Each step, every worker materializes the full
parameter set for its layers via all-gather, computes its
micro-batch's forward/backward, then reduce-scatters gradients into the
sharded optimizer state. Per-GPU memory drops as $1/d$; per-step
collective traffic is $\Theta(\text{params})$. FSDP2's per-parameter
DTensor sharding reduces peak memory by ~7 % and improves throughput
~1.5 % vs. FSDP1's flat-parameter design
`[torchtitan2024 §2.1.2; kb/excerpts/torchtitan2024#sec-2-1-2-fsdp2]`.

**TP + SP.** Splits each linear's weight matrix into $t$ row- or
column-shards. Forward and backward each insert one all-reduce per
linear. Sequence-Parallel additionally shards normalization layers
along the token dimension to cut activation memory. Naive TP serializes
matmul on collective; *Async TP* fractionalizes the sharded matmul and
overlaps each chunk's compute with the previous chunk's collective via
intra-node `SymmetricMemory` shared buffers
`[torchtitan2024 §2.2.3; kb/excerpts/torchtitan2024#sec-2-2-3-async-tp]`.

**PP.** Split the layer stack into $p$ stages, each stage on a distinct
worker group. Microbatches flow stage-to-stage via P2P send/recv. The
classic 1F1B schedule has bubble cost $(p-1)(F+B)$ where $F, B$ are
per-microbatch forward/backward time; ZeroBubble (ZB1P) reduces it to
$(p-1)(F + B - 2W)$ where $W$ is the weight-gradient sub-step
`[deepseek-v3 §3.2.1, table; kb/excerpts/deepseek-v3-training#sec-3-2-1-bubble]`.

**CP (Context Parallel).** Splits along the token-position axis;
attention computation is partitioned via ring-attention-style
communication. The reported trainable context length on Llama 3.1 8B
with 8 H100s reaches 262 144 tokens with CP enabled, vs. OOM without
`[torchtitan2024 §2.1.5; kb/excerpts/torchtitan2024#sec-2-1-5-cp]`.

**EP.** Each MoE layer's experts are partitioned across $e$ workers; on
each token's forward pass, the dispatch is an *all-to-all* that routes
each token to the worker(s) hosting its top-$k$ experts, followed by a
combine all-to-all. This is bandwidth-heavy and cross-node by default,
which is the dominant communication cost in MoE training and the
specific problem DualPipe (below) is designed to hide.

## DeepSeek-V3 training plan as case study {#deepseek}

DeepSeek-V3 is the highest-throughput open recipe of 2024 for an MoE at
scale (671 B total / 37 B active, 14.8 T tokens, 2.788 M H800-hours,
$5.576 M cluster cost) `[deepseek-v3 §abstract, §1;
kb/excerpts/deepseek-v3-training#abstract,
kb/excerpts/deepseek-v3-training#sec-1-cost]`. The plan is
`[deepseek-v3 §3.2; kb/excerpts/deepseek-v3-training#sec-3-2-parallelism]`:

- $d$ = ZeRO-1 DP (optimizer-state sharding only — gradients and params
  stay replicated within DP)
- $p = 16$ PP (DualPipe schedule)
- $e = 64$ EP, spanning 8 nodes (8 GPUs per node, 1 expert shard each)
- *no TP*, deliberately — the FP8 + DualPipe + AsyncTP-style overlap
  removes the activation-memory pressure that would normally force TP

**DualPipe — the bidirectional bubble-eater.** Each forward + backward
chunk is split into four sub-stages
(`attention`, `all-to-all dispatch`, `MLP`, `all-to-all combine`),
backward is further split into `backward-for-input` and
`backward-for-weights` (ZeroBubble decomposition), and microbatches are
fed *from both ends of the pipeline simultaneously*. The bubble formula
becomes $(\tfrac{p}{2} - 1)(F\&B + B - 3W)$ (about half of 1F1B) at the
cost of 2× parameter memory and $+1$ activation copy
`[deepseek-v3 §3.2.1, table; kb/excerpts/deepseek-v3-training#sec-3-2-1-bubble]`.

**Cross-node all-to-all kernels.** NVLink offers 160 GB/s intra-node;
IB offers 50 GB/s inter-node — a 3.2× ratio. To stay IB-bound only on
the inter-node hop, DeepSeek-V3 limits each token to **at most 4
nodes**, sends one IB transmit per (token, node) to the GPU with the
matching in-node index, then NVLink-forwards inside the destination node
to the GPU that actually hosts the expert. This makes the IB and
NVLink phases overlap and lets each token select 3.2 experts/node
average without extra NVLink cost
`[deepseek-v3 §3.2.2; kb/excerpts/deepseek-v3-training#sec-3-2-2-allreduce]`.

## Memory accounting {#memory}

Let $P$ = parameters, $A$ = peak per-microbatch activations.
Approximate per-GPU memory under each axis:

| Axis combo | Per-GPU param | Per-GPU optimizer state | Per-GPU activation |
|---|---|---|---|
| Pure DP (ZeRO-0) | $P$ | $\sim 4P$ (FP32 + Adam) | $A$ |
| ZeRO-1 / FSDP optimizer-shard | $P$ | $4P/d$ | $A$ |
| ZeRO-2 / FSDP grad-shard | $P$ | $4P/d$ + grads $/d$ | $A$ |
| ZeRO-3 / FSDP full-shard | $P/d$ | $4P/d$ | $A$ |
| TP only (size $t$) | $P/t$ | $4P/t$ | $A/t$ along sharded dim |
| PP only (stages $p$) | $P/p$ | $4P/p$ | $A \cdot p$ in flight |
| CP (size $c$) | $P$ | $4P$ | $A/c$ |

Two tensions visible in the table:

- *PP saves params but multiplies in-flight activations* by the number of
  microbatches in the pipeline — exactly why DualPipe's $+1$ activation
  copy is a *small* cost relative to its bubble savings.
- *FSDP and TP both shard parameters*, but FSDP communicates per-step
  while TP communicates per-layer. At small models, FSDP's per-step
  collective dominates; at large models, TP's per-layer collective
  dominates.

## Variants & lineage {#variants}

| System | Year | Distinctive choice | Source |
|---|---|---|---|
| Megatron-LM (NVIDIA) | 2019- | First TP+PP at scale | Shoeybi et al. |
| ZeRO-1/2/3 (DeepSpeed) | 2020-21 | Optimizer/grad/param sharding gradient | Rajbhandari et al. |
| FSDP1 (PyTorch) | 2022 | Native ZeRO-3 with FlatParameter | Zhao et al. |
| FSDP2 (PyTorch) | 2024 | Per-parameter DTensor sharding | `[torchtitan2024 §2.1.2]` |
| Megatron 3D | 2021 | TP × PP × DP composition | Narayanan et al. |
| GSPMD (XLA) | 2021 | Sharding propagation in compiler | Xu et al. |
| TorchTitan 4D | 2024 | DP × TP × PP × CP composable | `[torchtitan2024 §abstract]` |
| DeepSpeed-MoE / Tutel | 2022-23 | EP + all-to-all dispatch kernels | Rajbhandari et al. |
| DeepSeek-V3 / DualPipe | 2024 | Bidirectional PP + ZB decomposition | `[deepseek-v3 §3.2.1]` |
| Distributed Muon (Moonlight) | 2025 | Per-rank Newton-Schulz inside FSDP | `[muon-moonlight2025 §2.3]` |

The trajectory has been towards *composability*: 2020-era systems forced
a single axis choice (Megatron = TP+PP, DeepSpeed = ZeRO); 2024-era
systems (TorchTitan, the DeepSeek stack) treat each axis as an
independent dimension on a DTensor mesh and let users compose them.

## Intuitions & analogies {#intuitions}

**[ANALOGY] Parallelism axes as a 5-dimensional crystal.** Each
GPU-worker sits at one cell of a $d \times t \times c \times p \times e$
mesh; the "shape" of the crystal determines which collectives fire on
which axis. Returning to canonical form: this is exactly what
`DTensor.DeviceMesh` represents in PyTorch
`[torchtitan2024 §2.1.1; kb/excerpts/torchtitan2024#sec-2-1-1-meta]` —
the analogy is literal, not just suggestive.

**[INTUITION] Pipeline bubble = startup + shutdown of an assembly
line.** With $p$ stages and $m$ microbatches, the first microbatch must
walk forward through all $p$ stages before any backward can start; the
last microbatch's backward must walk *back* through all $p$ stages
after the last forward finishes. Those $2(p-1)$ stage-times are
"bubble." 1F1B reduces them but cannot eliminate them; ZeroBubble
splits backward into two sub-steps, of which one ($W$, weight-gradient)
can be reordered into the bubble; DualPipe additionally feeds the
pipeline from both ends so the first-microbatch ramp on one side
overlaps with the last-microbatch ramp on the other
`[deepseek-v3 §3.2.1; kb/excerpts/deepseek-v3-training#sec-3-2-1-dualpipe]`.

**[INTUITION] Why MoE forces EP.** At $E$ experts per layer with each
expert holding $P_E$ parameters, the layer's parameter count is
$E \cdot P_E$. DeepSeek-V3 has 256 routed experts per MoE layer; one
GPU cannot hold that. EP slices experts across workers; the cost is
that *every token's routing decision becomes an all-to-all*, which is
why a full third of the DeepSeek-V3 paper is communication kernels.

**[CONTRADICTION] TP yes vs. TP no.** Llama 3 405B used heavy TP+PP+DP
(Meta's choice). DeepSeek-V3 671B-A37B used **no TP**, claiming the
combination of ZeRO-1 DP + EP + DualPipe + FP8 had enough activation
relief without it. This is a real disagreement, not a wash: TP costs
inter-GPU collectives on every linear; EP costs inter-node all-to-all
on every MoE layer; DeepSeek picked the second because their MoE was
already paying for it. The general lesson is that "which parallelism is
best" is a function of model architecture, not a universal answer
`[deepseek-v3 §3.2; torchtitan2024 §2.1.3]`.

## Frontier & open questions {#frontier}

1. **Optimal plan from the model graph.** No public tool reliably picks
   $(d, t, c, p, e)$ from $(N, D, \text{cluster shape})$. The current
   practice is hand-tuning by ML systems engineers. GSPMD and
   AlpaDiscovery moved this direction but neither is the dominant tool
   in 2026 open practice.

2. **5D + Float8 + AsyncTP composability.** TorchTitan's headline claim
   is that all five axes compose; in practice, FP8 + AsyncTP + PP
   schedules with non-trivial overlap (ZeroBubble, Flexible-Interleaved-1F1B)
   are still the bleeding edge and reportedly fragile.

3. **Distributed Muon at very large scale.** Moonlight (16 B-A3 B MoE,
   5.7 T tokens) demonstrated Muon can be sharded inside FSDP via
   per-rank Newton-Schulz on a partial-gradient `[muon-moonlight2025
   §2.3; kb/excerpts/muon-moonlight2025#sec-2-3-distributed]`. Whether
   this scales to 70 B+ dense or 600 B+ MoE is an open empirical
   question. See [optimization](optimization.md).

4. **Communication-aware scheduling.** DualPipe's bidirectional schedule
   is a special case of "schedule-as-IR + compiler passes for
   collectives" `[torchtitan2024 §2.1.4;
   kb/excerpts/torchtitan2024#sec-2-1-4-pp]`. Generalizing this to
   arbitrary models without per-architecture hand-tuning is open.

5. **InfiniBand vs. NVLink ratio.** DeepSeek-V3's 4-node-cap rule
   `[deepseek-v3 §3.2.2;
   kb/excerpts/deepseek-v3-training#sec-3-2-2-allreduce]` exploits the
   3.2× NVLink/IB ratio on H800. Future hardware (Blackwell NVL72) flips
   this — the entire 72-GPU rack is one NVLink domain. Cross-rack-only
   IB will change which token-routing strategies are optimal, and the
   topology-aware kernels above will need to be rewritten.
