---
topic: training/mixed-precision-and-stability
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - micikevicius2017-mixed-precision
  - kalamkar2019-bfloat16
  - deepseek-v3
  - scaling-laws-precision-2024
  - torchtitan2024
secondary_sources:
  - muon-moonlight2025
  - shoeybi2019-megatron
  - ma2024-bitnet
related_topics:
  - training/optimization
  - training/distributed-training
  - inference/quantization
  - scaling/scaling-frontier
---

# Mixed precision and training stability

Mixed-precision training is the collection of techniques that let a
forward/backward pass use lower numerical precision than the master
weights, without observable divergence from a higher-precision baseline.
The lineage runs FP32 → FP16 + master-FP32 (Micikevicius et al. 2017
`[micikevicius2017-mixed-precision]`) → BF16 (Kalamkar et al. 2019
`[kalamkar2019-bfloat16]`) → FP8 fine-grained (DeepSeek-V3 §3.3
`[deepseek-v3 §3.3; kb/excerpts/deepseek-v3-training#sec-3-3-fp8]`,
TorchTitan Float8 `[torchtitan2024 §2.2.4]`) → NVFP4 / FP4 (Blackwell-
era, 2025–26).

Beyond compute precision, the topic includes the **stability machinery**
that prevents loss spikes and divergence: loss scaling, scale-block
fine-graining, accumulation-precision promotion, attention-logit
scaling, embedding-norm control, and outlier handling. The headline
empirical fact as of 2026 is that **DeepSeek-V3's full 14.8T-token /
2-month / 2048-H800 pre-training reported zero irrecoverable loss
spikes and zero rollbacks**
`[deepseek-v3 §abstract; kb/excerpts/deepseek-v3-training#abstract]` —
demonstrating that fine-grained FP8 training is now production-stable.

## 1. Formal definition

### 1.1 The three precision axes

Mixed precision training has three independently-tunable precision
choices `[micikevicius2017-mixed-precision §2]`:

1. **Storage precision** — the format of the master weights $\theta^*$,
   gradients $g$, and optimizer state ($m, v$ for Adam).
2. **Compute precision** — the format of the matrix-multiply (Tensor
   Core GEMM) inputs and accumulators.
3. **Communication precision** — the format used for all-reduce and
   reduce-scatter in distributed training (often coupled to compute
   precision).

The master-weights pattern (Micikevicius et al. 2017) keeps a high-
precision copy $\theta^* \in \mathbb{R}^P$ in FP32 and a low-precision
working copy $\theta \in \mathbb{R}^P$ for the GEMM. Updates are
applied to $\theta^*$, then $\theta^*$ is cast down to $\theta$.
Schematically, with FP16 forward/backward and FP32 master weights:

$$
\begin{aligned}
y &= \mathrm{Forward}(\theta_{\text{FP16}}, x) \\
g_{\text{FP16}} &= \mathrm{Backward}(y, \mathrm{loss}) \\
g^*_{\text{FP32}} &= \mathrm{cast}(g_{\text{FP16}}) \\
\theta^*_{\text{FP32}} &\leftarrow \theta^*_{\text{FP32}} - \eta \cdot \mathrm{Optim}(g^*_{\text{FP32}}) \\
\theta_{\text{FP16}} &= \mathrm{cast}(\theta^*_{\text{FP32}})
\end{aligned} \tag{1}
$$

`[micikevicius2017-mixed-precision §2]`

| Symbol | Meaning |
|---|---|
| $\theta^*$ | master weights (high precision) |
| $\theta$ | working weights (low precision; same $P$ entries) |
| $g$ | gradients (computed in low precision via autograd) |
| $\eta$ | learning rate |
| $\mathrm{Optim}$ | optimizer step (Adam, Muon, etc.) |

### 1.2 Quantization-aware GEMM: $1/Z$ decode and FP32 accumulation

For sub-FP16 formats (FP8, FP4) the per-tensor or per-block scale
$Z$ becomes a first-class object. The operation
$Y = X W$ in FP8 is implemented as:

$$
Y = (Z_X \cdot X_{\text{FP8}})\, (Z_W \cdot W_{\text{FP8}}) = Z_X Z_W \, (X_{\text{FP8}} W_{\text{FP8}})_{\text{FP8 accum}} \tag{2}
$$

with the inner GEMM done on FP8 Tensor Cores and the scalar product
$Z_X Z_W$ folded in afterward. The **accumulator precision** of the
FP8 Tensor Core matters: H100's FP8 GEMM accumulates internally in
"around 14 bits" rather than full FP32
`[deepseek-v3 §3.3.2; kb/excerpts/deepseek-v3-training#sec-3-3-2-accum]`,
giving up to ~2% relative error on $K = 4096$ dot products. The fix is
**promotion to CUDA cores at interval $N_C$** — every $N_C = 128$
elements of $K$, copy partial sums to FP32 registers and accumulate in
true FP32 there `[deepseek-v3 §3.3.2;
kb/excerpts/deepseek-v3-training#sec-3-3-2-accum]`.

### 1.3 Format zoo

| Format | Bits | Exponent | Mantissa | Dyn. range $\approx$ | Subnormal | Used for |
|---|---|---|---|---|---|---|
| **FP32** | 32 | 8 | 23 | $10^{\pm 38}$ | yes | master weights, FP accum, sensitive ops |
| **FP16 (half)** | 16 | 5 | 10 | $6.55\times10^{4}$ to $6\times10^{-5}$ | yes | original mixed-precision compute |
| **BF16 (bfloat16)** | 16 | 8 | 7 | $10^{\pm 38}$ (FP32-like) | yes | Ampere+ default; less mantissa precision |
| **FP8 E4M3** | 8 | 4 | 3 | $\pm 448$ | yes | Hopper FP8 Fprop / Dgrad / Wgrad |
| **FP8 E5M2** | 8 | 5 | 2 | $\pm 5.7\times 10^{4}$ | yes | NVIDIA hybrid: gradients only |
| **NVFP4 / FP4 E2M1** | 4 | 2 | 1 | $\pm 6$ | (varies) | Blackwell-era; experimental |

`[micikevicius2017-mixed-precision §3, §4]` for FP16; Kalamkar et al.
2019 §2 `[kalamkar2019-bfloat16]` for BF16; NVIDIA Hopper specs
`[FORUM-SIGNAL: NVIDIA H100 white paper 2022]` for FP8 hybrid;
DeepSeek-V3 §3.3.2 `[deepseek-v3 §3.3.2;
kb/excerpts/deepseek-v3-training#sec-3-3-2-mantissa]` for the all-E4M3
choice.

The defining tradeoff: **range vs. precision**. FP16 has more mantissa
bits (more precision per representable value) but a much smaller
dynamic range than BF16, making FP16 prone to underflow on small
gradient values. BF16 has FP32-like range but only 7 mantissa bits, so
products of small numbers lose precision. FP8 doubles the underflow
problem.

## 2. Mechanism — stability machinery

### 2.1 Loss scaling (FP16)

Plain FP16 training underflows: small gradient values fall below
$6\times 10^{-5}$ (the smallest normal FP16 value) and round to zero.
Micikevicius et al. 2017 introduce **loss scaling**: multiply the loss
by a scale factor $S$ before backprop, divide gradients by $S$ before
the master-weights update `[micikevicius2017-mixed-precision §3]`:

$$
\tilde{g} = \nabla (S \cdot \mathcal{L}) = S \cdot g, \quad g^*_{\text{FP32}} = \mathrm{cast}(\tilde{g}_{\text{FP16}}) / S \tag{3}
$$

The chain rule is exact; the only effect is to shift the gradient
distribution upward in the FP16 representable range. **Dynamic loss
scaling** doubles $S$ each step until an Inf/NaN occurs, then halves
$S$ and skips the update. BF16 and FP8 don't need loss scaling (their
exponent range is FP32-like or scaled per-block).

### 2.2 BF16 — drop-in replacement for FP32 on Ampere+

Kalamkar et al. 2019 `[kalamkar2019-bfloat16 §2]` showed BF16's range-
matched-to-FP32 design lets training proceed without loss scaling and
without master-FP32 weights in many cases. The 7-mantissa precision is
the cost. By 2023 BF16 is the universal pre-training default for
LLaMA-2/3, Mistral, Qwen, Gemma, OLMo `[FORUM-SIGNAL: model-card
inspection across releases]`.

[INTUITION] BF16's "free lunch" relative to FP16 is range. Range
dominates for gradients (whose magnitude varies wildly across the
network); precision loss matters less than underflow.

### 2.3 FP8 fine-grained quantization (DeepSeek-V3)

DeepSeek-V3's recipe is the most aggressive published FP8 training to
date `[deepseek-v3 §3.3, §3.3.1, §3.3.2, §3.3.3;
kb/excerpts/deepseek-v3-training#sec-3-3-fp8,
kb/excerpts/deepseek-v3-training#sec-3-3-1-mixed,
kb/excerpts/deepseek-v3-training#sec-3-3-2-finegrained,
kb/excerpts/deepseek-v3-training#sec-3-3-2-accum,
kb/excerpts/deepseek-v3-training#sec-3-3-2-mantissa,
kb/excerpts/deepseek-v3-training#sec-3-3-3-storage]`:

1. **All three GEMMs in FP8.** Forward (`Fprop`), activation backward
   (`Dgrad`), weight backward (`Wgrad`) all run at FP8.
2. **Fine-grained scaling.** Activations: per-token-per-128-channel
   tile (1×128). Weights: per 128×128 block. Gradients: same as
   activations. Per-group scale factors along the inner dimension of
   the GEMM allow scale to adapt to local outliers.
3. **E4M3 on every tensor.** NVIDIA's recommendation is hybrid (E4M3
   forward, E5M2 for gradients) for more dynamic range. DeepSeek
   instead uses E4M3 *everywhere*, relying on fine-grained scaling to
   absorb the dynamic range that hybrid would have provided.
4. **Promotion to CUDA cores at $N_C = 128$.** Tensor Core's 14-bit
   accumulation is promoted to FP32 accumulation in CUDA cores every
   128 elements of the inner reduction.
5. **Sensitive operators kept higher.** Embedding, output head, MoE
   gating, normalization, attention all remain BF16 or FP32. The
   **master weights, weight gradients, and optimizer state** stay in
   FP32 (master, gradients) or BF16 (Adam first/second moments).
6. **Empirical envelope.** Relative loss error vs BF16 stays $\leq
   0.25\%$ at scale `[deepseek-v3 §3.3;
   kb/excerpts/deepseek-v3-training#sec-3-3-fp8]`.

[INTUITION] Fine-grained scaling is what makes FP8 viable: a single
per-tensor scale must compromise between outliers (which need a large
$Z$) and bulk values (which need a small $Z$ for precision). Per-tile
scaling lets each 128-element tile choose its own $Z$, so outliers in
one tile don't blow up precision in adjacent tiles. Returning to math:
Eq. (2) generalizes to per-tile $Z_X^{(t)}$ and $Z_W^{(b)}$ along the
inner dimension, and the GEMM accumulates across tiles in CUDA-core FP32.

### 2.4 TorchTitan's Float8 path

TorchTitan `[torchtitan2024 §2.2.4]` integrates `torchao.float8` per-
tensor scaling (dynamic / delayed / static) selectively on `Linear`
layers. This is a less aggressive scheme than DeepSeek-V3's tile/block
scaling: per-tensor scaling means a single scale per Linear input or
weight. The TorchTitan path achieves a fraction of DeepSeek-V3's FP8
compute savings (TorchTitan reports speedups in the 30–65% range on
Llama-3.1 with combined FP8 + FSDP2 + TP, depending on scale
`[torchtitan2024 §abstract;
kb/excerpts/torchtitan2024#abstract]`), but is much easier to apply
generically and doesn't require recompiling Tensor-Core kernels.

### 2.5 Loss-spike taxonomy

Pre-training loss spikes — sudden jumps in $\mathcal{L}$ that may or
may not recover — are observed across all frontier-scale runs. Sources
documented across PaLM (Chowdhery et al. 2022), OPT (Zhang et al.
2022), GLM-130B (Zeng et al. 2023), and discussed in OPT's logbook:

1. **Gradient-norm explosion** — typical of LR-overshoot or batch-size
   transitions; addressed by **gradient clipping** (clip $\|g\|_2 \leq
   c$) and a brief LR drop.
2. **Attention-logit divergence** — softmax inputs grow in magnitude
   as training proceeds, creating numeric overflow in attention. The
   2024–2025 fix is **QK-clip** (clamping $QK^\top$ before softmax) or
   **QK-LayerNorm** (Henry et al. 2020; LLaMA-2's "norm-Q-K"),
   normalizing $Q$ and $K$ before the dot product.
3. **Embedding-norm growth** — token embeddings can grow without bound
   under weight decay if the optimizer's preconditioning interacts
   poorly with embedding sparsity. **Tied embeddings** (sharing input
   and output embedding matrices) and **weight-tied LayerNorm scale**
   help.
4. **FP8 underflow on rare tokens** — rare vocabulary entries' embedding
   columns receive few updates and small gradient magnitudes, which
   underflow in FP8. Mitigation: keep embeddings in BF16/FP32 (DeepSeek-
   V3 does so explicitly).
5. **MoE expert collapse** — auxiliary load-balancing loss can fail in
   FP8, especially for expert-routing softmax. DeepSeek-V3's auxiliary-
   loss-free load balancing partially addresses this `[deepseek-v3
   §abstract; kb/excerpts/deepseek-v3-training#abstract]`.

The DeepSeek-V3 result — zero irrecoverable spikes over 14.8T tokens —
is partly attributable to careful engineering of items 2, 3, 4, and 5
in tandem with FP8's increased precision pressure.

### 2.6 Muon's interaction with low precision

Muon's update RMS is naturally $\sqrt{1/\max(A,B)}$, which combined
with growing weight RMS "exceeds the high-precision range of bf16"
without weight decay `[muon-moonlight2025 §2.2;
kb/excerpts/muon-moonlight2025#sec-2-2-wd]`. The Muon-Moonlight fix
is mandatory weight decay and per-shape RMS-scaled updates so that the
update + weight stay within BF16's representable range — partly a
stability-against-low-precision mechanism `[muon-moonlight2025 §2.2;
kb/excerpts/muon-moonlight2025#sec-2-2-rms]`. Optimizer choice and
precision choice are not orthogonal.

## 3. Variants and lineage

### 3.1 Comparison table

| Recipe | Year | Activation/weight | Optim. state | Speedup vs FP32 | Stability machinery |
|---|---|---|---|---|---|
| **FP32 baseline** | <2017 | FP32 | FP32 | 1× | none |
| **FP16 + master FP32** `[micikevicius2017-mixed-precision]` | 2017 | FP16 / FP16 | FP32 | ~2–3× | dynamic loss scaling |
| **BF16** `[kalamkar2019-bfloat16]` | 2019 | BF16 / BF16 | FP32 | ~2× | none required |
| **FP8 hybrid (NVIDIA)** | 2022–23 | E4M3 fwd / E5M2 bwd | FP32 | ~3–4× | per-tensor scaling |
| **FP8 fine-grained (DeepSeek-V3)** `[deepseek-v3 §3.3]` | 2024 | E4M3 / E4M3 (1×128 / 128×128) | FP32 master, BF16 Adam | ~4× | promotion to CUDA cores @ $N_C{=}128$ |
| **TorchTitan Float8** `[torchtitan2024 §2.2.4]` | 2024 | per-tensor FP8 on Linear | FP32 | up to 65% speedup | dynamic / delayed / static scale |
| **NVFP4** | 2025 | FP4 (E2M1) per-block | FP32/BF16 | ~2× over FP8 [SPECULATION] | hardware MXScale; per-32-element |
| **BitNet 1.58** `[ma2024-bitnet]` | 2024 | ternary $\{-1,0,1\}$ wts; INT8 act | FP32/BF16 | inference-only | not reported as training-stable at scale |

### 3.2 Scaling laws for precision

Kumar, Pearce, Huang et al. 2024 `[scaling-laws-precision-2024]` derive
scaling laws connecting **training-bit-width** $b_{\mathrm{train}}$,
**inference-bit-width** $b_{\mathrm{inf}}$, parameter count $N$, and
data $D$ to expected loss. Two headline claims:

1. **Effective parameter count** $N_{\text{eff}}(N, b_{\text{train}})$
   shrinks as training precision drops below ~7 bits. Below this
   threshold, training a larger model in lower precision can be
   compute-equivalent to training a smaller model in higher precision.
2. **Inference-bit-width loss penalty** is well-fit by a function of
   $D / N$ and $b_{\mathrm{inf}}$: heavily-overtrained models (large
   $D/N$) suffer more from low inference precision. This is a critical
   prediction for Chinchilla-violating regimes (Llama-3 at $D/N \sim
   200$).

The work is recent and the headline numbers depend on small-scale fits;
frontier-scale validation pending.

### 3.3 Communication precision

In distributed training the all-reduce of gradients is a separate
precision choice. Megatron-LM `[shoeybi2019-megatron]` showed that all-
reduce in FP16 with FP32 reduction on a master node maintains stability
for tensor-parallel runs. TorchTitan's **AsyncTP** and SymmetricMemory
features `[torchtitan2024]` let the all-reduce overlap with compute,
making low-precision comm even more attractive. DeepSeek-V3's cross-
node all-to-all kernels `[deepseek-v3 §3.2.2;
kb/excerpts/deepseek-v3-training#sec-3-2-2-allreduce]` use FP8 dispatch
and BF16 combine, with the bandwidth budget split between IB and NVLink.

## 4. Intuitions and analogies

[INTUITION] **Precision as compression**. Each precision-step (FP32 →
BF16 → FP8 → FP4) halves storage and roughly doubles compute throughput
on supporting hardware. Training in lower precision is a form of
*lossy compression of the gradient computation*, recovering quality
through (a) keeping a high-precision master copy that aggregates many
small updates and (b) per-block scaling that makes the loss tile-local
rather than tensor-wide. Returning to canonical form: the master-
weights pattern in Eq. (1) is exactly the lossy-compression-with-error-
correction pattern.

[ANALOGY] **Tile/block scaling is to FP8 what KV pages are to long
context**. Both factor a tensor into smaller homogeneous units, each
with its own metadata (scale or block ID), so that the ill-behaved
parts (outliers / long sequences) are isolated from the well-behaved
parts. This [ANALOGY] returns to math: per-tile scaling makes the
quantization error in Eq. (2) bounded by the **per-tile** outlier-to-
median ratio rather than the tensor-wide ratio; PagedAttention makes
KV-cache memory cost depend on per-page utilization rather than worst-
case sequence length.

[INTUITION] **Why E4M3 everywhere works in DeepSeek-V3 but not in
NVIDIA's hybrid**. NVIDIA's E5M2-for-gradients is meant to give
gradients more dynamic range. DeepSeek-V3 substitutes that with **per-
tile scaling** (1×128 for activations and gradients), which gives each
tile its own dynamic range. Once the tile owns the range, the format
need not — so E4M3's extra mantissa bit is pure precision gain. Returns
to math: the effective dynamic range of an E4M3 tile with scale $Z$ is
$Z \cdot [-448, 448]$, and choosing $Z$ per-tile recovers the dynamic
range hybrid was buying with E5M2.

[CONTRADICTION] **Whether NVFP4 (FP4) is production-stable for full
pre-training**. As of 2026-05 several Blackwell-deploying vendors are
running FP4 pilots; DeepSeek and Anthropic have not publicly confirmed
FP4 pre-training. The scaling-laws-for-precision result
`[scaling-laws-precision-2024]` predicts effective-parameter loss
becomes substantial below ~7 bits, but the prediction is fitted on
small models. Treat FP4 frontier-scale stability as open until a
post-hoc tech report appears.

## 5. Frontier and open questions (as of 2026-05)

### 5.1 NVFP4 / FP4 at frontier scale

Blackwell-class GPUs (B100/B200) ship native FP4 Tensor Cores with
hardware MXScale (32-element per-block scaling). The compute-throughput
gain over FP8 is roughly 2× by spec. Whether the
fine-grained-scaling-plus-promotion recipe of DeepSeek-V3 transfers
cleanly to FP4 — or whether the loss-spike envelope grows — is the
open frontier question. [SPECULATION] The 4-bit mantissa loss is
qualitatively different from 8-bit and may break the "fine-grained
scaling absorbs all dynamic range" intuition.

### 5.2 Stability of attention with FP8 logits

A live concern as model size grows is **attention-logit blow-up**.
Softmax over FP8 logits can saturate when $|QK^\top|$ exceeds E4M3's
$\pm 448$ bound. DeepSeek-V3 keeps the attention operator in BF16 for
this reason `[deepseek-v3 §3.3.1;
kb/excerpts/deepseek-v3-training#sec-3-3-1-mixed]`. Whether attention
itself can move to FP8 with QK-clip + per-head scale is an open
research question; FlashAttention-3 reports FP8 attention with block-
quantization in inference but not training `[shah2024 §2]`.

### 5.3 [CONTRADICTION] Empirical loss-spike taxonomy

Frontier labs report consistent qualitative spike causes (gradient
norm, attention logit, embedding norm) but disagree on the relative
prevalence. PaLM, OPT, and Llama-2 each call out different causes as
dominant in their runs. No published controlled study compares spike
causes across architectures + scales + precision regimes. This is
implementation- and infrastructure-dependent more than scientifically
characterized.

### 5.4 Optimizer-precision coupling

The Muon-Moonlight result `[muon-moonlight2025 §2.2;
kb/excerpts/muon-moonlight2025#sec-2-2-wd]` exposes that optimizer
choice and precision choice are entangled. AdamW + BF16 has a
well-understood envelope; Muon + BF16 needs additional weight decay to
stay in range; future optimizers (Schedule-Free, Sophia) may have their
own precision interactions. No general theory exists; each (optimizer,
precision) pairing is engineered separately.

### 5.5 Verifying training stability scientifically

`[FORUM-SIGNAL]` Most "no spikes" claims in tech reports are
unverified by external observers. DeepSeek-V3's claim is well-
documented (loss curves are plotted), but PaLM's and OPT's were partly
post-hoc. As precision drops, the bar for *verifying* stability
(versus just claiming it) rises — small frequent spikes that don't
produce divergence may still affect downstream quality. No standard
metric for "spike severity" exists.

### 5.6 FP4 inference vs FP8 training: the precision asymmetry

`[scaling-laws-precision-2024]` predicts inference can be lower-bit
than training without losing quality. Empirically, GPTQ
`[frantar2022; kb/excerpts/frantar2022#abstract]` and AWQ run
inference at INT4 from BF16-trained models with minimal degradation.
Training in INT4 / FP4 is a much harder problem because gradients,
not just weights, must be representable. The asymmetry between
training and inference precision is a permanent feature of the
landscape, not an artifact of immature tooling.

## 6. See also

- `kb/notes/training/optimization.md` — Muon's precision interaction;
  AdamW conventions for $(\beta_1, \beta_2, \lambda)$ in mixed precision.
- `kb/notes/training/distributed-training.md` — TorchTitan's 4D
  parallelism + Float8; DeepSeek-V3's 16-way PP + 64-way EP +
  ZeRO-1 DP.
- `kb/notes/inference/quantization.md` — GPTQ / AWQ / SmoothQuant /
  BitNet for *inference* quantization of *trained* models.
- `kb/notes/scaling/scaling-frontier.md` — the precision-scaling-law
  paper situates training precision as a fourth axis alongside
  $N$, $D$, $C$.
