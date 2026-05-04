---
topic: inference/quantization
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - frantar2022          # GPTQ
  - xiao2023-smoothquant # SmoothQuant
  - ma2024-bitnet        # BitNet b1.58
secondary_sources:
  - dettmers2022-llmint8  # LLM.int8 — outlier discovery (cited via primary text)
  - lin2023-awq          # AWQ — activation-aware weight scaling
  - bitnet-2b4t-2025     # BitNet b1.58 2B4T release
  - deepseek-v3          # FP8 training at 671B (cross-link to training)
related_topics:
  - inference/kv-cache-management
  - training/mixed-precision-and-stability
  - scaling/scaling-frontier
---

# Quantization

LLM quantization is **not one problem**. It is a family of decisions
indexed by *what* you quantize (weights only, activations, KV cache,
gradients), *when* (post-training vs. quantization-aware training vs.
native low-bit pre-training), and *to what precision* (8-bit, 4-bit,
2-bit, ternary, 1-bit). Each cell of the (what × when × precision)
grid has its own Pareto frontier; the canonical methods below each
own one cell.

This note covers **inference-side** quantization — methods applied to a
trained model to make serving cheaper. Training-side mixed precision
(FP8 / FP6 training, gradient quantization) is in
`kb/notes/training/mixed-precision-and-stability.md`.

## 1. The fundamental quantization map

Uniform integer quantization (the building block underneath all
methods below) is

$$\bar{X} = \left\lceil \frac{X}{\Delta} \right\rfloor, \quad \Delta = \frac{\max(|X|)}{2^{N-1} - 1}, \tag{1}$$

with $N$-bit signed integer levels and step size $\Delta$
`[xiao2023-smoothquant §2 Eq.1; kb/excerpts/xiao2023-smoothquant#sec-2]`.
The two design axes are:

- **Granularity.** Per-tensor (one $\Delta$ for the whole matrix) is
  cheapest at runtime but most lossy. Per-token / per-channel /
  per-group strikes a Pareto trade.
- **Symmetric vs. asymmetric.** Symmetric uses the formula above;
  asymmetric introduces a zero-point offset and is needed for
  positive-skewed distributions (post-ReLU, softmax outputs).

Per-channel of *activations* maps poorly onto INT8 GEMM kernels:
hardware tensor cores apply scaling factors only along outer dimensions
(token dimension $T$ or output channel $C_o$), not along the inner
contraction dimension $C_i$ `[xiao2023-smoothquant §3 / Fig.3;
kb/excerpts/xiao2023-smoothquant#sec-3]`. This hardware constraint is
the source of every clever scheme that follows: you can quantize
weights per-channel for free, but per-channel-of-activations is
infeasible without re-engineering the GEMM kernel.

## 2. The outlier problem

When LLMs cross ~6.7B parameters, **systematic activation outliers**
emerge: roughly $\sim 100\times$ larger than typical values, persistent
in fixed channels across tokens (Dettmers et al. 2022, "LLM.int8";
re-cited and quantified in
`[xiao2023-smoothquant §3; kb/excerpts/xiao2023-smoothquant#sec-3]`).
Per-tensor INT8 then catastrophically degrades quality (e.g.
OPT-175B accuracy 71.6% FP16 → 32.3% per-tensor INT8
`[xiao2023-smoothquant §5.2 Table 1; kb/excerpts/xiao2023-smoothquant#sec-5-2]`).

There are three high-level responses to outliers:

1. **Mixed precision (LLM.int8):** keep outlier channels in FP16,
   quantize the rest to INT8. Correct in principle, slow in practice
   (Dettmers et al., cited via SmoothQuant
   `[xiao2023-smoothquant §1 / Fig.1; kb/excerpts/xiao2023-smoothquant#abstract]`).
2. **Migration (SmoothQuant, AWQ):** move the outlier difficulty from
   activations to weights via a mathematically equivalent rescaling.
3. **Weight-only (GPTQ, AWQ):** quantize only the weights, leave
   activations FP16. This sidesteps activation outliers entirely; the
   payoff is reduced — memory bandwidth wins, but compute is unchanged
   for the quantized matmul.

## 3. Weight-only PTQ — GPTQ

**GPTQ** is the canonical accurate one-shot weight-only post-training
quantizer for LLMs at 100B+ scale. It solves a layer-wise
reconstruction problem `[frantar2022 §3 Eq.1;
kb/excerpts/frantar2022#sec-3-layerwise]`:

$$\operatorname*{argmin}_{\widehat{W}} \, \|W X - \widehat{W} X\|_2^2 \tag{2}$$

over a calibration set of $m$ samples (typically a few thousand
sequences from the pretraining distribution).

The baseline is **Optimal Brain Quantization (OBQ)** which quantizes
one weight at a time, using the Hessian inverse $\mathbf{H}^{-1} =
(2 \mathbf{X}_F \mathbf{X}_F^\top)^{-1}$ to compute both (a) the
optimal weight to quantize next and (b) the compensating updates to
all not-yet-quantized weights `[frantar2022 §3 Eq.2,3;
kb/excerpts/frantar2022#sec-3-obq]`. OBQ has cubic-in-$d_{\text{col}}$
runtime, infeasible at LLM scale.

GPTQ accelerates OBQ via three modifications
`[frantar2022 §4; kb/excerpts/frantar2022#sec-4-algorithm]`:

1. **Arbitrary order quantization** — quantize all rows of $W$ in the
   *same* fixed column order, instead of OBQ's per-row greedy order.
   Reduces from $O(d_r d_c^3)$ to $O(\max\{d_r d_c^2, d_c^3\})$
   `[frantar2022 §4 Step 1; kb/excerpts/frantar2022#sec-4-step-1]`. The
   loss in quality is empirically negligible for LLM-scale weight
   matrices.
2. **Lazy batched updates** — process columns in blocks of $B=128$,
   keep updates local to the block, do one global update after each
   block `[frantar2022 §4 Step 2 Eq.4,5;
   kb/excerpts/frantar2022#sec-4-step-2]`. This restores GPU
   compute-to-memory ratio.
3. **Cholesky reformulation** — replace the iterative Hessian inverse
   updates with a single Cholesky factorization plus dampening
   $\lambda \approx 1\%$ of the average diagonal
   `[frantar2022 §4 Step 3; kb/excerpts/frantar2022#sec-4-step-3]`.
   Solves numerical instabilities that surface at the 100B+ scale.

Headline empirics: OPT-175B 4-bit RTN diverges (perplexity 110); GPTQ
4-bit tracks FP16 (perplexity ~9). BLOOM-176B 3-bit RTN perplexity
571; GPTQ ~12 vs. FP16 ~10
`[frantar2022 §1 / Fig.1; kb/excerpts/frantar2022#sec-1-figure-1]`. End-
to-end inference speedup ~3.25× on A100, ~4.5× on A6000
`[frantar2022 abstract; kb/excerpts/frantar2022#abstract]`.

## 4. Activation-aware weight scaling — AWQ

**AWQ** `[lin2023-awq]` is the activation-aware sibling of GPTQ. Its
core observation is that **only ~1% of weight channels are "salient"**
— meaning their corresponding activation magnitudes are large — and
preserving those channels is what determines downstream quality
(`[lin2023-awq abstract]`, full abstract verified by arxiv abstract
fetch but verbatim section excerpts of the scaling formula are not in
this KB; treat the per-channel scaling formula as a tier-A claim
needing a future excerpt-fill). The mechanism: collect per-channel
activation statistics offline, scale up salient weight channels by a
per-channel factor $s_j$ (and scale down activations by $1/s_j$ on the
other side, mathematically equivalent to SmoothQuant migration but
optimized specifically for weight-only INT4).

[CONTRADICTION] As of 2025–2026, the field reports AWQ generally
outperforms GPTQ on instruction-tuned and multimodal LLMs at INT4 —
because AWQ is calibration-dataset-agnostic (uses simple max-magnitude
statistics, not per-sample Hessians) and so generalizes better
out-of-distribution `[FORUM-SIGNAL: vendor blog comparisons]`. GPTQ
still wins on tasks closely matching the calibration distribution. The
practitioner default in 2026 is AWQ for general deployment, GPTQ for
domain-specific compression.

## 5. W8A8 weight-and-activation — SmoothQuant

To quantize *both* weights and activations to INT8 (the "W8A8"
problem) without per-channel-activation infeasibility, SmoothQuant
applies the **migration trick**. The math is a free rescaling
`[xiao2023-smoothquant §4 Eq.3; kb/excerpts/xiao2023-smoothquant#sec-4]`:

$$Y = (X \operatorname{diag}(s)^{-1}) \cdot (\operatorname{diag}(s) W) = \hat{X} \hat{W}, \tag{3}$$

so the layer is mathematically unchanged. The per-channel smoothing
factor uses migration strength $\alpha$:

$$s_j = \frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}}, \quad \alpha = 0.5 \text{ default}, \tag{4}$$

`[xiao2023-smoothquant §4 Eq.4; kb/excerpts/xiao2023-smoothquant#sec-4]`.
With $\alpha=0$ all difficulty stays on activations (broken); with
$\alpha=1$ all difficulty migrates to weights (also breaks because
weights now have the outlier shape they previously lacked); $\alpha
\approx 0.5$ is the sweet spot. The smoothing factor is **fused
offline** into the preceding linear or LayerNorm, so the runtime cost
is zero.

Headline: OPT-175B from 32.3% (per-tensor INT8) → 71.4% (per-channel
SmoothQuant), matching the 71.6% FP16 baseline
`[xiao2023-smoothquant §5.2 Table 1; kb/excerpts/xiao2023-smoothquant#sec-5-2]`.
1.56× speedup, 2× memory at INT8.

[INTUITION] SmoothQuant moves the outlier "shape" from where it
breaks the GEMM (activations, where you can't per-channel-quantize) to
where it doesn't (weights, where you can). The mathematical equivalence
is the load-bearing claim — Eq. (3) is exactly the original linear
layer. There is no model change, no fine-tuning, no extra runtime cost.
The price is that weights become slightly harder to quantize than they
were originally, hence the $1-\alpha$ exponent in $s_j$ that splits
the difficulty.

## 6. Native low-bit pre-training — BitNet b1.58

BitNet b1.58 takes the dual position: instead of quantizing a trained
model after the fact, **train from scratch with ternary weights**.
Every weight is constrained to $\{-1, 0, +1\}$ (1.58 bits in
information-theoretic terms, since $\log_2 3 \approx 1.585$).

The quantization function is *absmean* with RoundClip
`[ma2024-bitnet §2 Eq.1–3; kb/excerpts/ma2024-bitnet#sec-2]`:

$$\widetilde{W} = \operatorname{RoundClip}\!\left( \frac{W}{\gamma + \epsilon}, -1, 1 \right), \tag{5}$$
$$\gamma = \frac{1}{nm} \sum_{ij} |W_{ij}|. \tag{6}$$

Activations remain 8-bit (this is **W1.58A8**, not pure 1-bit).

The architecture is otherwise LLaMA-alike — RMSNorm, SwiGLU, RoPE, no
biases — so it integrates into HuggingFace, vLLM, and llama.cpp with
minimal effort
`[ma2024-bitnet §2 LLaMA-alike; kb/excerpts/ma2024-bitnet#sec-2-llama]`.

Reported scaling crossover: BitNet b1.58 starts to match FP16 LLaMA
quality at the **3B** size. At 3B, BitNet is 2.71× faster, 3.55× less
memory, and slightly *better* perplexity (9.91 vs. 10.04)
`[ma2024-bitnet §3 Table 1; kb/excerpts/ma2024-bitnet#sec-3-table-1]`.
At 70B, throughput is 8.9× higher (2977 vs. 333 tokens/s) at 11×
larger max batch size
`[ma2024-bitnet §3 Table 3; kb/excerpts/ma2024-bitnet#sec-3-table-3]`.
The follow-up tech report (BitNet b1.58 2B4T,
`[bitnet-2b4t-2025]`) is the first **open-source native 1-bit checkpoint**
trained on 4T tokens.

[INTUITION] Multiplying by $\{-1, 0, +1\}$ is not a multiply — it is a
*sign flip + select*. So a BitLinear layer is dominated by INT8 *adds*,
not multiplies, on appropriate hardware. The 71× arithmetic-energy
saving on 7nm chips
`[ma2024-bitnet §3 / Fig.3; kb/excerpts/ma2024-bitnet#sec-3-energy]`
follows directly from this — it is a hardware claim about
addition vs. multiplication energy, not a model claim. The model claim
is that ternary suffices for LLM-quality pretraining at 3B+.

[CONTRADICTION] BitNet b1.58 has been validated up to 3.9B with the
2024 paper and 2B with the open release. **Whether ternary QAT
maintains parity at 70B trained from scratch is empirically open.**
Most published "1-bit at 70B" numbers come from running inference of a
70B BitNet *that was hypothetically trained that way* — but a fully
pretrained native 70B BitNet does not exist publicly as of 2026-05.
The 70B numbers in `[ma2024-bitnet §3 Table 3]` are throughput /
memory measurements, not quality measurements, on architectures
matched in shape.

## 7. The (what × precision) grid as of 2026-05

| What \ Precision | INT8 | INT4 | INT2 | ~1-bit |
|---|---|---|---|---|
| **Weights only** | trivial / RTN works | GPTQ, AWQ (production default) | extreme regime, GPTQ §1 | requires BitNet pretraining |
| **W + A** | SmoothQuant, ZeroQuant | GPTQ-W4A4 (research) | n/a | n/a |
| **KV cache** | trivial | KVQuant `[kvquant2024]`, KIVI | KVQuant 2-bit | n/a |
| **FP8 (training)** | DeepSeek-V3 671B | recent | n/a | n/a |
| **MX formats** | MXFP8, MXINT8 (OCP) | MXFP6, MXINT4 | n/a | n/a |

FP8 production training was first validated at trillion-parameter
scale by DeepSeek-V3
`[deepseek-v3]` with tile-based quantization (128×128 weight blocks,
1×128 activation vectors). H100 / H200 / Blackwell natively accelerate
FP8 GEMMs. Microscaling (MX) formats from the OCP standard add a
shared exponent per small block, splitting the difference between
per-tensor and per-element scales; supported in H100 (limited) and
Blackwell (full).

[FORUM-SIGNAL: 2026 community benchmarks] The practitioner default
stack is: **GGUF Q4_K_M / IQ4_XS for weights** (CPU and consumer GPU),
**AWQ-INT4 for high-throughput GPU serving**, **FP8 for frontier
training**. GPTQ is still preferred for domain-tuned compression;
SmoothQuant retains a niche where INT8 throughput on older hardware
(A100, H100 INT8 paths) outperforms FP16 by enough to matter.

## 8. Frontier and open questions (as of 2026-05)

- **Scaling Laws for Precision** `[scaling-laws-precision-2024]` argue
  that *post-training quantization has increasing cost as training
  extends* — a model trained on 1× tokens loses less quality from
  4-bit PTQ than the same model trained on 5× tokens. This inverts the
  naive "more training = better" prior in the over-trained regime
  (e.g. LLaMA 3 at ~1875 tokens/parameter
  `[meta-llama3]`). Cross-link to
  `kb/notes/scaling/scaling-frontier.md`.
- **Native low-bit at frontier scale.** BitNet b1.58 is validated to
  3.9B as of the 2024 paper; the field is watching for native-1-bit
  results at 70B+ from-scratch.
- **MX formats vs. FP8.** Both are OCP-blessed; MX uses block-shared
  exponents while FP8 uses per-tensor scales. Which wins is hardware-
  generation dependent and not yet stably benchmarked at LLM scale
  `[FORUM-SIGNAL: NVIDIA / AMD documentation]`.
- **AWQ for multimodal.** AWQ's calibration-agnosticism is widely
  reported to give it an edge on instruction-tuned and multimodal
  models, but a comprehensive benchmark across 2026 multimodal LLMs
  with strict protocol controls is missing.
- **Quantization-aware fine-tuning vs. PTQ.** A 2025 Pareto question:
  for a fixed deployment budget, is it cheaper to (a) train at FP16
  and PTQ to 4-bit, or (b) fine-tune QAT-style for 1B tokens? Open
  empirically.

## 9. See also

- `kb/notes/training/mixed-precision-and-stability.md` — FP8 training,
  μP, unit-scaling, stability fixes for low-precision training.
- `kb/notes/inference/kv-cache-management.md §5` — KVQuant and KIVI
  apply the outlier-channel intuition specifically to the KV tensor.
- `kb/notes/scaling/scaling-frontier.md` — scaling laws for
  precision; how PTQ cost grows with training-tokens budget.
- `kb/notes/inference/serving-systems.md` — how serving stacks
  consume quantized checkpoints (GGUF, AWQ, GPTQ formats; W4A16 vs
  W4A8 kernel paths).
