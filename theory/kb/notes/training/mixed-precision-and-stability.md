---
topic_key: mixed-precision-and-stability
area: training
status: stub
last_updated: 2026-05-04
primary_sources: [deepseek-v3]
secondary_sources: [torchtitan2024, muon-moonlight2025]
related_topics: [distributed-training, optimization]
---

# Mixed precision & training stability

> **Status: STUB.** Skeleton with anchored citations; needs full
> Feynman-bar treatment. Promote to DRAFT once a unified note covering
> (BF16 baseline, FP8 fine-grained quantization, NVFP4, scaling-laws of
> precision, loss-spike taxonomy) is written.

## Scope

The collection of techniques that let a forward/backward pass use lower
numerical precision than the master weights without observable
divergence. Three layers:

1. **Storage precision** — what number format the master weights,
   gradients, and optimizer state are stored in.
2. **Compute precision** — what format the matrix-multiply (Tensor Core
   GEMM) inputs and accumulators use.
3. **Stability machinery** — loss-scaling, scale-block fine-graining,
   accumulation-precision promotion, outlier handling.

## Open notes for the full draft

- BF16 (the 2020-era baseline) — keep as background.
- **FP8 fine-grained quantization, DeepSeek-V3 recipe.** Activations
  scaled per (1×128 tile = per token per 128 channels); weights scaled
  per (128×128 block); gradients also FP8 in `Dgrad`/`Wgrad`. E4M3 on
  *all* tensors (vs. NVIDIA's hybrid E4M3 / E5M2 default). FP32
  promotion to CUDA cores at $N_C = 128$ accumulation interval to
  recover precision lost by the 14-bit Tensor-Core accumulator. BF16
  optimizer moments; FP32 master weights and gradients.
  Result: ≤ 0.25 % relative loss vs. BF16 baseline at scale
  `[deepseek-v3 §3.3, §3.3.1, §3.3.2, §3.3.3;
  kb/excerpts/deepseek-v3-training#sec-3-3-fp8,
  kb/excerpts/deepseek-v3-training#sec-3-3-1-mixed,
  kb/excerpts/deepseek-v3-training#sec-3-3-2-finegrained,
  kb/excerpts/deepseek-v3-training#sec-3-3-2-accum,
  kb/excerpts/deepseek-v3-training#sec-3-3-2-mantissa,
  kb/excerpts/deepseek-v3-training#sec-3-3-3-storage]`.
- **TorchTitan Float8 path.** `torchao.float8` per-tensor scaling
  (dynamic / delayed / static) selectively on `Linear` layers; less
  aggressive than DeepSeek-V3's tile/block scheme but easier to apply
  generically `[torchtitan2024 §2.2.4;
  kb/excerpts/torchtitan2024#sec-2-2-4-fp8]`.
- **NVFP4** (2025) — hardware-supported 4-bit float on Blackwell.
  Pending excerpt; will not back hard claims until added.
- **Scaling laws for precision** (Kumar, Pearce, Huang et al. 2024) —
  predict effective parameter count as a function of training-bit-width
  and inference-bit-width. Pending excerpt.
- **Stable training in practice.** DeepSeek-V3's full pre-training
  (14.8 T tokens, 2 months, 2048 H800s) reports zero irrecoverable loss
  spikes, no rollbacks `[deepseek-v3 §abstract;
  kb/excerpts/deepseek-v3-training#abstract]`. Causes-of-loss-spikes
  taxonomy (gradient norm explosion, attention-logit divergence, embedding
  norm growth, fp8 underflow on rare tokens) needs a primary citation;
  PaLM, OPT, and Llama-2 papers each cover subsets.

## Cross-references

- [optimization](optimization.md) — Muon's per-shape RMS normalization is
  partly a stability-against-low-precision mechanism
  `[muon-moonlight2025 §2.2;
  kb/excerpts/muon-moonlight2025#sec-2-2-rms]`.
- [distributed-training](distributed-training.md) — FP8 plus 5D
  parallelism is the bleeding edge in TorchTitan and DeepSeek-V3.
