---
paper_key: dettmers2023-qlora
title: "QLoRA: Efficient Finetuning of Quantized LLMs"
authors: Dettmers, Pagnoni, Holtzman, Zettlemoyer
year: 2023
venue: NeurIPS 2023
arxiv: 2305.14314
local_pdf: null
type: excerpts
note: "Excerpts from arXiv abstract page (verbatim) plus methodology summary widely reproduced in PEFT and quantization surveys. PDF not yet downloaded."
---

# Excerpts — Dettmers et al. 2023, "QLoRA"

## Abstract — 65B fine-tune on 48GB GPU {#abstract}

> We present QLoRA, an efficient finetuning approach that reduces
> memory usage enough to **finetune a 65B parameter model on a single
> 48GB GPU** while preserving full 16-bit finetuning task performance.
> QLoRA backpropagates gradients through a frozen, 4-bit quantized
> pretrained language model into Low Rank Adapters (LoRA). Our best
> model family, which we name Guanaco, outperforms all previous openly
> released models on the Vicuna benchmark, reaching 99.3% of the
> performance level of ChatGPT while only requiring 24 hours of
> finetuning on a single GPU. QLoRA introduces a number of innovations
> to save memory without sacrificing performance: (a) **4-bit
> NormalFloat (NF4)**, a new data type that is information theoretically
> optimal for normally distributed weights (b) **double quantization**
> to reduce the average memory footprint by quantizing the quantization
> constants, and (c) **paged optimizers** to manage memory spikes.

## §3 NF4: 4-bit NormalFloat quantization {#sec-3-nf4}

The NormalFloat-4 (NF4) format is a quantile-spaced 4-bit
quantization, with 16 bin centers chosen as the quantiles of a
standard normal distribution. Pretrained weights are empirically
near-Gaussian per block, so NF4's bins are information-theoretically
near-optimal for weight quantization in this regime.

Per-block (block size 64) quantization scales $Z$ are stored as FP32.
The forward pass dequantizes per-block:
$$
W_{\text{BF16}} = Z \cdot \mathrm{NF4\_decode}(W_{\text{NF4}})
$$
right before the GEMM, then runs the GEMM in BF16. Plus the LoRA path:
$$
y = W_{\text{BF16}} x + (\alpha/r) B A x
$$
where $A, B$ are BF16 trainable.

## §3 Double quantization {#sec-3-doublequant}

The per-block FP32 scales themselves are quantized to FP8 (8-bit) with
a second-level scale. Average overhead per parameter drops from
~0.5 bits (single-level FP32 scale per 64-element block) to ~0.37 bits.
At 65B parameters this is meaningful.

## §3 Paged optimizers {#sec-3-paged}

Optimizer state (Adam $m, v$) for the LoRA factors is allocated in
NVIDIA unified-memory pages that automatically swap to CPU RAM under
GPU pressure. This eliminates OOM failures during peak-activation
spikes (e.g., long-sequence batches).

## §4 Empirical {#sec-4}

> We use QLoRA to finetune more than 1,000 models, providing a
> detailed analysis of instruction following and chatbot performance
> across 8 instruction datasets, multiple model types (LLaMA, T5),
> and model scales that would be infeasible to run with regular
> finetuning (e.g. 33B and 65B parameter models).

> Our results show that QLoRA finetuning on a small high-quality
> dataset leads to state-of-the-art results, even when using smaller
> models than the previous SoTA.

[NOTE — pdf-not-available] Section numbers approximate; equations are
the canonical form reproduced in PEFT and quantization surveys.
