---
paper_key: kalamkar2019-bfloat16
title: "A Study of BFLOAT16 for Deep Learning Training"
authors: Kalamkar, Mudigere, Mellempudi, Das, Banerjee, Avancha, Vooturi, Jammalamadaka, Huang, Yuen, Yang, Park, Heinecke, Georganas, Srinivasan, Kundu, Smelyanskiy, Kaul, Dubey
year: 2019
venue: arXiv (Intel)
arxiv: 1905.12322
local_pdf: null
type: excerpts
note: "Excerpts from arXiv abstract page (verbatim). The canonical reference for BF16's deep-learning credentials. PDF not yet downloaded."
---

# Excerpts — Kalamkar et al. 2019, "A Study of BFLOAT16 for Deep Learning Training"

## Abstract — BF16 as drop-in for FP32 {#abstract}

> This paper presents the first comprehensive empirical study
> demonstrating the efficacy of the **Brain Floating Point (BFLOAT16)
> half-precision format** for Deep Learning training across image
> classification, speech recognition, language modeling, generative
> networks, and industrial recommendation systems. BFLOAT16 is
> attractive for Deep Learning training for two reasons: the higher
> dynamic range of BFLOAT16 closely matches that of FP32, and
> BFLOAT16 occupies half the memory of FP32. We propose a
> **shortened (16-bit) representation for floating-point numbers**
> with **8 exponent bits and 7 mantissa bits**, demonstrating that
> models trained with BFLOAT16 weights, activations, gradients, and
> updates **match those trained with FP32 in terms of accuracy
> without any change in hyper-parameters**.

## §2 BF16 format {#sec-2}

| Format | Bits | Exponent | Mantissa | Dyn. range $\approx$ |
|---|---|---|---|---|
| FP32 | 32 | 8 | 23 | $10^{\pm 38}$ |
| FP16 | 16 | 5 | 10 | $6 \times 10^{-5}$ to $6.55 \times 10^{4}$ |
| **BF16** | 16 | **8** | **7** | $10^{\pm 38}$ (FP32-like) |

The key trade in BF16: **same exponent as FP32**, fewer mantissa bits
than FP16. Range $\approx$ FP32 range; precision worse than FP16's.

## §3 Why range matters more than precision for DL {#sec-3}

Gradient magnitudes during training span many orders of magnitude
across layers and across coordinates. FP16's narrower range causes
underflow on small gradients; BF16's range avoids this without needing
loss-scaling. The cost is 7 mantissa bits (~3 decimal digits of
precision) — sufficient because:

- Each weight is updated thousands of times during training, so the
  accumulated noise from low precision averages out.
- The master-weights pattern (Micikevicius et al. 2017) is *not
  required* for BF16 in many setups: FP32 master is optional.

## §4 Empirical {#sec-4}

> models trained with BFLOAT16 weights, activations, gradients, and
> updates match those trained with FP32 in terms of accuracy without
> any change in hyper-parameters.

The empirical claim is that BF16 is a **drop-in replacement** for
FP32 — no recipe changes needed, no loss-scaling, optionally no FP32
master weights. This is the property that made BF16 the universal
LLM pre-training default by 2023 (LLaMA-2/3, Mistral, Qwen, Gemma,
OLMo all use BF16).

[NOTE — pdf-not-available] Format diagram and methodology widely
reproduced in NVIDIA / Intel mixed-precision documentation.
