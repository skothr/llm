---
paper_key: ma2024-bitnet
title: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
authors: Ma, Wang, Ma, Wang, Wang, Huang, Dong, Wang, Xue, Wei
year: 2024
venue: arXiv (Microsoft Research)
arxiv: 2402.17764
local_pdf: theory/sources/papers/ma2024_bitnet.pdf
type: excerpts
note: Verbatim from the v1 arXiv PDF (Feb 2024). BitNet b1.58 uses ternary {-1, 0, +1} weights with 8-bit activations (W1.58A8); the 1.58 bits comes from log2(3) ≈ 1.585. The follow-up "BitNet b1.58 2B4T Technical Report" (2504.12285) is the open-source 2B-scale checkpoint.
---

# Excerpts — Ma et al. 2024, "BitNet b1.58"

## Abstract — ternary weights match FP16 {#abstract}

> Recent research, such as BitNet [WMD+23], is paving the way for a new
> era of 1-bit Large Language Models (LLMs). In this work, we introduce
> a 1-bit LLM variant, namely BitNet b1.58, in which every single
> parameter (or weight) of the LLM is ternary $\{-1, 0, 1\}$. It
> matches the full-precision (i.e., FP16 or BF16) Transformer LLM with
> the same model size and training tokens in terms of both perplexity
> and end-task performance, while being significantly more cost-
> effective in terms of latency, memory, throughput, and energy
> consumption. More profoundly, the 1.58-bit LLM defines a new scaling
> law and recipe for training new generations of LLMs that are both
> high-performance and cost-effective. Furthermore, it enables a new
> computation paradigm and opens the door for designing specific
> hardware optimized for 1-bit LLMs.

## §1 — why ternary changes the multiply {#sec-1}

> Vanilla LLMs are in 16-bit floating values (i.e., FP16 or BF16), and
> the bulk of any LLMs is matrix multiplication. Therefore, the major
> computation cost comes from the floating-point addition and
> multiplication operations. In contrast, the matrix multiplication of
> BitNet only involves integer addition, which saves orders of energy
> cost for LLMs. As the fundamental limit to compute performance in
> many chips is power, the energy savings can also be translated into
> faster computation.

## §2 BitNet b1.58 — quantization function Eq.(1)–(3) {#sec-2}

> BitNet b1.58 is based on the BitNet architecture, which is a
> Transformer that replaces nn.Linear with *BitLinear*. It is trained
> from scratch, with 1.58-bit weights and 8-bit activations.

> **Quantization Function.** To constrain the weights to $-1$, $0$, or
> $+1$, we adopt an *absmean* quantization function. It first scales
> the weight matrix by its average absolute value, and then round each
> value to the nearest integer among $\{-1, 0, +1\}$:
>
> $$\widetilde{W} = \operatorname{RoundClip}\!\left( \frac{W}{\gamma + \epsilon}, -1, 1 \right), \tag{1}$$
> $$\operatorname{RoundClip}(x, a, b) = \max(a, \min(b, \operatorname{round}(x))), \tag{2}$$
> $$\gamma = \frac{1}{nm} \sum_{ij} |W_{ij}|. \tag{3}$$
>
> The quantization function for activations follows the same
> implementation in BitNet, except that we do not scale the activations
> before the non-linear functions to the range $[0, Q_b]$. Instead,
> the activations are all scaled to $[-Q_b, Q_b]$ per token to get rid
> of the zero-point quantization.

## §2 LLaMA-alike components {#sec-2-llama}

> The architecture of LLaMA has been the de-facto backbone for
> open-source LLMs. To embrace the open-source community, our design of
> BitNet b1.58 adopts the LLaMA-alike components. Specifically, it uses
> RMSNorm, SwiGLU, rotary embedding, and removes all biases. In this
> way, BitNet b1.58 can be integrated into the popular open-source
> software (e.g., Huggingface, vLLM, and llama.cpp) with minimal
> efforts.

## §3 Results — Table 1 perplexity vs cost {#sec-3-table-1}

> [Table 1, BitNet b1.58 vs LLaMA LLM matched-size:]
>
> | Models       | Size   | Memory (GB) | Latency (ms) | PPL    |
> |--------------|--------|-------------|--------------|--------|
> | LLaMA LLM    | 700M   | 2.08 (1.00x) | 1.18 (1.00x) | 12.33  |
> | BitNet b1.58 | 700M   | 0.80 (2.60x) | 0.96 (1.23x) | 12.87  |
> | LLaMA LLM    | 1.3B   | 3.34 (1.00x) | 1.62 (1.00x) | 11.25  |
> | BitNet b1.58 | 1.3B   | 1.14 (2.93x) | 0.97 (1.67x) | 11.29  |
> | LLaMA LLM    | 3B     | 7.89 (1.00x) | 5.07 (1.00x) | 10.04  |
> | BitNet b1.58 | 3B     | 2.22 (3.55x) | 1.87 (2.71x) | 9.91   |
> | BitNet b1.58 | 3.9B   | 2.38 (3.32x) | 2.11 (2.40x) | 9.62   |

> BitNet b1.58 starts to match full precision LLaMA LLM at 3B model
> size in terms of perplexity, while being 2.71 times faster and using
> 3.55 times less GPU memory. In particular, BitNet b1.58 with a 3.9B
> model size is 2.4 times faster, consumes 3.32 times less memory, but
> performs significantly better than LLaMA LLM 3B.

## §3 Throughput at 70B (Table 3) {#sec-3-table-3}

> | Models       | Size | Max Batch Size | Throughput (tokens/s) |
> |--------------|------|----------------|------------------------|
> | LLaMA LLM    | 70B  | 16 (1.0x)      | 333 (1.0x)             |
> | BitNet b1.58 | 70B  | 176 (11.0x)    | 2977 (8.9x)            |

## §3 — energy {#sec-3-energy}

> According to the energy model in [Hor14, ZZL22], BitNet b1.58 saves
> 71.4 times arithmetic operations energy consumption for matrix
> multiplication on 7nm chips. … Our results show that as the model
> size scales, BitNet b1.58 becomes increasingly more efficient in
> terms of energy consumption compared to the FP16 LLaMA LLM baseline.

## §3 — new scaling-law equivalence {#sec-3-scaling}

> BitNet b1.58 is enabling a new scaling law with respect to model
> performance and inference cost. As a reference, we can have the
> following equivalence between different model sizes in 1.58-bit and
> 16-bit based on the results in Figure 2 and 3.
>
> - 13B BitNet b1.58 is more efficient, in terms of latency, memory
>   usage and energy consumption, than 3B FP16 LLM.
> - 30B BitNet b1.58 is more efficient … than 7B FP16 LLM.
> - 70B BitNet b1.58 is more efficient … than 13B FP16 LLM.
