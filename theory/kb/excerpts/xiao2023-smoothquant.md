---
paper_key: xiao2023-smoothquant
title: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
authors: Xiao, Lin, Seznec, Wu, Demouth, Han
year: 2023
venue: ICML
arxiv: 2211.10438
local_pdf: theory/sources/papers/xiao2023_smoothquant.pdf
type: excerpts
note: Verbatim from the v7 arXiv PDF (Mar 2024). SmoothQuant is the canonical solution to the W8A8 (8-bit weight + 8-bit activation) quantization problem caused by activation outliers. The migration trick — scale activations down by per-channel s, scale weights up by 1/s — became the template for AWQ-style activation-aware methods.
---

# Excerpts — Xiao et al. 2023, "SmoothQuant"

## Abstract — W8A8 made accurate and hardware-friendly {#abstract}

> Quantization can reduce memory and accelerate inference. However,
> existing methods cannot maintain accuracy and hardware efficiency at
> the same time. We propose SmoothQuant, a training-free, accuracy-
> preserving, and general-purpose post-training quantization (PTQ)
> solution to enable 8-bit weight, 8-bit activation (W8A8) quantization
> for LLMs. Based on the fact that weights are easy to quantize while
> activations are not, SmoothQuant smooths the activation outliers by
> offline *migrating* the quantization difficulty from activations to
> weights with a mathematically equivalent transformation. SmoothQuant
> enables an INT8 quantization of *both* weights and activations for
> all the matrix multiplications in LLMs, including OPT, BLOOM, GLM,
> MT-NLG, Llama-1/2, Falcon, Mistral, and Mixtral models. We
> demonstrate up to 1.56× speedup and 2× memory reduction for LLMs
> with negligible loss in accuracy. SmoothQuant enables serving 530B
> LLM within a single node.

## §2 Preliminaries — uniform quantization Eq.(1) {#sec-2}

> Quantization maps a high-precision value into discrete levels. We
> study integer uniform quantization (Jacob et al., 2018) (specifically
> INT8) for better hardware support and efficiency. The quantization
> process can be expressed as:
>
> $$\bar{\mathbf{X}}^{\text{INT8}} = \left\lceil \frac{\mathbf{X}^{\text{FP16}}}{\Delta} \right\rfloor, \quad \Delta = \frac{\max(|\mathbf{X}|)}{2^{N-1} - 1}, \tag{1}$$
>
> where $\mathbf{X}$ is the floating-point tensor, $\bar{\mathbf{X}}$
> is the quantized counterpart, $\Delta$ is the quantization step size,
> $\lceil \cdot \rfloor$ is the rounding function, and $N$ is the
> number of bits (8 in our case).

## §3 Review of Quantization Difficulty — three observations {#sec-3}

> 1. Activations are harder to quantize than weights. The weight
>    distribution is quite uniform and flat, which is easy to quantize.
>    Previous work has shown that quantizing the weights of LLMs with
>    INT8 or even INT4 does not degrade accuracy.
>
> 2. Outliers make activation quantization difficult. The scale of
>    outliers in activations is ~100× larger than most of the activation
>    values. In the case of per-tensor quantization, the large outliers
>    dominate the maximum magnitude measurement, leading to low
>    *effective quantization bits/levels*: suppose the maximum
>    magnitude of channel $i$ is $m_i$, and the maximum value of the
>    whole matrix is $m$, the effective quantization levels of channel
>    $i$ is $2^8 \cdot m_i / m$. For non-outlier channels, the effective
>    quantization levels would be very small (2-3), leading to large
>    quantization errors.
>
> 3. Outliers persist in fixed channels. Outliers appear in a small
>    fraction of the *channels*. If one channel has an outlier, it
>    persistently appears in all tokens. The variance amongst the
>    channels for a given token is large … but the variance between
>    the magnitudes of a given channel across tokens is small (outlier
>    channels are consistently large).

## §4 SmoothQuant — migration formula Eq.(3),(4) {#sec-4}

> Instead of per-channel activation quantization (which is infeasible),
> we propose to "smooth" the input activation by dividing it by a
> per-channel smoothing factor $\mathbf{s} \in \mathbb{R}^{C_i}$. To
> keep the mathematical equivalence of a linear layer, we scale the
> weights accordingly in the reversed direction:
>
> $$\mathbf{Y} = (\mathbf{X} \operatorname{diag}(\mathbf{s})^{-1}) \cdot (\operatorname{diag}(\mathbf{s}) \mathbf{W}) = \hat{\mathbf{X}} \hat{\mathbf{W}}. \tag{3}$$
>
> Considering input $\mathbf{X}$ is usually produced from previous
> linear operations (e.g., linear layers, layer norms, etc.), we can
> easily fuse the smoothing factor into previous layers' parameters
> *offline*, which does not incur kernel call overhead from an extra
> scaling.

> Migrate the quantization difficulty from activations to weights. We
> aim to choose a per-channel smoothing factor $\mathbf{s}$ such that
> $\hat{\mathbf{X}} = \mathbf{X} \operatorname{diag}(\mathbf{s})^{-1}$
> is easy to quantize. … Here we introduce a hyper-parameter,
> migration strength $\alpha$, to control how much we want to migrate
> from activations to weights, using the following equation:
>
> $$\mathbf{s}_j = \max(|\mathbf{X}_j|)^{\alpha} / \max(|\mathbf{W}_j|)^{1-\alpha}. \tag{4}$$
>
> We find that for most of the models, e.g., all OPT and BLOOM models,
> $\alpha = 0.5$ is a well-balanced point to evenly split the
> quantization difficulty, especially when we are using the same
> quantizer for weights and activations (e.g., per-tensor, static
> quantization). … For some other models where activation outliers
> are more significant (e.g., GLM-130B has ~30% outliers), we can
> choose a larger $\alpha$ to migrate more quantization difficulty to
> weights (like 0.75).

## §5.2 Empirical accuracy headline {#sec-5-2}

> [Table 1, average accuracy on WinoGrande/HellaSwag/PIQA/LAMBADA, OPT
> family:]
>
> | Method              | OPT-6.7B | OPT-13B | OPT-30B | OPT-66B | OPT-175B |
> |---------------------|----------|---------|---------|---------|----------|
> | FP16                | 64.9%    | 65.6%   | 67.9%   | 69.5%   | 71.6%    |
> | INT8 per-tensor     | 39.9%    | 33.0%   | 32.8%   | 33.1%   | 32.3%    |
> | INT8 per-token      | 42.5%    | 33.0%   | 33.1%   | 32.9%   | 31.7%    |
> | INT8 per-channel*   | 64.8%    | 65.6%   | 68.0%   | 69.4%   | 71.4%    |
> [*per-channel for activations is *infeasible* on hardware GEMM kernels.]

## §1 Introduction headline {#sec-1-headline}

> Remarkably, SmoothQuant allows serving large models like OPT-175B
> using only half number of GPUs compared to FP16 while being faster,
> and enabling the serving of a 530B model within one 8-GPU node.
