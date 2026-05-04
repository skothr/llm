---
paper_key: frantar2022
title: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
authors: Frantar, Ashkboos, Hoefler, Alistarh
year: 2022
venue: ICLR 2023
arxiv: 2210.17323
local_pdf: theory/sources/papers/frantar2022_gptq.pdf
type: excerpts
note: Verbatim from the v2 arXiv PDF (Mar 2023, ICLR 2023 publication). GPTQ is the canonical accurate one-shot weight-only PTQ method for LLMs at the 100B+ scale; downstream of OBQ (Optimal Brain Quantization) and the Optimal Brain Surgeon line. Three insights drive the practical scaling: arbitrary-order quantization, lazy batched updates, and Cholesky reformulation of the Hessian inverse.
---

# Excerpts — Frantar et al. 2022, "GPTQ"

## Abstract — what GPTQ delivers {#abstract}

> In this paper, we address this challenge, and propose GPTQ, a new
> one-shot weight quantization method based on approximate second-order
> information, that is both highly-accurate and highly-efficient.
> Specifically, GPTQ can quantize GPT models with 175 billion parameters
> in approximately four GPU hours, reducing the bitwidth down to 3 or 4
> bits per weight, with negligible accuracy degradation relative to the
> uncompressed baseline. Our method more than doubles the compression
> gains relative to previously-proposed one-shot quantization methods,
> preserving accuracy, allowing us for the first time to execute an 175
> billion-parameter model inside a single GPU for generative inference.
> … We show experimentally that these improvements can be leveraged for
> end-to-end inference speedups over FP16, of around 3.25x when using
> high-end GPUs (NVIDIA A100) and 4.5x when using more cost-effective
> ones (NVIDIA A6000).

## §3 Background — Layer-Wise Quantization Eq.(1) {#sec-3-layerwise}

> At a high level, our method follows the structure of state-of-the-art
> post-training quantization methods, by performing quantization
> layer-by-layer, solving a corresponding reconstruction problem for
> each layer. Concretely, let $\mathbf{W}_\ell$ be the weights
> corresponding to a linear layer $\ell$ and let $\mathbf{X}_\ell$
> denote the layer input corresponding to a small set of $m$ data points
> running through the network. Then, the objective is to find a matrix
> of quantized weights $\widehat{\mathbf{W}}$ which minimizes the
> squared error, relative to the full precision layer output. Formally,
> this can be restated as
>
> $$\operatorname*{argmin}_{\widehat{\mathbf{W}}} \, \|\mathbf{W} \mathbf{X} - \widehat{\mathbf{W}} \mathbf{X}\|_2^2. \tag{1}$$
>
> Further … we assume that the quantization grid for $\widehat{W}$ is
> fixed before the process, and that individual weights can move freely
> as in (Hubara et al., 2021; Frantar et al., 2022).

## §3 Background — OBQ Eq.(2),(3) — the baseline GPTQ accelerates {#sec-3-obq}

> The OBQ method starts from the observation that Equation (1) can be
> written as the sum of the squared errors, over each row of $\mathbf{W}$.
> Then, OBQ handles each row $\mathbf{w}$ independently, quantizing one
> weight at a time while always updating all not-yet-quantized weights,
> in order to compensate for the error incurred by quantizing a single
> weight. Since the corresponding objective is a quadratic, whose
> Hessian is $\mathbf{H}_F = 2 \mathbf{X}_F \mathbf{X}_F^\top$, where
> $F$ denotes the set of remaining full-precision weights, the
> greedy-optimal weight to quantize next, which we denote by $w_q$, and
> the corresponding optimal update of all weights in $F$, denoted by
> $\boldsymbol{\delta}_F$, are given by the following formulas, where
> $\operatorname{quant}(w)$ rounds $w$ to the nearest value on the
> quantization grid:
>
> $$w_q = \operatorname{argmin}_{w_q} \frac{(\operatorname{quant}(w_q) - w_q)^2}{[\mathbf{H}_F^{-1}]_{qq}}, \quad \boldsymbol{\delta}_F = -\frac{w_q - \operatorname{quant}(w_q)}{[\mathbf{H}_F^{-1}]_{qq}} \cdot (\mathbf{H}_F^{-1})_{:,q}. \tag{2}$$
>
> OBQ quantizes weights iteratively using these two equations, until all
> the weights of $\mathbf{w}$ are quantized. This is done efficiently,
> avoiding expensive full recomputations of $\mathbf{H}^{-1}$, by
> removing the $q$th row and column of $\mathbf{H}$, which is necessary
> after quantizing $w_q$, directly in the inverse via one step of
> Gaussian elimination. Namely, the updated inverse is given by the
> formula
>
> $$\mathbf{H}_{-q}^{-1} = \left( \mathbf{H}^{-1} - \frac{1}{[\mathbf{H}^{-1}]_{qq}} \mathbf{H}_{:,q}^{-1} \mathbf{H}_{q,:}^{-1} \right)_{-p}. \tag{3}$$
>
> However, the fact that OBQ's runtime for a $d_{\text{row}} \times
> d_{\text{col}}$ matrix $\mathbf{W}$ has cubic input dependency
> $O(d_{\text{row}} \cdot d_{\text{col}}^3)$ means that applying it to
> models with billions of parameters is extremely expensive.

## §4 GPTQ Algorithm — three modifications {#sec-4-algorithm}

### Step 1: Arbitrary Order Insight {#sec-4-step-1}

> As explained in the previous section, OBQ quantizes weights in greedy
> order, i.e. it always picks the weight which currently incurs the
> least additional quantization error. Interestingly, we find that,
> while this quite natural strategy does indeed seem to perform very
> well, its improvement over quantizing the weights in arbitrary order
> is generally small, in particular on large, heavily-parametrized
> layers. Most likely, this is because the slightly lower number of
> quantized weights with large individual error is balanced out by those
> weights being quantized towards the end of the process, when only few
> other unquantized weights that can be adjusted for compensation
> remain.
>
> The original OBQ method quantizes rows of $\mathbf{W}$ independently,
> in a specific order defined by the corresponding errors. By contrast,
> we will aim to quantize the weights of all rows in the same order, and
> will show that this typically yields results with a final squared
> error that is similar to the original solutions. As a consequence, the
> set of unquantized weights $F$ and similarly $\mathbf{H}_F^{-1}$ is
> always the same for all rows. … This reduces the overall runtime from
> $O(d_{\text{row}} \cdot d_{\text{col}}^3)$ to $O(\max\{d_{\text{row}}
> \cdot d_{\text{col}}^2, d_{\text{col}}^3\})$, i.e., by a factor of
> $\min\{d_{\text{row}}, d_{\text{col}}\}$.

### Step 2: Lazy Batch-Updates Eq.(4),(5) {#sec-4-step-2}

> First, a direct implementation of the scheme described previously
> will not be fast in practice, because the algorithm has a relatively
> low compute-to-memory-access ratio. … Fortunately, this problem can
> be resolved by the following observation: The final rounding decisions
> for column $i$ are only affected by updates performed on this very
> column, and so updates to later columns are irrelevant at this point
> in the process. This makes it possible to "lazily batch" updates
> together, thus achieving much better GPU utilization. Concretely, we
> apply the algorithm to $B = 128$ columns at a time, keeping updates
> contained to those columns and the corresponding $B \times B$ block of
> $\mathbf{H}^{-1}$. Only once a block has been fully processed, we
> perform global updates of the entire $\mathbf{H}^{-1}$ and $\mathbf{W}$
> matrices using the multi-weight versions of Equations (2) and (3)
> given below, with $Q$ denoting a set of indices, and
> $\mathbf{H}_{-Q}^{-1}$ denoting the inverse matrix with the
> corresponding rows and columns removed:
>
> $$\boldsymbol{\delta}_F = -(\mathbf{w}_Q - \operatorname{quant}(\mathbf{w}_Q))([\mathbf{H}_F^{-1}]_{QQ})^{-1} (\mathbf{H}_F^{-1})_{:,Q}, \tag{4}$$
> $$\mathbf{H}_{-Q}^{-1} = \left( \mathbf{H}^{-1} - \mathbf{H}_{:,Q}^{-1} ([\mathbf{H}^{-1}]_{QQ})^{-1} \mathbf{H}_{Q,:}^{-1} \right)_{-Q}. \tag{5}$$

### Step 3: Cholesky Reformulation {#sec-4-step-3}

> The final technical issue we have to address is given by numerical
> inaccuracies, which can become a major problem at the scale of
> existing models, especially when combined with the block updates
> discussed in the previous step. Specifically, it can occur that the
> matrix $\mathbf{H}_F^{-1}$ becomes indefinite, which we notice can
> cause the algorithm to aggressively update the remaining weights in
> incorrect directions, resulting in an arbitrarily-bad quantization of
> the corresponding layer. … For smaller models, applying dampening,
> that is adding a small constant $\lambda$ (we always choose 1% of the
> average diagonal value) to the diagonal elements of $\mathbf{H}$
> appears to be sufficient to avoid numerical issues. However, larger
> models require a more robust and general approach. … Indeed, the row
> removal via (3) for our symmetric $\mathbf{H}^{-1}$ essentially
> corresponds to taking a Cholesky decomposition, except for the minor
> difference that the latter divides row $q$ by
> $([\mathbf{H}_{F_q}^{-1}]_{qq})^{1/2}$.

## §4 Algorithm 1 — pseudocode {#sec-4-algorithm-1}

> Quantize $\mathbf{W}$ given inverse Hessian
> $\mathbf{H}^{-1} = (2\mathbf{X}\mathbf{X}^\top + \lambda \mathbf{I})^{-1}$
> and blocksize $B$.
>
> $\mathbf{Q} \leftarrow 0_{d_{\text{row}} \times d_{\text{col}}}$ // quantized output
> $\mathbf{E} \leftarrow 0_{d_{\text{row}} \times B}$ // block quantization errors
> $\mathbf{H}^{-1} \leftarrow \operatorname{Cholesky}(\mathbf{H}^{-1})^\top$
> for $i = 0, B, 2B, \ldots$ do
>   for $j = i, \ldots, i + B - 1$ do
>     $\mathbf{Q}_{:,j} \leftarrow \operatorname{quant}(\mathbf{W}_{:,j})$
>     $\mathbf{E}_{:,j-i} \leftarrow (\mathbf{W}_{:,j} - \mathbf{Q}_{:,j}) / [\mathbf{H}^{-1}]_{jj}$
>     $\mathbf{W}_{:,j:(i+B)} \leftarrow \mathbf{W}_{:,j:(i+B)} - \mathbf{E}_{:,j-i} \cdot \mathbf{H}^{-1}_{j,j:(i+B)}$
>   end for
>   $\mathbf{W}_{:,(i+B):} \leftarrow \mathbf{W}_{:,(i+B):} - \mathbf{E} \cdot \mathbf{H}^{-1}_{i:(i+B),(i+B):}$
> end for

## §1 Empirical headline (Figure 1) {#sec-1-figure-1}

> Figure 1: Quantizing OPT models to 4 and BLOOM models to 3 bit
> precision, comparing GPTQ with the FP16 baseline and round-to-nearest
> (RTN). [Plot shows: 4-bit RTN catastrophically degrades for OPT-66B
> and OPT-175B (e.g., perplexity 110); 4-bit GPTQ tracks FP16 closely
> across all sizes. 3-bit RTN BLOOM-176B reaches perplexity 571 vs. GPTQ
> ~12, FP16 ~10.]

> To our knowledge, we are the first to show that extremely accurate
> language models with hundreds of billions of parameters can be
> quantized to 3-4 bits/component: prior post-training methods only
> remain accurate at 8 bits, while prior training-based techniques have
> only tackled models that are smaller by one to two orders of
> magnitude.
