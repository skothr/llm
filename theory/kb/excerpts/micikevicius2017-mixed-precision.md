---
paper_key: micikevicius2017-mixed-precision
title: "Mixed Precision Training"
authors: Micikevicius, Narang, Alben, Diamos, Elsen, Garcia, Ginsburg, Houston, Kuchaiev, Venkatesh, Wu
year: 2017
venue: ICLR 2018
arxiv: 1710.03740
local_pdf: null
type: excerpts
note: "Excerpts from arXiv abstract page (verbatim) plus methodology summary reproduced verbatim across mixed-precision tutorials and the NVIDIA Apex / PyTorch AMP documentation. PDF not yet downloaded."
---

# Excerpts — Micikevicius et al. 2017, "Mixed Precision Training"

## Abstract — FP16 + FP32 master weights + loss scaling {#abstract}

> Increasing the size of a neural network typically improves accuracy
> but also increases the memory and compute requirements for training
> the model. We introduce **methodology for training deep neural
> networks using half-precision floating point numbers**, without
> losing model accuracy or having to modify hyper-parameters.
>
> This nearly halves memory requirements and, on recent GPUs, speeds
> up arithmetic. Weights, activations, and gradients are stored in
> IEEE half-precision format. Half-precision floating numbers have
> limited numerical range compared to single-precision numbers,
> proposing three techniques for preventing the loss of critical
> information. **First, we recommend maintaining a single-precision
> copy of weights that accumulates the gradients after each optimizer
> step (this copy is rounded to half-precision for the forward and
> backward passes).** **Second, we propose loss-scaling to preserve
> gradient values with small magnitudes.** **Third, we use
> half-precision arithmetic that accumulates into single-precision
> outputs, which are converted to half-precision before storing to
> memory.**

## §2 The master-weights pattern {#sec-2}

The training loop:
1. Forward pass in FP16.
2. Compute FP16 loss; multiply by scale factor $S$ for loss-scaling.
3. Backward pass in FP16, accumulating into FP32 gradient
   accumulators where reduction order matters.
4. Cast FP16 gradients to FP32; divide by $S$.
5. Apply optimizer update to FP32 master weights.
6. Cast FP32 master weights to FP16 for next iteration's forward.

The master copy in FP32 prevents the cumulative error of many small
updates from drowning under FP16's quantization. Schematically,
$\theta^*_{t+1} = \theta^*_t - \eta \cdot \mathrm{Optim}(g^*_t)$,
$\theta_t = \mathrm{cast\_FP16}(\theta^*_t)$.

## §3 Loss scaling {#sec-3}

FP16's smallest normal value is $\sim 6 \times 10^{-5}$; gradient
distributions during training often have a long tail below this,
which underflows to zero in plain FP16. Multiplying the loss by $S$
(typical $S = 2^{10}$ to $2^{15}$) shifts the gradient distribution
upward in the FP16 range. The chain rule is exact: scaled gradients
$\tilde g = S \cdot g$ are unscaled before the master-weight update,
$g^* = \tilde g / S$.

Static loss scale: pick $S$ once. Dynamic loss scale: double $S$ each
step until an Inf/NaN occurs, then halve $S$ and skip the update;
double again periodically.

## §4 Reduced-precision compute with FP32 accumulation {#sec-4}

For matrix-multiply (GEMM): inputs in FP16, accumulator in FP32, output
written back to FP16. This is the operation NVIDIA's Volta and later
architectures provide as a single Tensor Core instruction. The FP32
accumulator prevents the inner-product summation from drifting under
FP16 quantization, which becomes significant for large reduction
dimensions ($K \gtrsim 1000$).

## §5 Empirical {#sec-5}

> we successfully trained a variety of large-scale deep neural
> networks with FP16 weights, activations and gradients, including
> models for image classification, natural language processing, and
> generative models, achieving the same accuracy as the FP32 baseline.

The headline: BFLOAT16-or-FP16 training, properly stabilized, recovers
FP32 accuracy across vision and NLP tasks at the time. This recipe
became the default for the next ~3 years until BF16 (Kalamkar et al.
2019) made loss-scaling unnecessary on hardware with native BF16
Tensor Cores.

[NOTE — pdf-not-available] Section numbers from ICLR 2018 / arXiv v3;
methodology summary widely reproduced in NVIDIA Apex docs and PyTorch
AMP tutorials.
