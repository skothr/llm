---
paper_key: rajamanoharan2024-jumprelu
title: "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders"
authors: Rajamanoharan, Lieberum, Sonnerat, Conmy, Varma, Kramár, Nanda
year: 2024
venue: arXiv
arxiv: 2407.14435
local_pdf: theory/sources/papers/rajamanoharan2024_jumprelu.pdf
type: excerpts
note: Verbatim quotations from the v3 arXiv PDF (Aug 2024). JumpReLU SAE: replace the SAE encoder's ReLU with a discontinuous JumpReLU activation, train with an L0 penalty (not L1), use straight-through estimators (STEs) to backprop through the discontinuity. SOTA reconstruction fidelity at fixed L0 on Gemma 2 9B vs. Gated and TopK SAEs. The SAE architecture used in Gemma Scope.
---

# Excerpts — Rajamanoharan et al. 2024, "JumpReLU SAEs"

## Abstract — what JumpReLU buys you {#abstract}

> Sparse autoencoders (SAEs) are a promising unsupervised approach for
> identifying causally relevant and interpretable linear features in a
> language model's (LM) activations. To be useful for downstream tasks,
> SAEs need to decompose LM activations faithfully; yet to be
> interpretable the decomposition must be sparse — two objectives that
> are in tension. In this paper, we introduce JumpReLU SAEs, which
> achieve state-of-the-art reconstruction fidelity at a given sparsity
> level on Gemma 2 9B activations, compared to other recent advances
> such as Gated and TopK SAEs. […] JumpReLU SAEs are a simple
> modification of vanilla (ReLU) SAEs — where we replace the ReLU with
> a discontinuous JumpReLU activation function — and are similarly
> efficient to train and run. By utilising straight-through-estimators
> (STEs) in a principled manner, we show how it is possible to train
> JumpReLU SAEs effectively despite the discontinuous JumpReLU function
> introduced in the SAE's forward pass. Similarly, we use STEs to
> directly train L0 to be sparse, instead of training on proxies such as
> L1, avoiding problems like shrinkage.

## §2 SAE preliminaries — the basic encoder/decoder {#sec-2}

> SAEs sparsely decompose language model activations $\mathbf{x} \in
> \mathbb{R}^n$ as a linear combination of a *dictionary* of $M \gg n$
> learned *feature* directions and then reconstruct the original
> activations using a pair of encoder and decoder functions $(\mathbf{f},
> \hat{\mathbf{x}})$ defined by:
>
> $$\mathbf{f}(\mathbf{x}) := \sigma(\mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}), \tag{1}$$
> $$\hat{\mathbf{x}}(\mathbf{f}) := \mathbf{W}_{\text{dec}} \mathbf{f} + \mathbf{b}_{\text{dec}}. \tag{2}$$
>
> In these expressions, $\mathbf{f}(\mathbf{x}) \in \mathbb{R}^M$ is a
> sparse, non-negative vector of feature magnitudes present in the
> input activation $\mathbf{x}$, whereas $\hat{\mathbf{x}}(\mathbf{f})
> \in \mathbb{R}^n$ is a reconstruction of the original activation from
> a feature representation $\mathbf{f} \in \mathbb{R}^M$. The columns of
> $\mathbf{W}_{\text{dec}}$, which we denote by $\mathbf{d}_i$ for $i =
> 1, \ldots, M$, represent the dictionary of directions into which the
> SAE decomposes $\mathbf{x}$.

The encoder pre-activations:

> $$\boldsymbol{\pi}(\mathbf{x}) := \mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}. \tag{3}$$

## §2 Activation functions across SAE variants {#sec-2-activations}

> The activation function $\sigma$ varies between architectures: Bricken
> et al. (2023) and Templeton et al. (2024) use the ReLU activation
> function, whereas TopK SAEs (Gao et al., 2024) use a TopK activation
> function (which zeroes out all but the top $K$ pre-activations). Gated
> SAEs (Rajamanoharan et al., 2024) in their general form do not fit the
> specification of Eq. (1); however with weight sharing between the two
> encoder kernels, they can be shown … to be equivalent to using a
> JumpReLU activation function, defined as
>
> $$\text{JumpReLU}_\theta(z) := z\, H(z - \theta) \tag{4}$$
>
> where $H$ is the Heaviside step function when $\theta > 0$ is the
> JumpReLU's threshold, below which pre-activations are set to zero.

## §3 The L0 loss + STE training recipe {#sec-3}

> A JumpReLU SAE is a SAE of the standard form Eq. (1) with a JumpReLU
> activation function:
>
> $$\mathbf{f}(\mathbf{x}) := \text{JumpReLU}_{\boldsymbol{\theta}}(\mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}). \tag{8}$$
>
> Compared to a ReLU SAE, it has an extra positive vector-valued
> parameter $\boldsymbol{\theta} \in \mathbb{R}^M_+$ that specifies, for
> each feature $i$, the threshold that encoder pre-activations need to
> exceed in order for the feature to be deemed active.

L0 loss:

> $$\mathcal{L}(\mathbf{x}) := \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}(\mathbf{f}(\mathbf{x}))\|_2^2}_{\mathcal{L}_{\text{reconstruct}}} + \underbrace{\lambda \|\mathbf{f}(\mathbf{x})\|_0}_{\mathcal{L}_{\text{sparsity}}}. \tag{9}$$
>
> This is a loss function of the standard form Eq. (5) where crucially
> we are using a L0 sparsity penalty to avoid the limitations of
> training with a L1 sparsity penalty (Rajamanoharan et al., 2024;
> Wright and Sharkey, 2024).

## §1 The Pareto headline — JumpReLU vs. TopK vs. Gated {#sec-1-pareto}

> We evaluate JumpReLU, Gated and TopK (Gao et al., 2024) SAEs on Gemma
> 2 9B (Gemma Team, 2024) residual stream, MLP output and attention
> output activations at several layers (Fig. 2). At any given level of
> sparsity, we find JumpReLU SAEs consistently provide more faithful
> reconstructions than Gated SAEs. JumpReLU SAEs also provide
> reconstructions that are at least as good as, and often slightly
> better than, TopK SAEs. Similar to simple ReLU SAEs, JumpReLU SAEs
> only require a single forward and backward pass during a training
> step and have an elementwise activation function (unlike TopK, which
> requires a partial sort), making them more efficient to train than
> either Gated or TopK SAEs.

## §1 Caveat — high-frequency features less interpretable {#sec-1-caveat}

> Compared to Gated SAEs, we find both TopK and JumpReLU tend to have
> more features that activate very frequently — i.e. on more than 10% of
> tokens (Fig. 5). Consistent with prior work evaluating TopK SAEs
> (Cunningham and Conerly, 2024) we find these high frequency JumpReLU
> features tend to be less interpretable, although interpretability does
> improve as SAE sparsity increases. Furthermore, only a small
> proportion of SAE features have very high frequencies: fewer than
> 0.06% in a 131k-width SAE.

This is an important caveat: high-frequency JumpReLU features are
*less* interpretable than typical ones. The headline "JumpReLU is the
Pareto winner" is correct for reconstruction-fidelity-at-fixed-sparsity,
not for "all features are interpretable."
