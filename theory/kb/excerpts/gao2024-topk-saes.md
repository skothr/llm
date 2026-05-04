---
paper_key: gao2024-topk-saes
title: "Scaling and evaluating sparse autoencoders"
authors: Gao, Dupré la Tour, Tillman, Goh, Troll, Radford, Sutskever, Leike, Wu
year: 2024
venue: OpenAI Tech Report (arXiv 2406.04093)
arxiv: 2406.04093
local_pdf: theory/sources/papers/gao2024_topk-saes.pdf
type: excerpts
note: Verbatim quotations from the OpenAI June-2024 PDF. Introduces TopK SAEs (use top-K activation, no L1) and trains a 16M-latent SAE on GPT-4 activations for 40B tokens. Establishes scaling laws L(C) and L(N, K) for SAE training. Methods now widely used: dead-latent prevention via auxiliary "AuxK" loss, encoder init from decoder transpose, learning-rate scaling 1/sqrt(n).
---

# Excerpts — Gao et al. 2024, "Scaling and evaluating sparse autoencoders"

## Abstract — TopK + scale + evaluation {#abstract}

> Sparse autoencoders provide a promising unsupervised approach for
> extracting interpretable features from a language model by
> reconstructing activations from a sparse bottleneck layer. Since
> language models learn many concepts, autoencoders need to be very
> large to recover all relevant features. However, studying the
> properties of autoencoder scaling is difficult due to the need to
> balance reconstruction and sparsity objectives and the presence of
> dead latents. We propose using $k$-sparse autoencoders [Makhzani and
> Frey, 2013] to directly control sparsity, simplifying tuning and
> improving the reconstruction-sparsity frontier. Additionally, we find
> modifications that result in few dead latents, even at the largest
> scales we tried. Using these techniques, we find clean scaling laws
> with respect to autoencoder size and sparsity. […] To demonstrate the
> scalability of our approach, we train a 16 million latent autoencoder
> on GPT-4 activations for 40 billion tokens.

## §2.2 Baseline: ReLU SAE recipe {#sec-2-2-baseline}

> For an input vector $x \in \mathbb{R}^d$ from the residual stream, and
> $n$ latent dimensions, we use baseline ReLU autoencoders from
> [Bricken et al., 2023]. The encoder and decoder are defined by:
>
> $$z = \text{ReLU}(W_{\text{enc}}(x - b_{\text{pre}}) + b_{\text{enc}}) \tag{1a}$$
> $$\hat{x} = W_{\text{dec}} z + b_{\text{pre}} \tag{1b}$$
>
> with $W_{\text{enc}} \in \mathbb{R}^{n \times d}$, $b_{\text{enc}} \in
> \mathbb{R}^n$, $W_{\text{dec}} \in \mathbb{R}^{d \times n}$, and
> $b_{\text{pre}} \in \mathbb{R}^d$. The training loss is defined by
> $\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|z\|_1$, where
> $\|x - \hat{x}\|_2^2$ is the reconstruction MSE, $\|z\|_1$ is an L1
> penalty promoting sparsity in latent activations $z$, and $\lambda$
> is a hyperparameter that needs to be tuned.

## §2.3 The TopK activation function {#sec-2-3}

> We use a $k$-sparse autoencoder [Makhzani and Frey, 2013], which
> directly controls the number of active latents by using an activation
> function (TopK) that only keeps the $k$ largest latents, zeroing the
> rest. The encoder is thus defined as:
>
> $$z = \text{TopK}(W_{\text{enc}}(x - b_{\text{pre}})) \tag{2}$$
>
> and the decoder is unchanged. The training loss is simply
> $\mathcal{L} = \|x - \hat{x}\|_2^2$.

The four claimed benefits:

> Using $k$-sparse autoencoders has a number of benefits:
>
> - It removes the need for the L1 penalty. L1 is an imperfect
>   approximation of L0, and it introduces a bias of shrinking all
>   positive activations toward zero (Section 5.1).
> - It enables setting the L0 directly, as opposed to tuning an L1
>   coefficient $\lambda$, enabling simpler model comparison and rapid
>   iteration. It can also be used in combination with arbitrary
>   activation functions.
> - It empirically outperforms baseline ReLU autoencoders on the
>   sparsity-reconstruction frontier (Figure 2a), and this gap
>   increases with scale (Figure 2b).
> - It increases monosemanticity of random activating examples by
>   effectively clamping small activations to zero (Section 4.3).

## §2.4 Preventing dead latents {#sec-2-4}

> Dead latents pose another significant difficulty in autoencoder
> training. In larger autoencoders, an increasingly large proportion of
> latents stop activating entirely at some point in training. For
> example, [Templeton et al., 2024] train a 34 million latent
> autoencoder with only 12 million alive latents, and in our ablations
> we find up to 90% dead latents (Figure 15) when no mitigations are
> applied. This results in substantially worse MSE and makes training
> computationally wasteful. We find two important ingredients for
> preventing dead latents: we initialize the encoder to the transpose of
> the decoder, and we use an auxiliary loss that models reconstruction
> error using the top-$k_\text{aux}$ dead latents (see Section A.2 for
> more details). Using these techniques, even in our largest (16
> million latent) autoencoder only 7% of latents are dead.

## §3 Scaling laws — L(C) compute-MSE frontier and L(N, k) {#sec-3-scaling}

> Following [Lindsey et al., 2024], we train autoencoders to the
> optimal MSE given the available compute, disregarding convergence.
> This method was introduced for pre-training language models [Kaplan
> et al., 2020; Hoffmann et al., 2022]. We find that MSE follows a
> power law $L(C)$ of compute […] We find that the number of tokens to
> convergence increases as approximately $\Theta(n^{0.6})$ for GPT-2
> small and $\Theta(n^{0.65})$ for GPT-4 (Figure 11).

Optimal-compute fit (from Figure 1 caption):

> $L = 0.09 + 0.056 \cdot C^{-0.084}$

## §1 Headline — 16M latents on GPT-4, 40B tokens {#sec-1-headline}

> We develop a state-of-the-art methodology to reliably train extremely
> wide and sparse autoencoders with very few dead latents on the
> activations of any language model. We systematically study the
> scaling laws with respect to sparsity, autoencoder size, and language
> model size. To demonstrate that our methodology can scale reliably,
> we train a 16 million latent autoencoder on GPT-4 [OpenAI, 2023]
> residual stream activations.
