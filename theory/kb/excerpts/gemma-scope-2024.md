---
paper_key: gemma-scope-2024
title: "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2"
authors: Lieberum, Rajamanoharan, Conmy, Smith, Sonnerat, Varma, Kramár, Dragan, Shah, Nanda
year: 2024
venue: arXiv:2408.05147 (Google DeepMind)
arxiv: 2408.05147
local_pdf: theory/sources/papers/gemma-scope-2024_jumprelu-saes.pdf
type: excerpts
note: Verbatim quotations from the v2 arXiv PDF (Aug 2024). The companion-release paper for the Gemma Scope SAE suite — 400+ JumpReLU SAEs trained on every layer/sublayer of Gemma 2 2B and 9B (plus select layers of 27B), totaling >30M learned features. Used >20% of GPT-3's training compute. Establishes JumpReLU SAEs as the production-grade default after Rajamanoharan 2024 introduced them; this is the at-scale validation. Deepened 2026-05-12: §2.2-jumprelu, §1-suite-scope, §4.1-pareto sections added; abstract verified verbatim.
---

# Excerpts — Lieberum et al. 2024, "Gemma Scope"

## Abstract {#abstract}

> Sparse autoencoders (SAEs) are an unsupervised method for learning a
> sparse decomposition of a neural network's latent representations into
> seemingly interpretable features. Despite recent excitement about
> their potential, research applications outside of industry are limited
> by the high cost of training a comprehensive suite of SAEs. In this
> work, we introduce Gemma Scope, an open suite of JumpReLU SAEs trained
> on all layers and sub-layers of Gemma 2 2B and 9B and select layers of
> Gemma 2 27B base models. We primarily train SAEs on the Gemma 2
> pre-trained models, but additionally release SAEs trained on
> instruction-tuned Gemma 2 9B for comparison. We evaluate the quality of
> each SAE on standard metrics and release these results. We hope that
> by releasing these SAE weights, we can help make more ambitious safety
> and interpretability research easier for the community. Weights and a
> tutorial can be found at https://huggingface.co/google/gemma-scope and
> an interactive demo can be found at https://neuronpedia.org/gemma-scope.

## §1 Compute headline {#sec-1-compute}

> Gemma Scope was a significant engineering challenge to train. It
> contains more than 400 sparse autoencoders in the main release, with
> more than 30 million learned features in total (though many features
> likely overlap), trained on 4-16B tokens of text each. We used over
> 20% of the training compute of GPT-3 (Brown et al., 2020), saved
> about 20 Pebibytes (PiB) of activations to disk, and produced
> hundreds of billions of sparse autoencoder parameters in total.

The justification for breadth:

> This was made more challenging by our decision to make a comprehensive
> suite of SAEs, on every layer and sublayer. We believe that a
> comprehensive suite is essential for enabling more ambitious
> applications of interpretability, such as circuit analysis (Conmy et
> al., 2023; Hanna et al., 2023; Wang et al., 2022), essentially scaling
> up Marks et al. (2024) to larger models, which may be necessary to
> answer mysteries about larger models like what happens during chain
> of thought or in-context learning.

## §2.1 SAE preliminaries {#sec-2-1}

> Given activations $\mathbf{x} \in \mathbb{R}^n$ from a language model,
> a sparse autoencoder (SAE) decomposes and reconstructs the activations
> using a pair of encoder and decoder functions $(\mathbf{f}, \hat{\mathbf{x}})$
> defined by:
>
> $$\mathbf{f}(\mathbf{x}) := \sigma(\mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}), \tag{1}$$
> $$\hat{\mathbf{x}}(\mathbf{f}) := \mathbf{W}_{\text{dec}} \mathbf{f} + \mathbf{b}_{\text{dec}}. \tag{2}$$
>
> These functions are trained to map $\hat{\mathbf{x}}(\mathbf{f}(\mathbf{x}))$
> back to $\mathbf{x}$, making them an autoencoder. Thus, $\mathbf{f}(\mathbf{x})
> \in \mathbb{R}^M$ is a set of linear weights that specify how to combine
> the $M \gg n$ columns of $\mathbf{W}_{\text{dec}}$ to reproduce
> $\mathbf{x}$.

## §3 Training details {#sec-3}

### Where the SAEs sit (§3.1, p.3)

The SAE locations follow Gemma 2's RMS-Norm-at-edges architecture:

> [SAEs are trained] on the residual stream […] on the MLP outputs after
> the RMSNorm has been applied and on the post MLP residual stream. For
> the attention output SAEs, we concatenate the outputs of the
> individual attention heads and learn a joint SAE for the full set of
> heads.

### #sec-3-2 — Hyperparameters (§3.2, p.4)

> **Optimization.** We use the same bandwidth $\varepsilon = 0.001$ and
> learning rate $\eta = 7 \times 10^{-5}$ across all training runs. We
> use a cosine learning rate warmup from $0.1\eta$ to $\eta$ over the
> first 1,000 training steps. We train with the Adam optimizer (Kingma
> and Ba, 2017) with $(\beta_1, \beta_2) = (0, 0.999)$, $\epsilon =
> 10^{-8}$ and a batch size of 4,096. We use a linear warmup for the
> sparsity coefficient from 0 to $\lambda$ over the first 10,000
> training steps.

> **Initialization.** We initialize the JumpReLU threshold as the
> vector $\boldsymbol{\theta} = \{0.001\}^M$. We initialize
> $\mathbf{W}_{\text{dec}}$ using He-uniform (He et al., 2015)
> initialization and rescale each latent vector to be unit norm.
> $\mathbf{W}_{\text{enc}}$ is initalized as the transpose of
> $\mathbf{W}_{\text{dec}}$, but they are not tied afterwards (Conerly
> et al., 2024; Gao et al., 2024). The biases $\mathbf{b}_{\text{dec}}$
> and $\mathbf{b}_{\text{enc}}$ are initialized to zero vectors.

The decoder normalization protocol:

> Throughout training, we restrict the columns of $\mathbf{W}_{\text{dec}}$
> to have unit norm by renormalizing after every update. We also project
> out the part of the gradients parallel to these columns before
> computing the Adam update, as described in Bricken et al. (2023).

## §4.1 Sparsity-fidelity trade-off finding {#sec-4-1-finding}

> Fig. 2 compares the sparsity-fidelity trade-off for SAEs in the middle
> of each Gemma model. […] Delta loss is consistently higher for
> residual stream SAEs compared to MLP and attention SAEs, whereas FVU
> (Fig. 13) is roughly comparable across sites. We conjecture this is
> because even small errors in reconstructing the residual stream can
> have a significant impact on LM loss, whereas relatively large errors
> in reconstructing a single MLP or attention sub-layer's output have a
> limited impact on the LM loss.

The footnote explanation:

> The residual stream is the bottleneck by which the previous layers
> communicate with all later layers. Any given MLP or attention layer
> adds to the residual stream, and is typically only a small fraction
> of the residual, so even a large error in the layer is a small error
> to the residual stream and so to the rest of the network's
> processing. On the other hand, a large error to the residual stream
> itself will significantly harm loss. For the same reason, mean
> ablating the residual stream has far higher impact on the loss than
> mean ablating a single layer.

This is methodologically load-bearing for circuit analysis on residual
SAEs: residual-stream-SAE reconstruction errors compound; MLP-output
or attention-output errors are absorbed by downstream layers' residual
contributions.

## §4 Evaluation framing {#sec-4}

> [W]e note however that as of now there is no consensus on what
> constitutes a reliable metric for the quality of a sparse autoencoder
> or its learned latents and that this is an ongoing area of research
> and debate (Gao et al., 2024; Karvonen et al., 2024; Makelov et al.,
> 2024).

This is the explicit frontier-and-open-question disclaimer: even at
2024-end, the field has no agreed-upon SAE quality metric. Gemma Scope
reports multiple metrics (delta-loss, FVU, downstream-task evaluation)
because no single one is canonical.

## Widths released — the suite scope

Per §3 / Appendix: SAEs released at widths {16.4K, 65.5K, 131K, 1M},
trained for 4-16B tokens each, on Gemma 2 2B and 9B (every layer and
sublayer), plus select layers of Gemma 2 27B. License: CC-BY-4.0.

## §2.2 JumpReLU Activation Function {#sec-2-2-jumprelu}

> JumpReLU activation The JumpReLU activation is a shifted Heaviside
> step function as a gating mechanism together with a conventional ReLU:
>
> $$\sigma(\mathbf{z}) = \text{JumpReLU}_{\boldsymbol{\theta}}(\mathbf{z}) := \mathbf{z} \odot H(\mathbf{z} - \boldsymbol{\theta}). \tag{3}$$
>
> Here, $\boldsymbol{\theta} > 0$ is the JumpReLU's vector-valued
> learnable threshold parameter, $\odot$ denotes elementwise
> multiplication, and $H$ is the Heaviside step function, which is 1 if
> its input is positive and 0 otherwise. Intuitively, the JumpReLU
> leaves the pre-activations unchanged above the threshold, but sets
> them to zero below the threshold, with a different learned threshold
> per latent.

The training loss that pairs with Eq. (3):

> As loss function we use a squared error reconstruction loss, and
> directly regularize the number of active (non-zero) latents using the
> L0 penalty:
>
> $$\mathcal{L} := \|\mathbf{x} - \hat{\mathbf{x}}(\mathbf{f}(\mathbf{x}))\|_2^2 + \lambda \|\mathbf{f}(\mathbf{x})\|_0, \tag{4}$$
>
> where $\lambda$ is the sparsity penalty coefficient. Since the L0
> penalty and JumpReLU activation function are piecewise constant with
> respect to threshold parameters $\boldsymbol{\theta}$, we use
> straight-through estimators (STEs) to train $\boldsymbol{\theta}$,
> using the approach described in Rajamanoharan et al. (2024b).

Equations (3) and (4) are the core training objective distinguishing JumpReLU SAEs from ReLU+L1 SAEs (Bricken 2023) and TopK SAEs (Gao 2024). The L0 penalty directly controls active latent count rather than using an L1 surrogate, and the per-latent threshold $\boldsymbol{\theta}$ is trained via STE. This is the equation cited downstream whenever JumpReLU's training formulation is referenced.

## §1 Suite Scope — 400+ SAEs, 2,000+ Released {#sec-1-suite-scope}

> Gemma Scope was a significant engineering challenge to train. It
> contains more than 400 sparse autoencoders in the main release, with
> more than 30 million learned features in total (though many features
> likely overlap), trained on 4-16B tokens of text each. We used over
> 20% of the training compute of GPT-3 (Brown et al., 2020), saved
> about 20 Pebibytes (PiB) of activations to disk, and produced
> hundreds of billions of sparse autoencoder parameters in total.

The footnote quantifying total release scope:

> For each model, layer and site we in fact release multiple SAEs with
> differing levels of sparsity; taking this into account, we release the
> weights of over 2,000 SAEs in total.

Coverage: every layer and sublayer of Gemma 2 2B (26 layers) and 9B (42 layers) at widths {16.4K, 65.5K, 131K, 1M} for attention, MLP, and residual stream sites; plus select layers of 27B (46 layers). See Table 1 of the paper for the full matrix. License: CC-BY-4.0.

## §4.1 Sparsity-Fidelity Pareto Frontier {#sec-4-1-pareto}

The evaluation metrics defined at §4.1:

> We use the mean L0-norm of latent activations, $\mathbb{E}_{\mathbf{x}} \|\mathbf{f}(\mathbf{x})\|_0$,
> as a measure of sparsity. To measure reconstruction fidelity, we use
> two metrics:
>
> - Our primary metric is delta LM loss, the increase in the
>   cross-entropy loss experienced by the LM when we splice the SAE into
>   the LM's forward pass.
> - As a secondary metric, we also use fraction of variance unexplained
>   (FVU) — also called the normalized loss (Gao et al., 2024) — as a
>   measure of reconstruction fidelity.

The headline finding from §4.1:

> Delta loss is consistently higher for residual stream SAEs compared to
> MLP and attention SAEs, whereas FVU (Fig. 13) is roughly comparable
> across sites. We conjecture this is because even small errors in
> reconstructing the residual stream can have a significant impact on LM
> loss, whereas relatively large errors in reconstructing a single MLP
> or attention sub-layer's output have a limited impact on the LM loss.

This finding is methodologically load-bearing for downstream circuit analysis: residual-stream SAE reconstruction errors compound through all later layers, while MLP-output or attention-output errors are absorbed by the residual sum. The Figure 2 plot (delta loss vs. L0, stratified by width and model) is the Pareto frontier visualization cited whenever comparing SAE quality across Gemma Scope configurations.

## Source notes

- Tier A (Google DeepMind tech report; companion to a published model
  release).
- PDF retrieved from arXiv (v2, 2024-08-19).
- Architectural prerequisite: `kb/excerpts/rajamanoharan2024-jumprelu`
  (the SAE architecture used). Methodological cousin:
  `kb/excerpts/gao2024-topk-saes` (OpenAI's analogous large-SAE
  release on GPT-4, using TopK rather than JumpReLU).
- Public weights: https://huggingface.co/google/gemma-scope. Demo:
  https://neuronpedia.org/gemma-scope.

[Verified from PDF on 2026-05-12] Added §2.2-jumprelu (Eq. 3 + Eq. 4), §1-suite-scope, §4.1-pareto. Abstract verified verbatim.
