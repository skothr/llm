---
paper_key: gemma-scope-2024
title: "Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2"
authors: Lieberum, Rajamanoharan, Conmy, Smith, Sonnerat, Varma, Kramár, Dragan, Shah, Nanda
year: 2024
venue: arXiv:2408.05147 (Google DeepMind)
arxiv: 2408.05147
local_pdf: theory/sources/papers/gemma-scope-2024_jumprelu-saes.pdf
type: excerpts
note: Verbatim quotations from the v2 arXiv PDF (Aug 2024). The companion-release paper for the Gemma Scope SAE suite — 400+ JumpReLU SAEs trained on every layer/sublayer of Gemma 2 2B and 9B (plus select layers of 27B), totaling >30M learned features. Used >20% of GPT-3's training compute. Establishes JumpReLU SAEs as the production-grade default after Rajamanoharan 2024 introduced them; this is the at-scale validation.
---

# Excerpts — Lieberum et al. 2024, "Gemma Scope"

## #abstract

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

## #sec-1 — The compute-headline (§1, p.1-2)

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

## #sec-2-1 — SAE preliminaries (§2.1, p.2)

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

## #sec-3 — Training details (§3, p.3-4)

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

## #sec-4-1 — Sparsity-fidelity trade-off finding (§4.1, p.5-6)

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

## #sec-4 — Evaluation framing (§4, p.6)

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
