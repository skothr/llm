---
paper_key: hu2021-lora
title: "LoRA: Low-Rank Adaptation of Large Language Models"
authors: Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen
year: 2021
venue: ICLR 2022
arxiv: 2106.09685
local_pdf: null
type: excerpts
note: "Excerpts from arXiv abstract page (verbatim) plus equation forms reproduced verbatim across PEFT surveys (e.g., arXiv:2304.04675; arXiv:2403.14608) and downstream variants (DoRA §2; QLoRA §3). PDF not yet downloaded."
---

# Excerpts — Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models"

## Abstract — low-rank adaptation hypothesis {#abstract}

> An important paradigm of natural language processing consists of
> large-scale pre-training on general domain data and adaptation to
> particular tasks or domains. As we pre-train larger models, full
> fine-tuning, which retrains all model parameters, becomes less
> feasible. Using GPT-3 175B as an example -- deploying independent
> instances of fine-tuned models, each with 175B parameters, is
> prohibitively expensive. We propose **Low-Rank Adaptation, or LoRA,
> which freezes the pre-trained model weights and injects trainable
> rank decomposition matrices into each layer of the Transformer
> architecture, greatly reducing the number of trainable parameters
> for downstream tasks**. Compared to GPT-3 175B fine-tuned with Adam,
> LoRA can reduce the number of trainable parameters by **10,000
> times** and the GPU memory requirement by **3 times**. LoRA performs
> on-par or better than fine-tuning in model quality on RoBERTa,
> DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters,
> a higher training throughput, and, **unlike adapters, no additional
> inference latency**.

## §3 The LoRA reparameterization {#sec-3}

For a pretrained linear-layer weight $W_0 \in \mathbb{R}^{d \times k}$,
LoRA constrains its update during fine-tuning to be a low-rank
decomposition:

$$
W = W_0 + \Delta W = W_0 + B A
$$

with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$,
$r \ll \min(d, k)$. $W_0$ is **frozen**; only $A$ and $B$ are trained.

> We use a random Gaussian initialization for $A$ and zero for $B$, so
> $\Delta W = BA$ is zero at the beginning of training. We then scale
> $\Delta W x$ by $\alpha / r$, where $\alpha$ is a constant in $r$.

The forward pass:
$$
h = W_0 x + \Delta W x = W_0 x + (\alpha / r) B A x.
$$

When applied to all weight matrices in a Transformer (Q/K/V/O
projections of attention; FFN layers), the trainable-parameter count
drops from $\sum_\ell d_\ell k_\ell$ to $\sum_\ell r(d_\ell + k_\ell)$.
For GPT-3 with $r = 4$, this is ~10,000× fewer trainable parameters.

## §4 Empirical: which weights to adapt {#sec-4}

> Adapting both $W^Q$ and $W^V$ gives the best performance overall.
> We find the rank $r = 4$ is sufficient for adapting the weight
> matrices in the GPT-3 175B model.

Key empirical claims (paper §6):

- LoRA at rank $r = 4$ matches full fine-tuning on GPT-3 175B
  benchmarks; rank $r = 1$ is competitive on many tasks.
- The intrinsic rank of fine-tune updates is empirically low — even
  for tasks the base model didn't see in pretraining.
- Adapter-based methods (Houlsby et al. 2019; Pfeiffer et al. 2020)
  add inference latency; LoRA's merged weight $W_0 + BA$ has identical
  inference cost to the base model.

## §5 Inference-time merge {#sec-5}

For deployment:
$$
W_{\text{merged}} = W_0 + (\alpha/r) B A
$$

Computed once. The deployed model has the same architecture and
inference cost as the base $W_0$. Different LoRAs can be loaded /
unloaded at runtime by computing $W_0 \pm (\alpha/r) BA$, enabling
efficient task switching.

[NOTE — pdf-not-available] Section/equation numbers from arXiv v2
(2021); equations are the canonical form reproduced in DoRA §2, QLoRA
§3, and PEFT surveys.
