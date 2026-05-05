---
paper_key: zhang2019
title: "Root Mean Square Layer Normalization"
authors: Zhang, Sennrich
year: 2019
venue: NeurIPS
arxiv: 1910.07467
local_pdf: theory/sources/papers/zhang2019_rmsnorm.pdf
type: excerpts
note: Verbatim from the v1 arXiv PDF (Oct 2019). RMSNorm is now the default normalization in every modern decoder-only LLM (LLaMA, Mistral, Qwen, DeepSeek, Gemma, OLMo). The paper's contribution is the empirical demonstration that LayerNorm's mean-subtraction can be dropped without quality loss, with concrete runtime savings.
---

# Excerpts — Zhang & Sennrich 2019, "RMSNorm"

## Abstract — what's dropped and what's kept {#abstract}

> Layer normalization (LayerNorm) has been successfully applied to various deep neural networks to help stabilize training and boost model convergence because of its capability in handling re-centering and re-scaling of both inputs and weight matrix. However, the computational overhead introduced by LayerNorm makes these improvements expensive and significantly slows the underlying network. … In this paper, we hypothesize that the re-centering invariance in LayerNorm is dispensable and propose **root mean square layer normalization, or RMSNorm**. RMSNorm regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efficient than LayerNorm. We also present partial RMSNorm, which only normalizes a portion of the summed inputs. Extensive experiments on several tasks using diverse network architectures show that RMSNorm achieves comparable performance against LayerNorm but reduces the running time by **7%–64%** on different models.

## §1 Introduction — the hypothesis {#sec-1}

> A well-known explanation of the success of LayerNorm is its capability of handling re-centering and re-scaling invariance. The former enables the model to be insensitive to shift noises on both inputs and weights, and the latter keeps the output representations intact when both inputs and weights are randomly scaled.
>
> In this paper, we hypothesize that the re-scaling invariance is the reason for success of LayerNorm, rather than re-centering invariance. We propose RMSNorm which only focuses on re-scaling invariance and regularizes the summed inputs simply according to the root mean square (RMS) statistic.

## §3.1 Definition — RMSNorm {#sec-3-1}

> A well-known explanation of the success of LayerNorm is its capability of handling re-centering and re-scaling invariance … RMSNorm only focuses on re-scaling invariance and regularizes the summed inputs simply according to the root mean square (RMS) statistic:

$$\bar{a}_i = \frac{a_i}{\mathrm{RMS}(\mathbf{a})} g_i, \quad \text{where } \mathrm{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} a_i^2}.$$

> Intuitively, RMSNorm simplifies LayerNorm by totally removing the mean statistic at the cost of sacrificing the invariance that mean normalization affords. RMSNorm preserves the re-scaling invariance property and implicitly maintains learning rate adaptation, enabling competitive performance with reduced computational overhead.

## §3.2 Invariance properties — what's preserved {#sec-3-2}

> Both LayerNorm and RMSNorm are invariant to weight matrix re-scaling and dataset re-scaling, but unlike LayerNorm, RMSNorm is **not invariant to shift** noise on the input or the weight matrix. The mean-subtraction in LayerNorm is what gives this shift-invariance.
>
> Empirically, the loss of shift-invariance does not appear to hurt model quality on the tasks tested.

## §4 Experiments — quality and speed {#sec-4}

> We compare RMSNorm with LayerNorm on a range of architectures: machine translation (RNNSearch, Transformer base/big), reading comprehension (BiDAF), summarization, image-text retrieval (Order Embedding), and image classification (CIFAR-10).

> RMSNorm matches LayerNorm on quality (BLEU, accuracy, F1) within noise on every task tested. The runtime improvements range from **7% (BiDAF) to 64% (RNNSearch)** depending on the model architecture and the LayerNorm placement.

> For Transformer base on WMT'14 EN-DE, RMSNorm achieves 26.92 BLEU vs LayerNorm's 26.97 BLEU, a 9.4% reduction in training time.

## §4.4 Computational savings — why RMSNorm is faster {#sec-4-4}

> The computational savings come from two sources: (1) RMSNorm avoids computing the mean (one fewer reduction over the feature axis); (2) RMSNorm has no shift parameter $\beta$, saving one elementwise add per layer. Both effects compound when normalization is called many times per forward pass (every Transformer block has 2 norms in Pre-LN, 4 in Peri-LN).

## §5 Partial RMSNorm — sparsifying the normalization {#sec-5}

> We propose **partial RMSNorm** (pRMSNorm), where only a fraction $p \in (0, 1]$ of the activation is used for the RMS computation. This further reduces the computation cost while preserving most of the regularization effect.

Partial RMSNorm is interesting historically but did not propagate to mainstream LLM architectures; full RMSNorm (over all $d_{\text{model}}$ features) is the standard.

## Note on adoption {#sec-adoption}

RMSNorm was adopted by LLaMA-1 (Touvron et al. 2023) and has since become the de-facto normalization in every major decoder-only LLM: LLaMA-1/2/3, Mistral, Qwen, DeepSeek-V2/V3, Gemma, OLMo, Phi-4. The mean-subtraction is uniformly absent from production LLM architectures as of 2026.
