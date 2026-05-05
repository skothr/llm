---
paper-key: xiong2020
title: "On Layer Normalization in the Transformer Architecture"
authors: Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang, Liu
year: 2020
venue: ICML
arxiv: 2002.04745
pdf: theory/sources/papers/xiong2020_layer_norm.pdf
extracted: 2026-05-05 (Phase 2.5 deepening)
---

# Verbatim excerpts — Xiong et al. 2020, "On Layer Normalization in the Transformer Architecture"

The paper that established Pre-LN as the default placement for layer
normalization in production Transformers. The theoretical contribution
is a mean-field-theory analysis showing Post-LN gradients near the
output layer scale with $O(\sqrt{L})$ (where $L$ is depth), motivating
the warmup, while Pre-LN gradients are $O(1)$ at initialization, allowing
warmup-free training. Every modern decoder-only LLM (GPT-3, LLaMA,
Mistral, Qwen, DeepSeek) uses Pre-LN.

## #abstract

> The Transformer is widely used in natural language processing tasks.
> To train a Transformer however, one usually needs a carefully designed
> learning rate warm-up stage, which is shown to be crucial to the final
> performance but will slow down the optimization and bring more
> hyper-parameter tunings. In this paper, we first study theoretically
> why the learning rate warm-up stage is essential and show that the
> location of layer normalization matters. Specifically, we prove with
> mean field theory that at initialization, for the original-designed
> Post-LN Transformer, which places the layer normalization between the
> residual blocks, the expected gradients of the parameters near the
> output layer are large. Therefore, using a large learning rate on
> those gradients makes the training unstable. The warm-up stage is
> practically helpful for avoiding this problem. On the other hand, our
> theory also shows that if the layer normalization is put inside the
> residual blocks (recently proposed as Pre-LN Transformer), the
> gradients are well-behaved at initialization. This motivates us to
> remove the warm-up stage for the training of Pre-LN Transformers. We
> show in our experiments that Pre-LN Transformers without the warm-up
> stage can reach comparable results with baselines while requiring
> significantly less training time and hyper-parameter tuning on a wide
> range of applications.

## #sec-1 — Contributions and motivation (§1)

> Our contributions are summarized as follows:
>
> - We investigate two Transformer variants, the Post-LN Transformer and
>   the Pre-LN Transformer, using mean field theory. By studying the
>   gradients at initialization, we provide evidence to show why the
>   learning rate warm-up stage is essential in training the Post-LN
>   Transformer.
>
> - We are the first to show that the learning-rate warm-up stage can be
>   removed for the Pre-LN Transformer, which eases the hyper-parameter
>   tuning. We further show that by using proper learning rate
>   schedulers, the training time can be largely reduced on a wide range
>   of applications.

## #sec-3-1 — Post-LN vs Pre-LN architectural definition (§3.1, Table 1)

The two variants differ only in where LayerNorm is applied. Post-LN
places it after the residual addition; Pre-LN places it before each
sub-layer (inside the residual branch) and adds a final LayerNorm before
the output projection.

**Post-LN Transformer** (the original design from Vaswani et al. 2017):

$$x_{l,i}^{\mathrm{post,1}} = \mathrm{MultiHeadAtt}(x_{l,i}^{\mathrm{post}}, [x_{l,1}^{\mathrm{post}}, \ldots, x_{l,n}^{\mathrm{post}}])$$
$$x_{l,i}^{\mathrm{post,2}} = x_{l,i}^{\mathrm{post}} + x_{l,i}^{\mathrm{post,1}}$$
$$x_{l,i}^{\mathrm{post,3}} = \mathrm{LayerNorm}(x_{l,i}^{\mathrm{post,2}})$$
$$x_{l,i}^{\mathrm{post,4}} = \mathrm{ReLU}(x_{l,i}^{\mathrm{post,3}} W^{1,l} + b^{1,l}) W^{2,l} + b^{2,l}$$
$$x_{l,i}^{\mathrm{post,5}} = x_{l,i}^{\mathrm{post,3}} + x_{l,i}^{\mathrm{post,4}}$$
$$x_{l+1,i}^{\mathrm{post}} = \mathrm{LayerNorm}(x_{l,i}^{\mathrm{post,5}})$$

**Pre-LN Transformer:**

$$x_{l,i}^{\mathrm{pre,1}} = \mathrm{LayerNorm}(x_{l,i}^{\mathrm{pre}})$$
$$x_{l,i}^{\mathrm{pre,2}} = \mathrm{MultiHeadAtt}(x_{l,i}^{\mathrm{pre,1}}, [x_{l,1}^{\mathrm{pre,1}}, \ldots, x_{l,n}^{\mathrm{pre,1}}])$$
$$x_{l,i}^{\mathrm{pre,3}} = x_{l,i}^{\mathrm{pre}} + x_{l,i}^{\mathrm{pre,2}}$$
$$x_{l,i}^{\mathrm{pre,4}} = \mathrm{LayerNorm}(x_{l,i}^{\mathrm{pre,3}})$$
$$x_{l,i}^{\mathrm{pre,5}} = \mathrm{ReLU}(x_{l,i}^{\mathrm{pre,4}} W^{1,l} + b^{1,l}) W^{2,l} + b^{2,l}$$
$$x_{l+1,i}^{\mathrm{pre}} = x_{l,i}^{\mathrm{pre,3}} + x_{l,i}^{\mathrm{pre,5}}$$
$$\text{Final LayerNorm: } x_{\mathrm{Final},i}^{\mathrm{pre}} \leftarrow \mathrm{LayerNorm}(x_{L+1,i}^{\mathrm{pre}})$$

## #sec-3-2 — The warmup schedule (§3.2, Eq. 1)

> A learning rate warm-up stage for the Post-LN Transformer seems
> critical (Popel & Bojar, 2018). We denote the learning rate of the
> t-th iteration as $\mathrm{lr}(t)$ and the maximum learning rate during
> training as $\mathrm{lr}_{\max}$. Given a predefined time frame
> $T_{\mathrm{warmup}}$, the learning rate scheduler for the first
> $T_{\mathrm{warmup}}$ iterations (Vaswani et al., 2018) is defined as
>
> $$\mathrm{lr}(t) = \frac{t}{T_{\mathrm{warmup}}} \mathrm{lr}_{\max}, \quad t \le T_{\mathrm{warmup}}. \tag{1}$$
>
> After this warm-up stage, the learning rate will be set by classical
> learning rate schedulers, such as the linear decay, the inverse
> square-root decay, or forced decay at particular iterations.

## #sec-4 — Mean-field-theory result on gradient scales (§4, Theorem 1)

The load-bearing theoretical claim. Paraphrased from §4.1 / Theorem 1:

> **Theorem 1 (Gradients at initialization, Post-LN).** Assume the
> Post-LN Transformer is initialized with the standard Xavier scheme.
> At initialization, the expected gradient norm of the parameters in
> the last layer's FFN sub-layer satisfies
>
> $$\|\partial \mathcal{L}/\partial W^{2,L}\|_F = O(d \sqrt{\ln d})$$
>
> independent of $L$, while the gradient through the residual stream to
> any earlier layer accumulates a factor of $O(\sqrt{L})$ near the
> output. As $L$ grows, the parameter updates near the output layer
> become disproportionately large.

> **Theorem 2 (Pre-LN counterpart).** For the Pre-LN Transformer with
> the same initialization, the expected gradient norm at every layer
> is $O(d \sqrt{\ln d / L})$. The gradients are well-balanced across
> depth and do not blow up near the output.

The practical consequence: **Post-LN needs warmup to compensate for
output-layer gradient scale; Pre-LN does not.**

## #sec-5 — Experimental confirmation (§5, paraphrased)

> We conduct experiments on IWSLT14 De-En, WMT14 En-De, and BERT
> pre-training. In all settings, the Pre-LN Transformer trained without
> any warm-up stage matches or exceeds the Post-LN baseline that uses
> the standard warm-up schedule, while reducing total training time.

> The Post-LN Transformer with $\mathrm{lr}_{\max} = 10^{-3}$ and no
> warmup diverges; with $T_{\mathrm{warmup}} = 4000$ the same setting
> trains stably. The Pre-LN Transformer with $\mathrm{lr}_{\max} = 10^{-3}$
> and no warmup trains stably and reaches the same final BLEU.

## Source notes

- Tier A canonical (ICML 2020 main conference). The mean-field-theory
  analysis is reproduced in Theorem 1/2 (§4) but the proofs occupy the
  technical appendices.
- PDF retrieved from arXiv. Section / equation anchors stable.
- Modern frontier LLMs all use Pre-LN. The 2025 Peri-LN paper (Lee et
  al.) revisits this design space and argues that Peri-LN — with norm
  on **both** the input and the output of each sub-layer — improves on
  Pre-LN at frontier scale; see `kb/notes/architecture/normalization.md`
  for the lineage.
