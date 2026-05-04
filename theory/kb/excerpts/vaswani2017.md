---
paper_key: vaswani2017
title: Attention Is All You Need
authors: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
year: 2017
venue: NeurIPS
arxiv: 1706.03762
local_pdf: theory/sources/papers/vaswani2017_attention_is_all_you_need.pdf
type: excerpts
note: Verbatim quotations anchored to sections + equation numbers in the v3 arXiv PDF (Aug 2023 stamp). Markdown anchors are stable; equations transcribed in LaTeX where the original used display math.
---

# Excerpts — Vaswani et al. 2017, "Attention Is All You Need"

## §3.2 Attention — operational definition {#sec-3-2}

> An attention function can be described as mapping a query and a set of
> key-value pairs to an output, where the query, keys, values, and output
> are all vectors. The output is computed as a weighted sum of the values,
> where the weight assigned to each value is computed by a compatibility
> function of the query with the corresponding key.

## §3.2.1 Scaled Dot-Product Attention {#sec-3-2-1}

> We call our particular attention "Scaled Dot-Product Attention". The
> input consists of queries and keys of dimension $d_k$, and values of
> dimension $d_v$. We compute the dot products of the query with all keys,
> divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the
> weights on the values.

Equation (1) — the canonical attention formula:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \tag{1}$$

> While for small values of $d_k$ the two mechanisms perform similarly,
> additive attention outperforms dot product attention without scaling for
> larger values of $d_k$. We suspect that for large values of $d_k$, the
> dot products grow large in magnitude, pushing the softmax function into
> regions where it has extremely small gradients. To counteract this
> effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$.

Footnote 4 (variance argument):

> To illustrate why the dot products get large, assume that the components
> of $q$ and $k$ are independent random variables with mean 0 and variance
> 1. Then their dot product, $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$, has
> mean 0 and variance $d_k$.

## §3.2.2 Multi-Head Attention {#sec-3-2-2}

> Instead of performing a single attention function with $d_{\text{model}}$-
> dimensional keys, values and queries, we found it beneficial to linearly
> project the queries, keys and values $h$ times with different, learned
> linear projections to $d_k$, $d_k$ and $d_v$ dimensions, respectively. On
> each of these projected versions of queries, keys and values we then
> perform the attention function in parallel, yielding $d_v$-dimensional
> output values. These are concatenated and once again projected, resulting
> in the final values…

Equations (2)–(3):

$$\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h) W^O$$
$$\text{where } \mathrm{head}_i = \mathrm{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

> Where the projections are parameter matrices
> $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$,
> $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$,
> $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and
> $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$.
>
> In this work we employ $h = 8$ parallel attention layers, or heads. For
> each of these we use $d_k = d_v = d_{\text{model}}/h = 64$. Due to the
> reduced dimension of each head, the total computational cost is similar
> to that of single-head attention with full dimensionality.

## §3.2.2 — motivation for splitting into heads {#sec-3-2-2-motivation}

> Multi-head attention allows the model to jointly attend to information
> from different representation subspaces at different positions. With a
> single attention head, averaging inhibits this.

## §3.2.3 Three uses of attention in the model {#sec-3-2-3}

> The Transformer uses multi-head attention in three different ways:
>
> - In "encoder-decoder attention" layers, the queries come from the
>   previous decoder layer, and the memory keys and values come from the
>   output of the encoder. This allows every position in the decoder to
>   attend over all positions in the input sequence.
> - The encoder contains self-attention layers. In a self-attention layer
>   all of the keys, values and queries come from the same place, in this
>   case, the output of the previous layer in the encoder. Each position
>   in the encoder can attend to all positions in the previous layer of
>   the encoder.
> - Similarly, self-attention layers in the decoder allow each position
>   in the decoder to attend to all positions in the decoder up to and
>   including that position. We need to prevent leftward information flow
>   in the decoder to preserve the auto-regressive property. We implement
>   this inside of scaled dot-product attention by masking out (setting
>   to $-\infty$) all values in the input of the softmax which correspond
>   to illegal connections.
