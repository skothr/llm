---
paper_key: vaswani2017
title: Attention Is All You Need
authors: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
year: 2017
venue: NeurIPS
arxiv: 1706.03762
local_pdf: theory/sources/papers/vaswani2017_attention_is_all_you_need.pdf
type: excerpts
extracted: 2026-05-04 (initial); 2026-05-05 (Phase 2.5 deepening — abstract + §3.1, §3.3, §3.5, §5.3 added)
note: Verbatim quotations anchored to sections + equation numbers in the v7 arXiv PDF (Aug 2023 stamp). Markdown anchors are stable; equations transcribed in LaTeX where the original used display math.
---

# Excerpts — Vaswani et al. 2017, "Attention Is All You Need"

## #abstract

> The dominant sequence transduction models are based on complex
> recurrent or convolutional neural networks that include an encoder
> and a decoder. The best performing models also connect the encoder
> and decoder through an attention mechanism. We propose a new simple
> network architecture, the Transformer, based solely on attention
> mechanisms, dispensing with recurrence and convolutions entirely.
> Experiments on two machine translation tasks show these models to
> be superior in quality while being more parallelizable and requiring
> significantly less time to train. Our model achieves 28.4 BLEU on
> the WMT 2014 English-to-German translation task, improving over the
> existing best results, including ensembles, by over 2 BLEU. On the
> WMT 2014 English-to-French translation task, our model establishes
> a new single-model state-of-the-art BLEU score of 41.8 after
> training for 3.5 days on eight GPUs, a small fraction of the
> training costs of the best models from the literature. We show that
> the Transformer generalizes well to other tasks by applying it
> successfully to English constituency parsing both with large and
> limited training data.

## §3.1 Encoder and Decoder Stacks {#sec-3-1}

> **Encoder:** The encoder is composed of a stack of $N = 6$ identical
> layers. Each layer has two sub-layers. The first is a multi-head
> self-attention mechanism, and the second is a simple, position-wise
> fully connected feed-forward network. We employ a residual connection
> [11] around each of the two sub-layers, followed by layer normalization
> [1]. That is, the output of each sub-layer is
> $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$
> is the function implemented by the sub-layer itself. To facilitate
> these residual connections, all sub-layers in the model, as well as
> the embedding layers, produce outputs of dimension $d_{\text{model}} = 512$.

> **Decoder:** The decoder is also composed of a stack of $N = 6$
> identical layers. In addition to the two sub-layers in each encoder
> layer, the decoder inserts a third sub-layer, which performs
> multi-head attention over the output of the encoder stack. Similar to
> the encoder, we employ residual connections around each of the
> sub-layers, followed by layer normalization. We also modify the
> self-attention sub-layer in the decoder stack to prevent positions
> from attending to subsequent positions. This masking, combined with
> fact that the output embeddings are offset by one position, ensures
> that the predictions for position $i$ can depend only on the known
> outputs at positions less than $i$.

This is the **post-LN** placement (`LayerNorm(x + Sublayer(x))`) — the
original Transformer recipe. Pre-LN
(`x + Sublayer(LayerNorm(x))`) emerged as the dominant variant after Xiong
et al. 2020 (`kb/notes/architecture/normalization.md`).

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

## §3.3 Position-wise Feed-Forward Networks {#sec-3-3}

> In addition to attention sub-layers, each of the layers in our encoder
> and decoder contains a fully connected feed-forward network, which is
> applied to each position separately and identically. This consists of
> two linear transformations with a ReLU activation in between.

Equation (2):

$$\mathrm{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2 \tag{2}$$

> While the linear transformations are the same across different
> positions, they use different parameters from layer to layer. Another
> way of describing this is as two convolutions with kernel size 1. The
> dimensionality of input and output is $d_{\text{model}} = 512$, and
> the inner-layer has dimensionality $d_{\text{ff}} = 2048$.

Sets the canonical $d_{\text{ff}} = 4 \cdot d_{\text{model}}$ width
ratio that survived as a default through GPT-2/3, OPT, LLaMA-1 (LLaMA-2/3
shift toward $\approx 2.67 \cdot d_{\text{model}}$ once SwiGLU is gating;
see `kb/notes/architecture/ffn.md`).

## §3.4 Embeddings and Softmax — weight tying {#sec-3-4}

> Similarly to other sequence transduction models, we use learned
> embeddings to convert the input tokens and output tokens to vectors of
> dimension $d_{\text{model}}$. We also use the usual learned linear
> transformation and softmax function to convert the decoder output to
> predicted next-token probabilities. In our model, we share the same
> weight matrix between the two embedding layers and the pre-softmax
> linear transformation, similar to [Press & Wolf 2017]. In the embedding
> layers, we multiply those weights by $\sqrt{d_{\text{model}}}$.

The original locus of **weight tying** in the Transformer recipe.

## §3.5 Positional Encoding {#sec-3-5}

> Since our model contains no recurrence and no convolution, in order
> for the model to make use of the order of the sequence, we must inject
> some information about the relative or absolute position of the tokens
> in the sequence. To this end, we add "positional encodings" to the
> input embeddings at the bottoms of the encoder and decoder stacks. The
> positional encodings have the same dimension $d_{\text{model}}$ as the
> embeddings, so that the two can be summed.

The sinusoidal formula:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

> where $pos$ is the position and $i$ is the dimension. That is, each
> dimension of the positional encoding corresponds to a sinusoid. The
> wavelengths form a geometric progression from $2\pi$ to $10000 \cdot
> 2\pi$. We chose this function because we hypothesized it would allow
> the model to easily learn to attend by relative positions, since for
> any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear
> function of $PE_{pos}$.

> We also experimented with using learned positional embeddings [9]
> instead, and found that the two versions produced nearly identical
> results (see Table 3 row (E)). We chose the sinusoidal version because
> it may allow the model to extrapolate to sequence lengths longer than
> the ones encountered during training.

The fixed/learned dichotomy here is the **starting point** of the
positional-encoding lineage — RoPE (Su 2021), ALiBi (Press 2022), and
NoPE (Kazemnejad 2023) all argue against this absolute-additive choice.
See `kb/notes/architecture/positional-encodings.md`.

## §5.3 Optimizer — the inverse-square-root LR schedule {#sec-5-3}

> We used the Adam optimizer [20] with $\beta_1 = 0.9$, $\beta_2 = 0.98$
> and $\epsilon = 10^{-9}$. We varied the learning rate over the course
> of training, according to the formula:

Equation (3):

$$\mathrm{lrate} = d_{\text{model}}^{-0.5} \cdot \min\!\big(\mathrm{step\_num}^{-0.5},\ \mathrm{step\_num} \cdot \mathrm{warmup\_steps}^{-1.5}\big) \tag{3}$$

> This corresponds to increasing the learning rate linearly for the
> first warmup_steps training steps, and decreasing it thereafter
> proportionally to the inverse square root of the step number. We used
> $\mathrm{warmup\_steps} = 4000$.

The "Noam schedule" — warmup-then-inverse-sqrt — that became the de-facto
default for early Transformer training. Modern LLM pretraining has
largely shifted to **cosine** decay (matched to total token count, per
Chinchilla §A1) and constant-with-cooldown variants. The $\beta_2 = 0.98$
choice (vs Adam's default 0.999) is the original specification.

## §5.4 Regularization {#sec-5-4}

> We employ three types of regularization during training:

> **Residual Dropout** We apply dropout [33] to the output of each
> sub-layer, before it is added to the sub-layer input and normalized.
> In addition, we apply dropout to the sums of the embeddings and the
> positional encodings in both the encoder and decoder stacks. For the
> base model, we use a rate of $P_{\text{drop}} = 0.1$.

> **Label Smoothing** During training, we employed label smoothing of
> value $\epsilon_{\text{ls}} = 0.1$ [36]. This hurts perplexity, as the
> model learns to be more unsure, but improves accuracy and BLEU score.

The label-smoothing-vs-perplexity contradiction is canonical: optimizing
for the calibrated next-token distribution (low PPL) is **not** the same
objective as optimizing for top-1 / beam-decoded BLEU. Modern LLM
pretraining typically omits label smoothing; SFT/RLHF reintroduces a
related trade-off (see `kb/notes/post-training/sft.md`).

## Source notes

- Tier A canonical (NeurIPS 2017; the founding paper of the Transformer
  architecture).
- PDF version v7 (Aug 2 2023 stamp on the arXiv PDF) — same content as
  v3/v5, footnotes preserved across revisions.
- Phase 2.5 (2026-05-05) deepening pass added the abstract and §3.1,
  §3.3, §3.4, §3.5, §5.3, §5.4 anchors. The original Phase 2 file held
  only the §3.2 attention block.
