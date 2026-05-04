---
paper_key: shazeer2019
title: "Fast Transformer Decoding: One Write-Head is All You Need"
authors: Shazeer
year: 2019
venue: arXiv
arxiv: 1911.02150
local_pdf: theory/sources/papers/shazeer2019_mqa.pdf
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (Nov 2019). This paper introduces multi-query attention (MQA). Code blocks are kept as-is; the einsum style is intentional in the original.
---

# Excerpts — Shazeer 2019, "Fast Transformer Decoding: One Write-Head is All You Need" (MQA)

## Abstract — the bandwidth problem and the MQA fix {#abstract}

> Multi-head attention layers, as used in the Transformer neural sequence
> model, are a powerful alternative to RNNs for moving information across
> and between sequences. While training these layers is generally fast and
> simple, due to parallelizability across the length of the sequence,
> incremental inference (where such parallelization is impossible) is
> often slow, due to the memory-bandwidth cost of repeatedly loading the
> large "keys" and "values" tensors. We propose a variant called
> multi-query attention, where the keys and values are shared across all
> of the different attention "heads", greatly reducing the size of these
> tensors and hence the memory bandwidth requirements of incremental
> decoding. We verify experimentally that the resulting models can indeed
> be much faster to decode, and incur only minor quality degradation from
> the baseline.

## §1 Introduction — bandwidth as the bottleneck {#sec-1}

> The Transformer neural sequence model … relies on attention layers to
> communicate information between and across sequences. One major
> challenge with Transformer is the speed of incremental inference. As we
> will discuss, the speed of incremental Transformer inference on modern
> computing hardware is limited by the memory bandwidth necessary to
> reload the large "keys" and "values" tensors which encode the state of
> the attention layers.

## §2.3.1 Performance analysis of batched MHA {#sec-2-3-1}

The arithmetic / memory ratio for batched standard multi-head attention,
under the simplifying assumptions $m=n$, $k=v=d/h$, $n \le d$:

> The total number of arithmetic operations is $\Theta(b n d^2)$. … The
> total size of memory to be accessed is equal to the sum of the sizes of
> all the tensors involved: $O(bnd + bhn^2 + d^2)$. … Dividing the two,
> we find that the ratio of memory access to arithmetic operations is
> $O\!\left(\frac{1}{k} + \frac{1}{bn}\right)$. This low ratio is necessary
> for good performance on modern GPU/TPU hardware, where the computational
> capacity can be two orders of magnitude higher than the memory bandwidth.

## §2.4 Multi-head attention (incremental) — the diagnosis {#sec-2-4}

> Across $n$ calls, the total number of arithmetic operations is again
> $\Theta(b n d^2)$. Across $n$ calls, the total amount of memory access
> is $\Theta(b n^2 d + n d^2)$, the first term due to $K$ and $V$ and the
> second term due to $P_q$, $P_k$, $P_v$ and $P_o$.
>
> Dividing the memory by the computations, we find that the ratio of
> memory access to arithmetic operations is $\Theta\!\left(\frac{n}{d} +
> \frac{1}{b}\right)$. When $n \approx d$ or $b \approx 1$, the ratio is
> close to 1, causing memory bandwidth to be a major performance
> bottleneck on modern computing hardware. In order to make incremental
> generation efficient, we must reduce both of these terms to be $\ll 1$.
> The $\frac{1}{b}$ term is the easier one — we just can use a larger
> batch size, memory size permitting.
>
> Reducing the $\frac{n}{d}$ term is harder. This term is related to the
> size of reloading at each step the $K$ and $V$ tensors representing the
> memory which have size $bhmk = bn^2$. … In this paper we present an
> orthogonal approach to reducing the size of the $K$ and $V$ tensors —
> namely removing their "heads" dimension, while maintaining the "heads"
> dimension in the queries.

## §2.2 MHA — code for reference {#sec-2-2-code}

The reference multi-head formulation against which MQA is differenced
(per-head $k$, $v$ projections):

```python
def MultiheadAttention(x, M, P_q, P_k, P_v, P_o):
    """Multi-head Attention on one query.
    Args:
      x: a vector with shape [d]
      M: a matrix with shape [m, d]
      P_q: a tensor with shape [h, d, k]
      P_k: a tensor with shape [h, d, k]
      P_v: a tensor with shape [h, d, v]
      P_o: a tensor with shape [h, d, v]
    """
    q = tf.einsum("d,hdk->hk", x, P_q)
    K = tf.einsum("md,hdk->hmk", M, P_k)
    V = tf.einsum("md,hdv->hmv", M, P_v)
    logits = tf.einsum("hk,hmk->hm", q, K)
    weights = tf.softmax(logits)
    o = tf.einsum("hm,hmv->hv", weights, V)
    y = tf.einsum("hv,hdv->d", o, P_o)
    return y
```

## §3 (paraphrase of MQA structural change) {#mqa-structure}

In the MQA variant the per-head $h$ index is dropped from $P_k$, $P_v$
(and from the resulting $K$, $V$ tensors). Queries retain the $h$ index;
keys and values do not. The KV cache shrinks from $O(b n d)$ to $O(b n
d/h)$, a factor-of-$h$ reduction at the cost of single-head key/value
capacity.
