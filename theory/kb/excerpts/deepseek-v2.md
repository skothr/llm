---
paper_key: deepseek-v2
title: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
authors: DeepSeek-AI
year: 2024
venue: arXiv (tech report)
arxiv: 2405.04434
local_pdf: theory/sources/papers/deepseek-v2.pdf
type: excerpts
note: Verbatim quotations from the v5 arXiv PDF (Jun 2024). The architecture-defining contributions are Multi-Head Latent Attention (MLA, §2.1) and DeepSeekMoE; only the MLA portion is excerpted here. MoE excerpts go in a separate file when that area is built.
---

# Excerpts — DeepSeek-AI 2024, "DeepSeek-V2" — MLA portion

## §2 Architecture — DeepSeek-V2 is still a Transformer {#sec-2}

> By and large, DeepSeek-V2 is still in the Transformer architecture
> (Vaswani et al., 2017), where each Transformer block consists of an
> attention module and a Feed-Forward Network (FFN). However, for both
> the attention module and the FFN, we design and employ innovative
> architectures. For attention, we design MLA, which utilizes low-rank
> key-value joint compression to eliminate the bottleneck of inference-
> time key-value cache, thus supporting efficient inference.

## §2.1 MLA: boosting inference efficiency — motivation {#sec-2-1}

> Conventional Transformer models usually adopts Multi-Head Attention
> (MHA) (Vaswani et al., 2017), but during generation, its heavy Key-
> Value (KV) cache will become the bottleneck that limit the inference
> efficiency. In order to reduce the KV cache, Multi-Query Attention
> (MQA) (Shazeer, 2019) and Grouped-Query Attention (GQA) (Ainslie et
> al., 2023) are proposed. They require a smaller magnitude of KV cache,
> but their performance does not match MHA…
>
> For DeepSeek-V2, we design an innovative attention mechanism called
> Multi-head Latent Attention (MLA). Equipped with low-rank key-value
> joint compression, MLA achieves better performance than MHA, but
> requires a significantly smaller amount of KV cache.

## §2.1.1 Standard MHA preliminaries — Eq. (1)–(8) {#sec-2-1-1}

Let $d$ be the embedding dimension, $n_h$ the number of heads, $d_h$ the
per-head dimension, $\mathbf{h}_t \in \mathbb{R}^d$ the attention input
at token $t$:

$$\mathbf{q}_t = W^Q \mathbf{h}_t \tag{1}$$
$$\mathbf{k}_t = W^K \mathbf{h}_t \tag{2}$$
$$\mathbf{v}_t = W^V \mathbf{h}_t \tag{3}$$

Sliced into heads:

$$[\mathbf{q}_{t,1}; \mathbf{q}_{t,2}; \ldots; \mathbf{q}_{t,n_h}] = \mathbf{q}_t \tag{4}$$
$$[\mathbf{k}_{t,1}; \mathbf{k}_{t,2}; \ldots; \mathbf{k}_{t,n_h}] = \mathbf{k}_t \tag{5}$$
$$[\mathbf{v}_{t,1}; \mathbf{v}_{t,2}; \ldots; \mathbf{v}_{t,n_h}] = \mathbf{v}_t \tag{6}$$

Per-head attention, then output projection:

$$\mathbf{o}_{t,i} = \sum_{j=1}^{t} \mathrm{Softmax}_j\!\left(\frac{\mathbf{q}_{t,i}^\top \mathbf{k}_{j,i}}{\sqrt{d_h}}\right) \mathbf{v}_{j,i} \tag{7}$$
$$\mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \mathbf{o}_{t,2}; \ldots; \mathbf{o}_{t,n_h}] \tag{8}$$

> During inference, all keys and values need to be cached to accelerate
> inference, so MHA needs to cache $2 n_h d_h l$ elements for each token.
> In model deployment, this heavy KV cache is a large bottleneck that
> limits the maximum batch size and sequence length.

## §2.1.2 Low-rank KV joint compression — Eq. (9)–(13) {#sec-2-1-2}

> The core of MLA is the low-rank joint compression for keys and values
> to reduce KV cache:

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \tag{9}$$
$$\mathbf{k}_t^C = W^{UK} \mathbf{c}_t^{KV} \tag{10}$$
$$\mathbf{v}_t^C = W^{UV} \mathbf{c}_t^{KV} \tag{11}$$

> where $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ is the compressed latent
> vector for keys and values; $d_c (\ll d_h n_h)$ denotes the KV
> compression dimension; $W^{DKV} \in \mathbb{R}^{d_c \times d}$ is the
> down-projection matrix; and $W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h
> \times d_c}$ are the up-projection matrices for keys and values,
> respectively. **During inference, MLA only needs to cache $\mathbf{c}_t^{KV}$,
> so its KV cache has only $d_c l$ elements**, where $l$ denotes the
> number of layers. In addition, during inference, since $W^{UK}$ can be
> absorbed into $W^Q$, and $W^{UV}$ can be absorbed into $W^O$, we even
> do not need to compute keys and values out for attention.

Low-rank compression for queries (does not reduce KV cache, but reduces
training activation memory):

$$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \tag{12}$$
$$\mathbf{q}_t^C = W^{UQ} \mathbf{c}_t^Q \tag{13}$$

## §2.1.3 Decoupled RoPE — Eq. (14)–(19) {#sec-2-1-3}

The crucial constraint: standard RoPE is incompatible with the cache-
absorption trick because it inserts a per-position rotation between $W^Q$
and $W^{UK}$, breaking commutativity. The fix:

> We propose the decoupled RoPE strategy that uses additional multi-head
> queries $\mathbf{q}_{t,i}^R \in \mathbb{R}^{d_h^R}$ and a shared key
> $\mathbf{k}_t^R \in \mathbb{R}^{d_h^R}$ to carry RoPE, where $d_h^R$
> denotes the per-head dimension of the decoupled queries and key.

$$[\mathbf{q}_{t,1}^R; \ldots; \mathbf{q}_{t,n_h}^R] = \mathbf{q}_t^R = \mathrm{RoPE}(W^{QR} \mathbf{c}_t^Q) \tag{14}$$
$$\mathbf{k}_t^R = \mathrm{RoPE}(W^{KR} \mathbf{h}_t) \tag{15}$$
$$\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^C; \mathbf{q}_{t,i}^R] \tag{16}$$
$$\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^C; \mathbf{k}_t^R] \tag{17}$$
$$\mathbf{o}_{t,i} = \sum_{j=1}^{t} \mathrm{Softmax}_j\!\left(\frac{\mathbf{q}_{t,i}^\top \mathbf{k}_{j,i}}{\sqrt{d_h + d_h^R}}\right) \mathbf{v}_{j,i}^C \tag{18}$$
$$\mathbf{u}_t = W^O [\mathbf{o}_{t,1}; \ldots; \mathbf{o}_{t,n_h}] \tag{19}$$

> Therefore, DeepSeek-V2 requires a total KV cache containing $(d_c +
> d_h^R) l$ elements.

Note the shared $\mathbf{k}_t^R$ across heads — the RoPE-carrying key is
single-head, similar in spirit to MQA's shared keys, but the
content-bearing key $\mathbf{k}_{t,i}^C$ is recovered per-head from the
latent.

## §2.1.4 KV-cache table {#sec-2-1-4}

Comparison from Table 1 (KV-cache elements per token, $l$ layers):

| Mechanism | KV Cache per Token | Capability |
|-----------|-------------------|------------|
| MHA       | $2 n_h d_h l$     | Strong     |
| GQA       | $2 n_g d_h l$     | Moderate   |
| MQA       | $2 d_h l$         | Weak       |
| **MLA**   | $(d_c + d_h^R) l \approx \tfrac{9}{2} d_h l$ | **Stronger** |

> MLA requires only a small amount of KV cache, equal to GQA with only
> 2.25 groups, but can achieve stronger performance than MHA.

For DeepSeek-V2, $d_c = 4 d_h$ and $d_h^R = d_h/2$, so MLA's per-token
cache is $\tfrac{9}{2} d_h l$ — larger than MQA but far smaller than
MHA, with quality matching or exceeding MHA.
