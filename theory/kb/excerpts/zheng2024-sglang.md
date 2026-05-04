---
paper_key: zheng2024-sglang
title: "SGLang: Efficient Execution of Structured Language Model Programs"
authors: Zheng, Yin, Xie, Sun, Huang, Yu, Cao, Kozyrakis, Stoica, Gonzalez, Barrett, Sheng
year: 2024
venue: NeurIPS
arxiv: 2312.07104
local_pdf: theory/sources/papers/zheng2024_sglang.pdf
type: excerpts
note: Verbatim from the v2 arXiv PDF (Jun 2024, NeurIPS 2024). SGLang is a frontend DSL + runtime; the runtime contributions of immediate inference relevance are RadixAttention (KV cache reuse via radix tree) and the compressed-FSM constrained-decoding engine. The paper carefully demarcates these as orthogonal to vLLM-style PagedAttention (compatible).
---

# Excerpts — Zheng et al. 2024, "SGLang"

## Abstract — frontend + runtime, three optimizations {#abstract}

> Large language models (LLMs) are increasingly used for complex tasks
> that require multiple generation calls, advanced prompting techniques,
> control flow, and structured inputs/outputs. However, efficient
> systems are lacking for programming and executing these applications.
> We introduce SGLang, a system for efficient execution of complex
> language model programs. SGLang consists of a frontend language and a
> runtime. The frontend simplifies programming with primitives for
> generation and parallelism control. The runtime accelerates execution
> with novel optimizations like RadixAttention for KV cache reuse and
> compressed finite state machines for faster structured output
> decoding. Experiments show that SGLang achieves up to 6.4× higher
> throughput compared to state-of-the-art inference systems on various
> large language and multi-modal models on tasks including agent
> control, logical reasoning, few-shot learning benchmarks, JSON
> decoding, retrieval-augmented generation pipelines, and multi-turn
> chat.

## §3 RadixAttention — automatic KV cache reuse {#sec-3}

> SGLang programs can chain multiple generation calls and create
> parallel copies with the "fork" primitive. Additionally, different
> program instances often share some common parts (e.g., system
> prompts). These scenarios create many shared prompt prefixes during
> execution, leading to numerous opportunities for reusing the KV
> cache. … KV cache computation depends only on prefix tokens.
> Therefore, requests with the same prompt prefix can reuse the KV
> cache, reducing redundant computation and memory usage.

> Given the KV cache reuse opportunity, a key challenge in optimizing
> SGLang programs is reusing the KV cache across multiple calls and
> instances. While some systems explore certain KV cache reuse cases,
> they often need manual configurations and cannot handle all reuse
> patterns (e.g., dynamic tree structures). Consequently, most
> state-of-the-art inference systems recompute the KV cache for each
> request.
>
> This section introduces RadixAttention, a novel technique for
> automatic and systematic KV cache reuse during runtime. Unlike
> existing systems that discard the KV cache after a generation request
> finishes, our system retains the cache for prompts and generation
> results in a radix tree, enabling efficient prefix search, reuse,
> insertion, and eviction. We implement an LRU eviction policy and a
> cache-aware scheduling policy to enhance the cache hit rate.
> RadixAttention is compatible with techniques like continuous batching,
> paged attention, and tensor parallelism.

> RadixAttention. A radix tree is a data structure that serves as a
> space-efficient alternative to a classical trie (prefix tree). Unlike
> typical trees, the edges of a radix tree can be labeled not just with
> single elements but also with sequences of elements of varying
> lengths, significantly enhancing efficiency. In our system, we
> utilize a radix tree to manage a mapping between sequences of tokens,
> and their corresponding KV cache tensors. These KV cache tensors are
> stored in a non-contiguous, paged layout, where the size of each
> page is equivalent to one token. Because GPU memory is quickly
> filled by the KV cache, we introduce a simple LRU eviction policy
> that evicts the least recently used leaf first. By evicting leaves
> first, we enable the re-use of their common ancestors until those
> ancestors become leaves and are also evicted.

> In the continuous batching setting, we cannot evict nodes used by
> the currently running batch. Therefore, each node maintains a
> reference counter indicating how many running requests are using it.
> A node is evictable if its reference counter is zero. Note that we
> do not preallocate a fixed-size memory pool as a cache. Instead, we
> let the cached tokens and the currently running requests share the
> same memory pool.

## §3 Cache-aware scheduling — Theorem 3.1 {#sec-3-cache-aware}

> Cache-aware scheduling. We define the cache hit rate as
> $\frac{\text{number of cached prompt tokens}}{\text{number of prompt tokens}}$.
> When there are many requests in the waiting queue, the order in
> which they are executed can significantly impact the cache hit rate.
> … We design a cache-aware scheduling algorithm to increase the
> cache hit rate. … Theorem 3.1. *For a batch of requests, we can
> achieve an optimal cache hit rate by visiting the radix tree of the
> requests in the depth-first search order, with a cache size $\ge$
> the maximum request length. The longest-shared-prefix-first order is
> equivalent to a depth-first search order.*

## §4 Compressed Finite State Machine — multi-token decoding inside fixed text {#sec-4}

> In LM programs, users often want to constrain the model's output to
> follow specific formats, such as JSON schemas. … Existing systems
> support this by converting a regular expression into a finite state
> machine (FSM) (Willard & Louf, 2023). During decoding, they maintain
> the current FSM state, retrieve allowed tokens from the next states,
> and set the probability of invalid tokens to zero, decoding token
> by token. This token-by-token approach, however, is inefficient when
> there are opportunities to decode multiple tokens at once. For
> example, the constant sequence `{"summary": "` in Fig. 2 spans
> multiple tokens in the normal decoding process …, requiring
> multiple decoding stages, even though there is only one valid next
> token when decoding it. … However, existing systems can only decode
> one token at a time because the lack of integration between the FSM
> and the model runner in existing systems prevents multi-token
> processing, resulting in slow decoding.
>
> SGLang overcomes this limitation by creating a fast constrained
> decoding runtime with a compressed FSM. This runtime analyzes the
> FSM and compresses adjacent singular-transition edges in the FSM
> into single edges as demonstrated in Fig. 4 (b), allowing it to
> recognize when multiple tokens can be decoded together.

## §1 Distributed extension {#sec-1-distributed}

> Distributed Cases. RadixAttention can be extended to multiple GPUs.
> For tensor parallelism, each GPU maintains a sharded KV cache. There
> is no need for additional synchronization because the tree
> operations are the same.
