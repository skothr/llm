---
paper_key: ainslie2023
title: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
authors: Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai
year: 2023
venue: EMNLP
arxiv: 2305.13245
local_pdf: theory/sources/papers/ainslie2023_gqa.pdf
type: excerpts
note: Verbatim quotations from the v3 arXiv PDF (Dec 2023). GQA is the standard between-MHA-and-MQA interpolation now used by LLaMA-2/3, Mistral, Qwen, and most decoder-only LLMs from 2023 onward.
---

# Excerpts — Ainslie et al. 2023, "GQA: Training Generalized Multi-Query Transformer Models" (GQA)

## Abstract — the two contributions {#abstract}

> Multi-query attention (MQA), which only uses a single key-value head,
> drastically speeds up decoder inference. However, MQA can lead to
> quality degradation, and moreover it may not be desirable to train a
> separate model just for faster inference. We (1) propose a recipe for
> uptraining existing multi-head language model checkpoints into models
> with MQA using 5% of original pre-training compute, and (2) introduce
> grouped-query attention (GQA), a generalization of multi-query attention
> which uses an intermediate (more than one, less than number of query
> heads) number of key-value heads. We show that uptrained GQA achieves
> quality close to multi-head attention with comparable speed to
> multi-query attention.

## §1 Introduction — why MQA-only is brittle {#sec-1}

> However, multi-query attention (MQA) can lead to quality degradation
> and training instability, and it may not be feasible to train separate
> models optimized for quality and inference. Moreover, while some
> language models already use multi-query attention, such as PaLM
> (Chowdhery et al., 2022), many do not, including publicly available
> language models such as T5 (Raffel et al., 2020) and LLaMA (Touvron et
> al., 2023).

## §2.1 Uptraining {#sec-2-1}

> Generating a multi-query model from a multi-head model takes place in
> two steps: first, converting the checkpoint, and second, additional
> pre-training to allow the model to adapt to its new structure. … The
> projection matrices for key and value heads are mean pooled into single
> projection matrices, which we find works better than selecting a single
> key and value head or randomly initializing new key and value heads
> from scratch.
>
> The converted checkpoint is then pre-trained for a small proportion
> $\alpha$ of its original training steps on the same pre-training recipe.

## §2.2 Grouped-query attention — the GQA-G definition {#sec-2-2}

> Grouped-query attention divides query heads into $G$ groups, each of
> which shares a single key head and value head. GQA-$G$ refers to
> grouped-query with $G$ groups. GQA-1, with a single group and therefore
> single key and value head, is equivalent to MQA, while GQA-$H$, with
> groups equal to number of heads, is equivalent to MHA. Figure 2 shows a
> comparison of grouped-query attention and multi-head/multi-query
> attention. When converting a multi-head checkpoint to a GQA checkpoint,
> we construct each group key and value head by mean-pooling all the
> original heads within that group.
>
> An intermediate number of groups leads to an interpolated model that is
> higher quality than MQA but faster than MHA and, as we will show,
> represents a favorable trade-off. Going from MHA to MQA reduces $H$ key
> and value heads to a single key and value head, reducing the size of the
> key-value cache and therefore amount of data that needs to be loaded by
> a factor of $H$. However, larger models generally scale the number of
> heads, such that multi-query attention represents a more aggressive cut
> in both memory bandwidth and capacity. GQA lets us keep the same
> proportional decrease in bandwidth and capacity as model size increases.
>
> Moreover, larger models suffer relatively less from memory bandwidth
> overhead from attention, as the KV-cache scales with model dimension
> while model FLOPs and parameters scale with the square of model
> dimension. Finally, standard sharding for large models replicates the
> single key and value head by the number of model partitions; GQA
> removes the waste from such partitioning. Therefore, we expect GQA to
> present a particularly good trade-off for larger models.

## §3.1 Experimental setup — what GQA is applied to {#sec-3-1}

> We apply MQA and GQA to decoder self-attention and cross-attention, but
> not encoder self-attention.
>
> We note that GQA is not applied to the encoder self-attention layers;
> encoder representations are computed in parallel, and memory bandwidth
> is therefore generally not the primary bottleneck.

## §3.2 Main results — the headline trade-off {#sec-3-2}

Quoted from Table 1 / Figure 3 captions:

> Uptrained MQA yields a favorable tradeoff compared to MHA with higher
> quality and faster speed than MHA-Large, and GQA achieves even better
> performance with similar speed gains and comparable quality to MHA-XXL.
> Average performance on all tasks as a function of average inference
> time per sample for T5-Large and T5-XXL with multi-head attention, and
> 5% uptrained T5-XXL models with MQA and GQA-8 attention.

## §3.3 Number of groups {#sec-3-3-groups}

> Figure 6 demonstrates the effect of the number of GQA groups on
> inference speed. For larger models the memory bandwidth overhead from
> the KV cache is less constraining (Shazeer, 2019), while the reduction
> in key-value size is sharper due to the increased number of heads. As a
> result, increasing the number of groups from MQA only results in modest
> slowdowns initially, with increasing cost as we move closer to MHA. We
> selected 8 groups as a favorable middle ground.
