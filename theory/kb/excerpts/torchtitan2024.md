---
paper_key: torchtitan2024
title: "TorchTitan: One-Stop PyTorch Native Solution for Production-Ready LLM Pre-training"
authors: Liang, Liu, Wright, Constable, Gu, Huang, Zhang, Feng, Huang, Wang, Purandare, Nadathur, Idreos
year: 2024
venue: ICLR 2025
arxiv: 2410.06511
local_pdf: theory/sources/papers/torchtitan2024.pdf
type: excerpts
note: Verbatim quotes from v3 PDF (7 Jun 2025). The PyTorch-native reference for 4D parallelism (FSDP2 + TP/SP + PP + Context Parallel) plus Float8 and AsyncTP. Stacks all five into composable layers via DTensor.
---

# Excerpts — Liang et al. 2024, "TorchTitan"

## Abstract — what TorchTitan unifies + speedup numbers {#abstract}

> This paper introduces **TorchTitan**, an open-source, PyTorch-native distributed training system that unifies and advances state-of-the-art techniques, streamlining integration and reducing engineering overhead. TorchTitan enables seamless application of 4D parallelism in a modular and composable manner … leveraging cutting-edge features like Float8 training and SymmetricMemory to maximize hardware utilization.

> We thoroughly assess TorchTitan on the Llama 3.1 family of LLMs, spanning 8 billion to 405 billion parameters … we demonstrate accelerations ranging from **65.08% on Llama 3.1 8B at 128 GPU scale (1D)**, **12.59% on Llama 3.1 70B at 256 GPU scale (2D)**, to **30% on Llama 3.1 405B at 512 GPU scale (3D)** on NVIDIA H100 GPUs over optimized baselines.

## §1 The five-axis breakdown {#sec-1-axes}

> To optimize resource utilization and achieve elastic scalability, it is crucial to combine multiple parallelism techniques, including Data Parallel … Tensor Parallel … Context Parallel … and Pipeline Parallel … By stacking these parallelisms with memory and computation optimization techniques, such as activation recomputation … mixed precision training … and deep learning compilers, it is possible to maximize hardware utilization.

## §2.1.1 Meta-device init {#sec-2-1-1-meta}

> TorchTitan enables meta device initialization, where the model is first created on a *meta* device that stores only metadata, making initialization ultra-fast. The model is then sharded into Distributed Tensors (DTensors), with the local shard of each parameter residing on the meta device. Finally, parameter initialization is performed using user-defined functions, ensuring correct DTensor sharding layouts and proper RNG seed usage.

## §2.1.2 FSDP2 advantages over FSDP1 {#sec-2-1-2-fsdp2}

> The original Fully Sharded Data Parallel (FSDP) is an effective implementation of ZeRO that offers large model training capability in PyTorch. However, the original implementation (FSDP1) in PyTorch suffers from various limitations due to its FlatParameter implementation.

> Given these limitations, TorchTitan integrates a new version of Fully Sharded Data Parallel (**FSDP2**), which uses the per-parameter Distributed Tensor sharding representation and thus provides better composability with model parallelism techniques and other features that require the manipulation of individual parameters.

> TorchTitan integrates and leverages FSDP2 as it's default 1D parallelism, benefiting from the improved memory management (often 7 percent lower per GPU memory requirement vs FSDP1) and the slight performance gains (average of 1.5 percent gain vs FSDP1).

## §2.1.3 TP+SP via DTensor {#sec-2-1-3-tp}

> Tensor Parallel (TP) (Narayanan et al., 2021), together with Sequence Parallel (SP) (Korthikanti et al., 2023), is a key model parallelism technique to enable large model training at scale. TP is implemented in TorchTitan using the PyTorch's `RowwiseParallel` and `ColwiseParallel` APIs, where the model parameters are partitioned to DTensors and perform sharded computation with it.

> While TP partitions the most computationally demanding aspects, Sequence Parallel (SP) performs a sharded computation for the normalization or dropout layers on the sequence dimension, which otherwise generate large replicated activation tensors and thus can be challenging to memory constraints per GPU.

## §2.1.4 Pipeline Parallel + ZeroBubble {#sec-2-1-4-pp}

> For large-scale pretraining, TorchTitan employs Pipeline Parallelism (PP), which minimizes communication overhead by leveraging P2P communications. PP divides the model into S stages, each running on a separate group of devices. … the input batch is split into microbatches, and the pipeline schedule overlaps computation and communication across microbatches.

> Recently, TorchTitan added support for new schedules including ZeroBubble and 'Flexible-Interleaved-1F1B', making use of pipeline IR to quickly express new schedules as a list of compute actions and rely on compiler passes to insert and optimize communication actions.

## §2.1.5 Context Parallelism — long context {#sec-2-1-5-cp}

> TorchTitan has been extended to incorporate **Context Parallelism (CP)**, enabling 4D parallelism by adding CP as an additional dimension to existing DP, TP, and PP. CP scales model training by splitting the sequence dimension across GPUs, significantly increasing the maximum trainable context length without causing out-of-memory (OOM) errors. **For example, on Llama 3.1 8B with 8 H100 GPUs, using CP enabled training at context lengths up to 262,144 tokens, achieving minor MFU degradation as CP degree increases.**

## §2.2.3 Async Tensor Parallel {#sec-2-2-3-async-tp}

> By default, TP incurs blocking communications before/after the sharded computations, causing computation resources to not be effectively utilized. **Asynchronous TP (AsyncTP)** (Wang et al., 2022) achieves computation-communication overlap by fractionalizing the TP matrix multiplications within attention and feed-forward modules into smaller chunks, and overlapping communication collectives in between each section. The overlap is achieved by a micro-pipelining optimization, where results are being communicated at the same time that the other chunks of the matmul are being computed.

> PyTorch AsyncTP is based on a `SymmetricMemory` abstraction, which creates intra-node buffers to write faster communication collectives. This is done by allocating a shared memory buffer on each GPU in order to provide direct P2P access.

## §2.2.4 Float8 / mixed precision support {#sec-2-2-4-fp8}

> TorchTitan also supports more advanced mixed precision training with Float8, a derived data type, applied selectively to linear layers (available on newer hardware like NVIDIA H100), achieving substantial performance gains while ensuring training stability … The Float8 feature from `torchao.float8` supports multiple per-tensor scaling strategies, including dynamic, delayed, and static.
