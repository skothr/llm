---
paper_key: deepseek-v3
title: DeepSeek-V3 Technical Report (training-side excerpts)
authors: DeepSeek-AI
year: 2024
venue: arXiv (DeepSeek Tech Report)
arxiv: 2412.19437
local_pdf: theory/sources/papers/deepseek-v3.pdf
type: excerpts
note: |
  Verbatim quotes from v2 PDF (18 Feb 2025), focused on the training-infrastructure
  half (FP8 mixed precision, DualPipe, all-to-all kernels). The architecture-side
  (MLA, DeepSeekMoE, MTP) is excerpted in `kb/excerpts/deepseek-v2.md` for the
  V2 versions and discussed in `kb/notes/architecture/`. Section anchors below
  are for the V3 PDF.
---

# Excerpts — DeepSeek-AI 2024, "DeepSeek-V3 Technical Report" (training)

## Abstract — what V3 introduces {#abstract}

> We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with **671B total parameters with 37B activated for each token**. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an **auxiliary-loss-free strategy for load balancing** and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on **14.8 trillion** diverse and high-quality tokens … DeepSeek-V3 requires only **2.788M H800 GPU hours** for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks.

## §1 Training-cost table {#sec-1-cost}

| Stage | H800 GPU Hours | USD |
|---|---|---|
| Pre-Training | 2664K | $5.328M |
| Context Extension | 119K | $0.238M |
| Post-Training | 5K | $0.01M |
| **Total** | **2788K** | **$5.576M** |

> Assuming the rental price of H800 is $2 per GPU hour … training DeepSeek-V3 on each trillion tokens requires only 180K H800 GPU hours, i.e., 3.7 days on our cluster with 2048 H800 GPUs. Consequently, our pre-training stage is completed in less than two months and costs 2664K GPU hours.

## §3.1 Cluster {#sec-3-1-cluster}

> DeepSeek-V3 is trained on a cluster equipped with **2048 NVIDIA H800 GPUs**. Each node in the H800 cluster contains 8 GPUs connected by NVLink and NVSwitch within nodes. Across different nodes, InfiniBand (IB) interconnects are utilized to facilitate communications.

## §3.2 Parallelism strategy {#sec-3-2-parallelism}

> DeepSeek-V3 applies **16-way Pipeline Parallelism (PP), 64-way Expert Parallelism (EP) spanning 8 nodes, and ZeRO-1 Data Parallelism (DP)**. … In order to facilitate efficient training of DeepSeek-V3, we implement meticulous engineering optimizations. Firstly, we design the DualPipe algorithm for efficient pipeline parallelism. Compared with existing PP methods, DualPipe has fewer pipeline bubbles. More importantly, it overlaps the computation and communication phases across forward and backward processes, thereby addressing the challenge of heavy communication overhead introduced by cross-node expert parallelism. Secondly, we develop efficient cross-node all-to-all communication kernels to fully utilize IB and NVLink bandwidths and conserve Streaming Multiprocessors (SMs) dedicated to communication. Finally, we meticulously optimize the memory footprint during training, thereby enabling us to train DeepSeek-V3 without using costly Tensor Parallelism (TP).

## §3.2.1 DualPipe — communication hiding via overlap {#sec-3-2-1-dualpipe}

> For DeepSeek-V3, the communication overhead introduced by cross-node expert parallelism results in an inefficient computation-to-communication ratio of approximately 1:1. To tackle this challenge, we design an innovative pipeline parallelism algorithm called **DualPipe**, which not only accelerates model training by effectively overlapping forward and backward computation-communication phases, but also reduces the pipeline bubbles.

> The key idea of DualPipe is to overlap the computation and communication within a pair of individual forward and backward chunks. To be specific, we divide each chunk into four components: `attention`, `all-to-all dispatch`, `MLP`, and `all-to-all combine`. Specially, for a backward chunk, both `attention` and `MLP` are further split into two parts, `backward for input` and `backward for weights`, like in ZeroBubble.

> we rearrange these components and manually adjust the ratio of GPU SMs dedicated to communication versus computation. In this overlapping strategy, we can ensure that both all-to-all and PP communication can be fully hidden during execution. … It employs a **bidirectional pipeline scheduling**, which feeds micro-batches from both ends of the pipeline simultaneously and a significant portion of communications can be fully overlapped.

## §3.2.1 Pipeline-bubble comparison {#sec-3-2-1-bubble}

| Method | Bubble | Parameter | Activation |
|---|---|---|---|
| 1F1B | $(PP-1)(F+B)$ | $1\times$ | $PP$ |
| ZB1P | $(PP-1)(F+B-2W)$ | $1\times$ | $PP$ |
| DualPipe (Ours) | $(\tfrac{PP}{2}-1)(F\&B+B-3W)$ | $2\times$ | $PP+1$ |

> compared with ZB1P (Qi et al. 2023b) and 1F1B (Harlap et al. 2018), DualPipe significantly reduces the pipeline bubbles while only increasing the peak activation memory by $\frac{1}{PP}$ times.

## §3.2.2 Cross-node all-to-all kernels {#sec-3-2-2-allreduce}

> in our cluster, cross-node GPUs are fully interconnected with IB, and intra-node communications are handled via NVLink. NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s). To effectively leverage the different bandwidths of IB and NVLink, **we limit each token to be dispatched to at most 4 nodes**, thereby reducing IB traffic. For each token, when its routing decision is made, it will first be transmitted via IB to the GPUs with the same in-node index on its target nodes. Once it reaches the target nodes, we will endeavor to ensure that it is instantaneously forwarded via NVLink to specific GPUs that host their target experts, without being blocked by subsequently arriving tokens. In this way, communications via IB and NVLink are fully overlapped, and each token can efficiently select an average of 3.2 experts per node without incurring additional overhead from NVLink.

## §3.3 FP8 mixed precision framework {#sec-3-3-fp8}

> we propose a fine-grained mixed precision framework utilizing the FP8 data format for training DeepSeek-V3. While low-precision training holds great promise, it is often limited by the presence of outliers in activations, weights, and gradients. … To address this challenge and effectively extend the dynamic range of the FP8 format, we introduce a **fine-grained quantization strategy: tile-wise grouping with $1 \times N_c$ elements or block-wise grouping with $N_c \times N_c$ elements**. The associated dequantization overhead is largely mitigated under our increased-precision accumulation process …

> compared with the BF16 baseline, the **relative loss error of our FP8-training model remains consistently below 0.25%**, a level well within the acceptable range of training randomness.

## §3.3.1 Mixed precision framework {#sec-3-3-1-mixed}

> Firstly, in order to accelerate model training, the majority of core computation kernels, i.e., GEMM operations, are implemented in FP8 precision. These GEMM operations accept FP8 tensors as inputs and produce outputs in BF16 or FP32. As depicted in Figure 6, all three GEMMs associated with the `Linear` operator, namely `Fprop` (forward pass), `Dgrad` (activation backward pass), and `Wgrad` (weight backward pass), are executed in FP8. **This design theoretically doubles the computational speed compared with the original BF16 method.**

> Despite the efficiency advantage of the FP8 format, certain operators still require a higher precision due to their sensitivity to low-precision computations. … For this reason, after careful investigations, we maintain the original precision (e.g., BF16 or FP32) for the following components: the **embedding module, the output head, MoE gating modules, normalization operators, and attention operators**. … To further guarantee numerical stability, we store the master weights, weight gradients, and optimizer states in higher precision.

## §3.3.2 Fine-grained quantization (1×128 act, 128×128 weight) {#sec-3-3-2-finegrained}

> we propose a fine-grained quantization method that applies scaling at a more granular level. As illustrated in Figure 7 (a), (1) for activations, we group and scale elements on a **1x128 tile basis (i.e., per token per 128 channels)**; and (2) for weights, we group and scale elements on a **128x128 block basis** (i.e., per 128 input channels per 128 output channels). This approach ensures that the quantization process can better accommodate outliers by adapting the scale according to smaller groups of elements.

> One key modification in our method is the introduction of **per-group scaling factors along the inner dimension of GEMM operations**. This functionality is not directly supported in the standard FP8 GEMM. However, combined with our precise FP32 accumulation strategy, it can be efficiently implemented.

## §3.3.2 Increasing accumulation precision {#sec-3-3-2-accum}

> we observe that the accumulation precision of FP8 GEMM on NVIDIA H800 GPUs is **limited to retaining around 14 bits**, which is significantly lower than FP32 accumulation precision. … Taking GEMM operations of two random matrices with K = 4096 for example, in our preliminary test, the limited accumulation precision in Tensor Cores results in a maximum relative error of nearly 2%.

> In order to address this issue, we adopt the strategy of **promotion to CUDA Cores for higher precision**. … during MMA (Matrix Multiply-Accumulate) execution on Tensor Cores, intermediate results are accumulated using the limited bit width. Once an interval of $N_C$ is reached, these partial results will be copied to FP32 registers on CUDA Cores, where full-precision FP32 accumulation is performed. … Based on our experiments, **setting $N_C = 128$ elements, equivalent to 4 WGMMAs, represents the minimal accumulation interval** that can significantly improve precision without introducing substantial overhead.

## §3.3.2 E4M3 on all tensors {#sec-3-3-2-mantissa}

> In contrast to the hybrid FP8 format adopted by prior work (NVIDIA 2024b; Peng et al. 2023b; Sun et al. 2019b), which uses E4M3 (4-bit exponent and 3-bit mantissa) in `Fprop` and E5M2 (5-bit exponent and 2-bit mantissa) in `Dgrad` and `Wgrad`, **we adopt the E4M3 format on all tensors for higher precision**. We attribute the feasibility of this approach to our fine-grained quantization strategy, i.e., tile and block-wise scaling. By operating on smaller element groups, our methodology effectively shares exponent bits among these grouped elements, mitigating the impact of the limited dynamic range.

## §3.3.3 Low-precision storage {#sec-3-3-3-storage}

> **Low-Precision Optimizer States.** We adopt the BF16 data format instead of FP32 to track the first and second moments in the AdamW optimizer, without incurring observable performance degradation. However, the master weights (stored by the optimizer) and gradients (used for batch size accumulation) are still retained in FP32 to ensure numerical stability throughout training.
