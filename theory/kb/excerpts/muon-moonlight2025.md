---
paper_key: muon-moonlight2025
title: Muon is Scalable for LLM Training
authors: Liu, Su, Yao, Jiang, Lai, Du, Qin, Xu, Lu, Yan, Chen, Zheng, Liu, Yin, He, Zhu, Wang, Wang, Dong, Zhang, Kang, Zhang, Xu, Zhang, Wu, Zhou, Yang
year: 2025
venue: arXiv (Moonshot AI / UCLA tech report)
arxiv: 2502.16982
local_pdf: theory/sources/papers/muon-moonlight2025.pdf
type: excerpts
note: Verbatim quotes from v1 (24 Feb 2025). The first paper to scale Muon (originally Keller Jordan 2024) to LLM-scale training; trains Moonlight, a 3B/16B-MoE on 5.7T tokens, with ~2× AdamW compute efficiency.
---

# Excerpts — Liu et al. 2025, "Muon is Scalable for LLM Training"

## Abstract — two crucial scaling techniques + ~2× efficiency {#abstract}

> Recently, the Muon optimizer (K. Jordan et al. 2024) based on matrix orthogonalization has demonstrated strong results in training small-scale language models, but the scalability to larger models has not been proven. We identify two crucial techniques for scaling up Muon: (1) adding weight decay and (2) carefully adjusting the per-parameter update scale. These techniques allow Muon to work out-of-the-box on large-scale training without the need of hyper-parameter tuning. Scaling law experiments indicate that Muon achieves ~2× computational efficiency compared to AdamW with compute optimal training. Based on these improvements, we introduce Moonlight, a 3B/16B-parameter Mixture-of-Expert (MoE) model trained with 5.7T tokens using Muon.

## §1 Three open challenges, one-line answers {#sec-1-challenges}

> Initial experiments with Muon have demonstrated promising results in small-scale language model training. However, … several critical challenges remain unaddressed: (1) how to effectively scale optimizers based on matrix orthogonalization to larger models with billions of parameters trained with trillions of tokens, (2) how to compute approximate orthogonalization in a distributed setting, and (3) whether such optimizers can generalize across different training stages including pre-training and supervised finetuning (SFT).

## §2.1 Muon update rule (Eq. 1) {#sec-2-1-update-rule}

The update rule of Muon at iteration $t$ for a matrix-shaped weight $\mathbf{W}_{t-1}$, momentum $\mu$, learning rate $\eta_t$, and loss $\mathcal{L}_t$:

$$\begin{aligned}
\mathbf{M}_t &= \mu \mathbf{M}_{t-1} + \nabla \mathcal{L}_t(\mathbf{W}_{t-1}) \\
\mathbf{O}_t &= \mathrm{Newton\text{-}Schulz}(\mathbf{M}_t) \\
\mathbf{W}_t &= \mathbf{W}_{t-1} - \eta_t \mathbf{O}_t
\end{aligned} \tag{1}$$

> Here, $\mathbf{M}_t$ is the momentum of gradient at iteration $t$, set as a zero matrix when $t=0$. In Equation 1, a Newton-Schulz iteration process … is adopted to approximately solve $(\mathbf{M}_t \mathbf{M}_t^\top)^{-1/2} \mathbf{M}_t = \mathbf{L} \mathbf{\Sigma} \mathbf{V}^\top = \mathbf{M}_t$ … which orthogonalizes $\mathbf{M}_t$. Intuitively, orthogonalization can ensure that the update matrices are isomorphic, preventing the weight from learning along a few dominant directions.

## §2.1 Newton-Schulz iteration (Eq. 2) {#sec-2-1-ns}

> Equation 1 is calculated in an iterative process. At the beginning, we set $\mathbf{X}_0 = \mathbf{M}_t / \|\mathbf{M}_t\|_F$. Then, at each iteration $k$, we update $\mathbf{X}_k$ from $\mathbf{X}_{k-1}$ as follows:

$$\mathbf{X}_k = a \mathbf{X}_{k-1} + b (\mathbf{X}_{k-1} \mathbf{X}_{k-1}^\top) \mathbf{X}_{k-1} + c (\mathbf{X}_{k-1} \mathbf{X}_{k-1}^\top)^2 \mathbf{X}_{k-1} \tag{2}$$

> where $\mathbf{X}_N$ is the result of such process after $N$ iteration steps. … In the original design of K. Jordan et al. 2024, the coefficients are set to $a = 3.4445, b = -4.7750, c = 2.0315$.

## §2.1 Norm constraint interpretation {#sec-2-1-norm}

> Bernstein et al. 2024 proposed to view the optimization process in deep learning as steepest descent under norm constraints. From this perspective, we can view the difference between Muon and Adam (Kingma et al. 2015, Loshchilov et al. 2019) as the difference in norm constraints. Whereas Adam is a steepest descent under a norm constraint dynamically adjusted from a Max-of-Max norm, Muon offers a norm constraint that lies in a static range of Schatten-$p$ norm for some large $p$. … In this sense, the norm constraint offered by Muon is more reasonable than that offered by Adam.

## §2.2 Why weight decay is crucial at scale {#sec-2-2-wd}

> While Muon performs significantly better than AdamW on a small scale … we found the performance gains diminish when we scale up to train a larger model with more tokens. We observed that both the weight and the layer output's RMS keep growing to a large scale, exceeding the high-precision range of bf16, which might hurt the model's performance. To resolve this issue, we introduced the standard AdamW … weight decay mechanism into Muon:

$$\mathbf{W}_t = \mathbf{W}_{t-1} - \eta_t (\mathbf{O}_t + \lambda \mathbf{W}_{t-1}) \tag{3}$$

> While vanilla Muon initially converges faster, we observed that some model weights grew too large over time, potentially limiting the model's long-term performances. Adding weight decay addressed this issue — the results demonstrate that Muon with weight decay outperforms both vanilla Muon and AdamW, achieving lower validation loss in the over-train regime.

## §2.2 Per-parameter update-RMS adjustment (Lemma 1, Eq. 4) {#sec-2-2-rms}

**Lemma 1.** *For a full-rank matrix parameter of shape $[A, B]$, its theoretical Muon update RMS is $\sqrt{1/\max(A,B)}$.*

> When $\max(A,B)$ is too large, e.g. the dense MLP matrix, the updates become too small … When $\max(A,B)$ is too small, e.g. treating each KV head in GQA (Shazeer 2019) or MLA (DeepSeek-AI et al. 2024) as a separate parameter, the updates become too large, thus causing training instabilities…

> We propose to match Muon's update RMS to be similar to that of AdamW. From empirical observations, AdamW's update RMS is usually around 0.2 to 0.4. Therefore, we scale Muon's update RMS to this range by the following adjustment:

$$\mathbf{W}_t = \mathbf{W}_{t-1} - \eta_t \!\left(0.2 \cdot \mathbf{O}_t \cdot \sqrt{\max(A,B)} + \lambda \mathbf{W}_{t-1}\right) \tag{4}$$

> We validated this choice with empirical results … Moreover, we highlighted that with this adjustment, Muon can directly **reuse** the learning rate and weight decay tuned for AdamW.

## §2.2 Hybrid use with AdamW {#sec-2-2-hybrid}

> Muon is designed to update matrix-based parameters. In practice, AdamW is used in couple with Muon to handle non-matrix based parameters, like RMSNorm, LM head, and embedding parameters. We would like the optimizer hyper-parameters (learning rate $\eta$, weight decay $\lambda$) to be shared among matrix and non-matrix parameters.

## §2.3 Distributed Muon = ZeRO-1 + DP-gather + Newton-Schulz {#sec-2-3-distributed}

> ZeRO-1 is efficient for AdamW because it calculates updates in an element-wise fashion. However, Muon requires the full gradient matrix to calculate the updates. Therefore, vanilla ZeRO-1 is not directly applicable to Muon. … Distributed Muon follows ZeRO-1 to partition the optimizer states on DP, and introduces two additional operations compared to a vanilla Zero-1 AdamW optimizer:
> 1. **DP Gather.** For a local DP partitioned master weight ($1/DP$ the size of the model weight), this operation is to gather the corresponding partitioned gradients into a full gradient matrix.
> 2. **Calculate Full Update.** After the above gathering, perform Newton-Schulz iteration steps on the full gradient matrix … Note that we will then discard part of the full update matrix, as we only need the partition corresponding to the local parameters to perform update.

## §3.2 Scaling law: 52% FLOPs of AdamW {#sec-3-2-scaling}

> The fitted scaling law curve can be found in figure 3 and the fitted equations are detailed in table 3. As shown in Figure 1a, **Muon only requires about 52% training FLOPs to match the performance of AdamW under compute-optimal setting.**

Table 3 fitted: LM loss = $2.506 \times C^{-0.052}$ (Muon) vs $2.608 \times C^{-0.054}$ (AdamW).

## §3.3 Moonlight architecture & 5.7T-token schedule {#sec-3-3-moonlight}

> To evaluate Muon against contemporary model architectures, we pretrained from scratch using the deepseek-v3-small architecture (DeepSeek-AI et al. 2024) … Our pretrained model has 2.24B activated and 15.29B total parameters (3B activated and 16B total when including embedding).

> 1. **0 to 33B tokens**: In this stage, the learning rate linearly increases to 4.2e-4 in 2k steps. The batch size is kept at 2048 examples;
> 2. **33B to 5.2T tokens**: In this stage, the learning rate decays from 4.2e-4 to 4.2e-5 in a cosine style. We keep the batch size at 2048 until 200B tokens, and then doubled to 4096 for the remaining;
> 3. **5.2T to 5.7T tokens**: In this stage (also referred as the cooldown stage), the learning rate increases to 1e-4 in in 100 steps, and then linearly decays to 0 in 500B tokens, and we keep a constant 4096 batch size. In this stage, we use the highest quality data, focusing on math, code, and reasoning.
