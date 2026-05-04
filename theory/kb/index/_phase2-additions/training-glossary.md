## Optimization

- **AdamW** — Adam with decoupled weight decay. Maintains per-parameter first and second moments $\mathbf{m}_t, \mathbf{v}_t$; update is $-\eta \cdot \mathbf{m}_t / (\sqrt{\mathbf{v}_t}+\varepsilon) - \eta\lambda\mathbf{w}_t$. The default LLM pre-training optimizer 2020-2024. `[hoffmann2022-chinchilla §3]`
- **Muon** — "MomentUm Orthogonalized via Newton-Schulz." Replaces Adam's per-parameter rescaling with a *per-shape* matrix-orthogonalization on momentum buffer $\mathbf{B}_t = \mu \mathbf{B}_{t-1} + \mathbf{G}_t$, then sets $\mathbf{O}_t = \text{Newton-Schulz}(\mathbf{B}_t)$ before the parameter update. Operates only on 2D weight matrices; uses AdamW for 1D parameters and embeddings. `[muon-moonlight2025 §2.1, eq.1; kb/excerpts/muon-moonlight2025#sec-2-1-update-rule]`
- **Newton-Schulz iteration** — Iterative polynomial $\mathbf{X}_k = a\mathbf{X}_{k-1} + b(\mathbf{X}_{k-1}\mathbf{X}_{k-1}^\top)\mathbf{X}_{k-1} + c(\mathbf{X}_{k-1}\mathbf{X}_{k-1}^\top)^2 \mathbf{X}_{k-1}$ that converges all singular values of $\mathbf{X}_0$ to 1, producing $\mathbf{X}_0(\mathbf{X}_0^\top \mathbf{X}_0)^{-1/2}$. Muon uses 5 iterations with coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$. `[muon-moonlight2025 §2.1, eq.2; kb/excerpts/muon-moonlight2025#sec-2-1-ns]`
- **Per-shape RMS scaling (Muon)** — Multiplier $0.2 \cdot \sqrt{\max(A, B)}$ applied to the orthogonalized update for an $A\times B$ weight matrix, equalizing update RMS across heterogeneous matrix shapes. `[muon-moonlight2025 §2.2, eq.4; kb/excerpts/muon-moonlight2025#sec-2-2-rms]`
- **Schatten-$p$ norm** — Matrix norm $\|\mathbf{M}\|_p^{(s)} = (\sum_i \sigma_i^p)^{1/p}$ on singular values. Muon's update is the steepest-descent direction under the Schatten-$\infty$ (spectral) norm constraint, which is why orthogonalization is the right pre-processing step. `[muon-moonlight2025 §2.1]`
- **Lion** — "Evolved sign momentum." Lighter optimizer using only sign of momentum: $\mathbf{w}_t \leftarrow \mathbf{w}_{t-1} - \eta \cdot \text{sign}(\mu \mathbf{m}_{t-1} + (1-\mu)\mathbf{g}_t) - \eta\lambda\mathbf{w}_{t-1}$. Halves optimizer state vs. Adam. (Chen et al. 2023)
- **Sophia** — "Second-order clipped stochastic optimization." Diagonal-Hessian preconditioning with clipping. Targets faster convergence in early pre-training. (Liu et al. 2023, arXiv 2305.14342)
- **Cosine schedule** — Learning-rate decay $\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})(1 + \cos(\pi t / T))$, peaking after warmup and decaying to $\eta_{\min}$ at training end. The de-facto LLM schedule 2020-2024.
- **WSD schedule (Warmup-Stable-Decay)** — Three-phase LR: linear warmup, then constant at $\eta_{\max}$ for the bulk of training, then short cooldown to ~$0.1\eta_{\max}$. Used by Moonlight (5.7T-token Muon run); avoids cosine's commitment to a specific budget. `[muon-moonlight2025 §3.3]`

## Pre-training data

- **Common Crawl (CC)** — Open monthly web crawl run by the Common Crawl Foundation since 2008. Source corpus for FineWeb (96 snapshots), C4, RefinedWeb, and most open web datasets. WARC format with extracted WET (text) variant.
- **trafilatura** — Python text-extraction library; extracts main article text from raw HTML. FineWeb's choice over CC's built-in WET, yielding ~25 % more text. `[fineweb2024 §3.2]`
- **MinHash deduplication** — Locality-sensitive hashing for near-duplicate detection: $n$-gram-shingle the document, sample $H$ hash-of-min-hash signatures arranged as $B$ buckets of $R$ hashes ($H = BR$); two documents collide if any bucket matches; threshold Jaccard $\approx 1 - (1-r^R)^B$ where $r$ is true Jaccard. FineWeb uses 5-grams, $H=112$, $B=14$, $R=8$. `[fineweb2024 §3.4; kb/excerpts/fineweb2024#sec-3-4-minhash]`
- **Individual-snapshot dedup** — FineWeb's choice to MinHash within each CC dump and not across dumps. Counterintuitively beats global dedup; the global version over-removes high-quality multi-snapshot text. `[fineweb2024 §3.4; kb/excerpts/fineweb2024#sec-3-4-global]`
- **Ablation model (FineWeb sense)** — Small reference model (1.71 B params, ~28 B tokens) trained on a candidate corpus to score it. Decisions are made by signed-difference of downstream-task accuracy across candidates. `[fineweb2024 §3.1]`
- **C4 line-level filters** — Per-line heuristics from Raffel et al. 2019 (T5/C4): remove lines lacking terminal punctuation, lines with "javascript", policy/cookie boilerplate, etc. FineWeb adopts these *minus* the curly-brace rule (which would strip code). `[fineweb2024 §3.3]`
- **Custom heuristic filters (FineWeb)** — Three quantile-derived rules: lines-ending-with-terminal-punctuation $\leq 0.12$, char-duplication $\geq 0.01$, short-line-fraction $\geq 0.67$. Lift HellaSwag ~1 pt at ablation scale. `[fineweb2024 §3.6]`
- **FineWeb-Edu** — 1.3 T-token quality-classifier-filtered subset of FineWeb. Llama-3-70B-Instruct annotates 500 K samples 0–5 for "educational value"; classifier learns the score; threshold ≥3 keeps the subset. ~6 K H100-hours of annotation cost. `[fineweb2024 §4]`
- **Data Mixing Law (Eq. 1)** — $L_i(\mathbf{r}) = c_i + k_i \exp(\sum_j t_{ij} r_j)$. Validation loss on domain $i$ as an exponential of a linear combination of training-mixture proportions $r_j$. Coefficients $(c_i, k_i, t_{ij})$ are fit per target domain. `[data-mixing-laws-2024 §1, eq.1]`
- **Nested scaling laws (mixing)** — Pipeline that fits (training-step power law) inside (model-size power law) inside (mixing-law exponential), enabling small-model small-step experiments to predict large-model end-of-training loss as a function of mixture. `[data-mixing-laws-2024 §1]`
- **Compatibility / symmetry principles (mixing law)** — The two structural constraints that pin down Eq. 1 from a candidate-function space: reduces to two-domain log-linear when $M=2$; invariant under re-labeling of domains. `[data-mixing-laws-2024 §3.2]`
- **Transfer affinity coefficient $t_{ij}$** — Fitted parameter in the data mixing law; sign indicates whether domain-$j$ training helps ($t_{ij}<0$) or hurts ($t_{ij}>0$) validation on domain $i$. `[data-mixing-laws-2024 §3.2]`

## Synthetic data & distillation

- **Synthetic pre-training data** — Documents generated by a teacher LLM (or LLM pipeline) for use in a student's pre-training mix. Phi-4 generated ~400 B tokens via 50 pipelines, ≈40 % of its 9.8 T-token training mix. `[phi4 §1, §2]`
- **Phi-4 50-pipeline architecture** — Multi-stage pipeline: seed curation → rewrite/augment → self-revision → quality filter. 50 distinct pipelines tuned for different content types (STEM, code, reasoning chains). `[phi4 §2.2; kb/excerpts/phi4#sec-2-2-pipelines]`
- **R1 distillation set** — DeepSeek-R1's 800 K SFT samples (600 K reasoning + 200 K non-reasoning) generated by R1 and used to fine-tune Qwen-2.5/Llama-3 students at 1.5/7/14/32/70 B. Transfers reasoning *capability* to architectures that cannot reach the same level via RLVR alone. `[deepseek-r1 §3, §4; kb/excerpts/deepseek-r1#sec-3-pipeline]`
- **Decontamination (Phi-4 sense)** — Held-out test sets, especially November 2024 AMC math contests and unleaked HumanEval+, used to verify the synthetic data did not leak benchmark answers via the teacher. `[phi4 §1.1]`
- **Spoonfeeding (Phi-4 intuition)** — `[INTUITION]` Synthetic data presents *concept-then-application-then-explanation* as a self-contained unit, vs. organic web data which scatters those parts across many documents. Enables high-density learning per token. `[phi4 §1, §2.1]`

## Distributed training (axes)

- **Data Parallel (DP)** — Same parameters on each worker; each worker processes a shard of the global batch; gradients all-reduced before optimizer step.
- **FSDP** — "Fully Sharded Data Parallel." Native PyTorch ZeRO implementation. Parameters and optimizer state sharded across DP workers; full param-set materialized via all-gather just before each forward/backward. `[torchtitan2024 §2.1.2]`
- **FSDP2** — Per-parameter `DTensor`-sharded successor to FSDP1. ~7 % per-GPU memory reduction and ~1.5 % throughput improvement vs. FSDP1's flat-parameter design. `[torchtitan2024 §2.1.2; kb/excerpts/torchtitan2024#sec-2-1-2-fsdp2]`
- **ZeRO-1 / ZeRO-2 / ZeRO-3** — Optimizer-state / gradient / parameter sharding levels of Rajbhandari et al.'s Zero-Redundancy Optimizer. ZeRO-1 shards only optimizer state; ZeRO-3 shards everything (FSDP's default).
- **Tensor Parallel (TP)** — Within-layer matmul split (column-parallel `Y = XA`, row-parallel `Z = YB`). Inserts one all-reduce per matmul. Each worker holds a $1/t$ slice of weights. `[torchtitan2024 §2.1.3]`
- **Sequence Parallel (SP)** — Companion to TP: shards normalization and dropout layers along the sequence dimension to reduce activation memory. `[torchtitan2024 §2.1.3]`
- **Pipeline Parallel (PP)** — Stage-level layer split. Microbatches flow stage-to-stage via P2P send/recv. Tradeoff: per-GPU param drops by $1/p$ but in-flight activations grow.
- **1F1B (one-forward-one-backward)** — Classic PP schedule. Each stage alternates forward-of-microbatch-$i$ with backward-of-microbatch-$j$ to bound peak activation memory.
- **Pipeline bubble** — Fraction of PP step time where some stage is idle, waiting for upstream/downstream. 1F1B bubble is $(p-1)(F+B)$ stage-times. `[deepseek-v3 §3.2.1, table]`
- **ZeroBubble (ZB1P)** — PP schedule that splits backward into "backward-for-input" ($B$) and "backward-for-weights" ($W$); reorders $W$ into the bubble. Reduces bubble to $(p-1)(F+B-2W)$. `[deepseek-v3 §3.2.1]`
- **DualPipe** — DeepSeek-V3's bidirectional PP. Microbatches feed from both ends of the pipeline simultaneously; each chunk split into 4 stages (attention / dispatch / MLP / combine), backward further split via ZB; achieves $(\tfrac{p}{2}-1)(F\&B+B-3W)$ bubble at $2\times$ parameter memory and $+1$ activation. `[deepseek-v3 §3.2.1; kb/excerpts/deepseek-v3-training#sec-3-2-1-dualpipe]`
- **Context Parallel (CP)** — Token-position sharding of the sequence dimension. Communication-pattern resembles ring attention. Enables 262 144-token training of Llama 3.1 8B on 8× H100. `[torchtitan2024 §2.1.5]`
- **Expert Parallel (EP)** — MoE-specific sharding: each worker hosts a subset of the experts. Forward pass requires an all-to-all dispatch (each token to its top-$k$ experts) and a combine all-to-all. DeepSeek-V3 uses $e=64$ across 8 nodes. `[deepseek-v3 §3.2; kb/excerpts/deepseek-v3-training#sec-3-2-parallelism]`
- **All-to-all dispatch / combine** — The two collectives bracketing every MoE forward. Dispatch routes tokens to expert hosts; combine returns expert outputs to original token positions. The dominant communication cost in MoE training.
- **AsyncTP (Asynchronous Tensor Parallel)** — Fractionalizes TP matmuls into chunks and overlaps each chunk's communication with the previous chunk's compute via intra-node `SymmetricMemory` shared buffers. `[torchtitan2024 §2.2.3]`
- **DTensor / DeviceMesh (PyTorch)** — Distributed tensor abstraction; weights are stored as DTensor with a placement spec on a $d \times t \times p \times c \times e$ device mesh. The composability layer for 4D/5D parallelism. `[torchtitan2024 §2.1.1]`
- **4-node-cap rule (DeepSeek-V3 routing)** — Each token's MoE dispatch is limited to at most 4 nodes, exploiting the 3.2× NVLink/IB bandwidth ratio: one IB hop per (token, target-node) followed by intra-node NVLink forwarding. `[deepseek-v3 §3.2.2; kb/excerpts/deepseek-v3-training#sec-3-2-2-allreduce]`
- **MFU (Model FLOP Utilization)** — Realized training throughput as a fraction of peak hardware FLOPS. Standard parallelism-plan quality metric.

## Mixed precision & stability

- **BF16 (bfloat16)** — Brain-float. 8-bit exponent, 7-bit mantissa (vs. FP16's 5-exp/10-mant). Same dynamic range as FP32; lower precision. The 2020-2024 default forward/backward dtype.
- **FP8 E4M3** — 4-bit exponent, 3-bit mantissa. Higher precision, lower dynamic range than E5M2. DeepSeek-V3 uses E4M3 on *all* tensors via fine-grained scaling. `[deepseek-v3 §3.3.2; kb/excerpts/deepseek-v3-training#sec-3-3-2-mantissa]`
- **FP8 E5M2** — 5-bit exponent, 2-bit mantissa. Higher dynamic range, lower precision. NVIDIA's recommended dtype for `Dgrad`/`Wgrad` in their hybrid recipe; DeepSeek-V3 *avoids* it.
- **Tile-wise quantization (1×128)** — DeepSeek-V3's activation scaling: one FP8 scale per (token, 128-channel block) tile. Adapts to per-token outliers. `[deepseek-v3 §3.3.2; kb/excerpts/deepseek-v3-training#sec-3-3-2-finegrained]`
- **Block-wise quantization (128×128)** — DeepSeek-V3's weight scaling: one FP8 scale per 128×128 weight block. `[deepseek-v3 §3.3.2]`
- **FP32 promotion to CUDA cores** — DeepSeek-V3's accumulation strategy: every $N_C=128$ MMA steps, copy partial sums from Tensor-Core (~14-bit) accumulator to FP32 registers on CUDA cores for full-precision summation. Recovers ~2 % systematic error in raw FP8 GEMM. `[deepseek-v3 §3.3.2; kb/excerpts/deepseek-v3-training#sec-3-3-2-accum]`
- **Precision pinning (DeepSeek-V3)** — Components held at BF16/FP32 even with FP8 elsewhere: embedding module, output head, MoE gating, normalization, attention. `[deepseek-v3 §3.3.1]`
- **Loss spike** — Sudden non-recoverable loss-curve excursion. Causes include gradient-norm explosion, attention-logit divergence, embedding-norm growth, and FP8 underflow on rare tokens. DeepSeek-V3 reports zero such spikes in 14.8 T tokens. `[deepseek-v3 §abstract]`

## Adaptation & merging

- **Continual / late-stage pre-training** — Continuing training of an already-trained model on a different (typically higher-quality or domain-specific) data mixture. Phi-4's "midtrain," DeepSeek-V3's 119 K-H800-hour context-extension stage, OLMo 2's Dolmino mix.
- **Context extension** — Sub-stage of late-stage pre-training that increases trained sequence length (e.g., 4 K → 32 K → 128 K) by adjusting RoPE scaling and continuing on long-context data. `[deepseek-v3 §1]`
- **PEFT (Parameter-Efficient Fine-Tuning)** — Family of fine-tuning methods that update only a small fraction of parameters: LoRA, DoRA, IA3, prefix-tuning. Pending excerpts.
- **Model merging** — Combining trained models via parameter-level operations (averaging, interpolation, sign-resolution, sparsification). Subtypes include weight averaging (Model Soups), task arithmetic, TIES, DARE, evolutionary merging. Pending excerpts.
- **Linear mode connectivity** — The empirical finding that fine-tunes of a shared base land in a connected loss basin: averaging their weights gives a model with loss similar to the endpoints. The leading explanation for why merging works at all. (Frankle, Dziugaite et al. 2020 — pending excerpt.)
