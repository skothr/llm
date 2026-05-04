## Position encoding

- **Sinusoidal positional encoding** — Vaswani 2017's deterministic sin/cos position vectors; additive at the input. `[vaswani2017 §3.5]`
- **Learned absolute positional embedding** — Per-position trainable vectors indexed by integer position; hard cutoff at maximum sequence length. Used by GPT-2, BERT. `[su2021 §2.2]`
- **Relative position encoding (T5 bias)** — Per-(query, key) distance bias added to attention logits; binned by relative distance. `[su2021 §2.3]`
- **ALiBi** — Attention with Linear Biases. Parameter-free additive bias $-|m-n| \cdot s_h$ at attention logits, with per-head slope. Used by BLOOM, MPT. (Press et al. 2021, arXiv 2108.12409)
- **Rotary Position Embedding (RoPE)** — Multiplicative position encoding that rotates query and key in 2D subspaces by per-frequency angles $m\theta_i$. Inserts position information as relative offsets only via rotation-matrix orthogonality. The de-facto standard in modern decoder-only LLMs. `[su2021 §3.2]`
- **RoPE base frequency $b$** — The base of the geometric frequency progression $\theta_i = b^{-2(i-1)/d}$. Default 10000 (su2021); LLaMA-3 uses 500000; long-context models use 1e6+. Larger $b$ stretches the longest "naturally-seen" wavelength.
- **NTK-aware RoPE scaling** — Per-frequency RoPE rescaling that scales the base $b$ rather than positions, leaving high-frequency dimensions unchanged. Empirically extends context length without degrading short-context quality.
- **YaRN** — "Yet another RoPE extensioN method." Combines NTK-aware scaling with a per-frequency interpolation schedule (no scaling on short-wavelength dims, full scaling on long-wavelength dims, smooth ramp between) and an attention-temperature rescale. ~10× more token-efficient than position interpolation. (Peng et al. 2023, arXiv 2309.00071)
- **LongRoPE / LongRoPE2** — Evolutionary search over per-frequency RoPE scaling factors; allows non-uniform rescaling no analytical method produces. LongRoPE2 reports near-lossless 128K extension at 80× fewer tokens than Meta's earlier approach. (Ding et al. 2024 / 2025)
- **NoPE** — "No Position Encoding." Decoder-only Transformer trained without any explicit positional encoding; relies on the causal mask to recover position information implicitly. Better length generalization on synthetic tasks, but underperforms on realistic LLM workloads. (Kazemnejad et al. 2023)
- **iRoPE** — "interleaved RoPE." LLaMA 4's position scheme: alternates RoPE-equipped attention layers with NoPE attention layers (3:1 ratio) plus temperature scaling in the NoPE layers. Trained at 256K, claimed to extrapolate to 10M-token inference. `[meta-llama4]`
- **M-RoPE** — Multi-axis RoPE for vision-language models. Splits the per-head dimension $d_h$ into temporal/height/width thirds; each third gets its own RoPE rotation. Introduced by Qwen2-VL.
- **Interleaved-MRoPE** — Qwen3-VL refinement of M-RoPE that interleaves the temporal/height/width axes per dimension-pair instead of partitioning the channel range; every dimension-pair sees all three axes.

## Mixture of Experts (MoE)

- **MoE FFN sublayer** — A Transformer FFN sublayer replaced by $E$ parallel FFN "experts" plus a router that selects $k \ll E$ experts per token. Compute scales with $k$, parameter count with $E$. Universal at frontier scale 2024–2026. `[fedus2021-switch §1, mixtral2024]`
- **Router (gate)** — Linear projection $W_g \in \mathbb{R}^{d_{\text{model}} \times E}$ producing per-token expert logits $\boldsymbol{g}_t = W_g \boldsymbol{x}_t$.
- **Top-$k$ routing (token-choice)** — Each token selects its top-$k$ experts by router logit. Switch uses $k=1$, Mixtral $k=2$, DeepSeek-V3 / Qwen3 $k=8$. The dominant routing mode in production. `[fedus2021-switch §2; mixtral2024]`
- **Expert-choice routing** — Inverse: each expert selects its top-$T$ tokens. Guarantees uniform load by construction but breaks autoregressive property; rarely used in production decoder-only LLMs. (Zhou et al. 2022, arXiv 2202.09368)
- **Shared experts ($E_s$)** — "Always-on" experts that process every token, alongside $k$ routed experts. DeepSeek-V2/V3 use $E_s = 1$ or $2$; Qwen3 uses $E_s = 0$. Argument: shared experts capture common knowledge, freeing routed experts to specialize. `[deepseekmoe2024 §3.2]`
- **Fine-grained expert segmentation** — Use many narrow experts ($E$ in hundreds, narrow $d_{\text{ff}}$ each) instead of few wide ones, increasing combinatorial expressiveness $\binom{E}{k}$ of the active set. DeepSeekMoE pattern, adopted by V2/V3 and Qwen3. `[deepseekmoe2024 §3.1]`
- **Expert capacity factor $C$** — Bound on tokens-per-expert per batch: $\lceil C \cdot Tk/E \rceil$. Tokens routed past the limit bypass the MoE layer (residual passthrough). $C \in [1, 2]$ typical. `[fedus2021-switch §2.2]`
- **Load-balancing auxiliary loss** — Regularizer added to the LM loss to push routing toward uniformity. Switch's form is $\mathcal{L}_{\text{aux}} = \alpha E \sum_e f_e P_e$ with $f_e$ = fraction of tokens routed to $e$ and $P_e$ = mean router prob for $e$. `[fedus2021-switch §2.2]`
- **Auxiliary-loss-free load balancing** — DeepSeek-V3's strategy: add a per-expert bias to router logits (not to gating weights), update bias online based on observed load. Removes the historical quality regression of high-coefficient aux loss. `[deepseek-v3 §2]`
- **Global-batch load balancing** — Qwen3's strategy: apply load-balancing loss across the entire training mini-batch (cluster-wide), not within microbatches. Compensates for dropping shared experts. `[qwen3]`
- **Routing collapse** — Pathological MoE training failure where the router routes most tokens to a few "favored" experts; the rest go cold. Self-reinforcing without auxiliary balancing.

## State-space models and hybrids

- **State-space model (SSM)** — Sequence model with recurrent state $\boldsymbol{h}_t$ updated linearly: $\boldsymbol{h}_t = \bar{A} \boldsymbol{h}_{t-1} + \bar{B} u_t$. Per-token compute $O(d^2)$, sequence-linear. (Gu, Goel, Ré 2022)
- **HiPPO** — "High-order Polynomial Projection Operator." Specific structured choice of $A$ matrix that optimally compresses input history under polynomial bases; foundation for S4. (Gu et al. 2020)
- **S4** — Structured State Space sequence model with HiPPO-NPLR $A$, FFT-based convolutional training, recurrent inference. Dual-form architecture. (Gu, Goel, Ré 2022)
- **Selective SSM (Mamba)** — Mamba's contribution: input-dependent $\bar{B}, C, \Delta$ (parameters produced by linear projection of current input), enabling content-aware selection. Breaks convolutional dual; uses parallel scan for training-time parallelism. `[gu2023-mamba]`
- **Parallel scan** — GPU-efficient associative-scan algorithm used by Mamba to compute the input-dependent recurrence in $O(N \log N)$ during training while the recurrence is sequential. (Blelloch 1990; Mamba uses Smith et al. 2022's hardware-aware variant.)
- **State Space Duality (SSD)** — Mamba-2's framing: selective SSMs and a structurally-restricted form of attention compute the same function. The restriction: attention matrix is lower-triangular semi-separable with rank $\le d$. `[mamba2]`
- **Semi-separable matrix** — A structured matrix admitting low-rank off-diagonal blocks; the structural class that encompasses Mamba-2's effective attention matrix.
- **Hybrid SSM/Transformer** — Architecture interleaving SSM blocks and attention blocks. Three patterns: serial (Jamba 1:7), shared-attention (Zamba), parallel-in-block (Hymba). The deployed form for production-scale SSM-using models.
- **Recall bottleneck** — SSM weakness: fixed-size state $\boldsymbol{h}_t$ cannot store every prior token's exact features, so exact-retrieval tasks (copy, NIAH) underperform attention. Primary motivation for hybrids.

## Long context

- **Sliding-window attention** — Each position attends only to the previous $w$ tokens; compute $O(N w d)$ vs $O(N^2 d)$. Local-only by construction; relies on stacked layers' growing receptive field for long-range dependencies. (Longformer 2020, Mistral 2023)
- **Native Sparse Attention (NSA)** — Three-branch sparse attention (compress + select + sliding-window) trained natively (not retrofit). GQA-group-aligned blockwise selection enables FlashAttention-style kernels under sparsity. `[yuan2025 §3.2–3.4]`
- **Token compression branch (NSA)** — Block-summarizes a window of $l$ keys/values into a single representation via learned MLP. Provides coarse global attention. `[yuan2025 §3.3.1 Eq.7]`
- **Token selection branch (NSA)** — Picks top-$n$ blocks by importance scores derived from the compression branch's attention scores. Provides fine-grained attention. `[yuan2025 §3.3.2 Eq.8–12]`
- **Sliding window branch (NSA)** — Dedicated local-window branch that absorbs local-pattern learning so the compression and selection branches focus on long-range. `[yuan2025 §3.3.3]`
- **Native sparse training** — Pretraining with a sparse attention pattern from the start, vs retrofitting sparsity onto a dense-attention pretrained model. NSA's signature design choice. `[yuan2025 §2.2]`
- **Compressive memory (Infini-Attention)** — Linear-attention-style $d \times d$ memory matrix updated by outer-product accumulation over the entire prefix; read by cross-attention from local-window queries. (Munkhdalai et al. 2024, arXiv 2404.07143)
- **Ring Attention** — Distributed exact attention: shard sequence across $D$ devices, rotate $K, V$ blocks around the ring, overlap communication with computation. Per-device memory $O(N/D)$; max sequence scales with device count. (Liu, Zaharia, Abbeel 2023, arXiv 2310.01889)
- **Context parallelism** — Production realization of Ring-Attention-style sharding for inference. Yang et al. 2024 reports 93% parallelization efficiency at 1M context, LLaMA 405B, 128 H100s.
- **Position interpolation (PI)** — Simplest RoPE extension: scale positions by $s = L_{\text{test}} / L_{\text{train}}$. Cheap, degrades short-context quality. (Chen et al. 2023)

## FFN

- **GLU (Gated Linear Unit)** — $\mathrm{GLU}(x, W, V) = \sigma(xW) \otimes (xV)$, the elementwise product of a sigmoid-gated and an identity-gated linear projection. Building block for FFN-GLU variants. `[shazeer2020 §2 Eq.4]`
- **SwiGLU** — $\mathrm{FFN}_{\mathrm{SwiGLU}}(x) = (\mathrm{Swish}_1(xW) \otimes xV) W_2$. Three-matrix FFN with Swish gating. Universal in modern decoder-only LLMs. `[shazeer2020 §2 Eq.6]`
- **GEGLU** — GLU with GELU activation: $(\mathrm{GELU}(xW) \otimes xV) W_2$. Equivalent quality to SwiGLU; used in some Google models (T5.1.1, GLaM, mT5). `[shazeer2020 §2 Eq.6]`
- **Bias-free FFN** — T5-style FFN omitting bias terms; SwiGLU is bias-free in production. `[shazeer2020 §1 Eq.2]`
- **2/3 parameter-equalization rule** — When converting a 2-matrix FFN to a 3-matrix GLU-variant, reduce $d_{\text{ff}}$ by factor 2/3 to keep parameter count constant. Gives the modern $d_{\text{ff}} = \tfrac{8}{3} d_{\text{model}}$ rule (rounded for hardware). `[shazeer2020 §2]`

## Reasoning architecture

- **Reasoning-capable LLM** — A base Transformer LLM trained (typically by RLVR) to emit extended chain-of-thought traces between special delimiters before producing a final answer. Architecturally minor (special tokens + format); training-protocol heavy. `[deepseek-r1, qwen3]`
- **Think tokens / `<think>...</think>`** — Reserved special tokens delimiting the reasoning region of a generated output. Reserved as single tokens (not multi-token UTF-8 sequences). `[deepseek-r1, qwen3]`
- **Two-checkpoint reasoning** — DeepSeek's pattern: distinct base (V3) and reasoning (R1) checkpoints. Cleanly separates concerns; doubles deployment.
- **Single-checkpoint hybrid reasoning** — Qwen3's pattern: one model trained to operate in either thinking or non-thinking mode based on a system-prompt flag. Amortizes deployment but trades off specialty.
- **Thinking budget** — Serving-time cap on reasoning-token count before forcing answer commit. The trained model must recover from a forcibly-truncated reasoning trace without hallucinating that the trace was complete. (Qwen3)
- **RLVR (Reinforcement Learning with Verifiable Rewards)** — Training method for reasoning models. Reward = +1 if a programmatic verifier (math grader, code executor, formal checker) accepts the answer, 0 otherwise. The reasoning trace is the policy's free choice. Used by DeepSeek-R1, R1-Zero, Qwen3-Reasoning. See `kb/notes/post-training/rlvr-and-grpo.md`.
- **R1-Zero pattern** — Pure RL from base (no cold-start SFT). Develops reasoning behavior emergently but produces less-readable traces. DeepSeek's ablation result.

## Multi-token prediction

- **Multi-token prediction (MTP)** — Training objective + architectural feature: in addition to the standard next-token head, the model has auxiliary heads predicting tokens at offsets $t+2, t+3, \ldots, t+k$. Used by DeepSeek-V3. `[deepseek-v3 §2.3]`
- **MTP module** — Small Transformer block on top of the backbone that produces the prediction for token $t+i$ given backbone hidden state $\boldsymbol{h}_t^L$. DeepSeek-V3 uses one MTP module ($k=2$ total: main next-token + one MTP step ahead).
- **Self-speculative decoding (via MTP)** — Inference mode where MTP heads emit draft tokens that the main head verifies in parallel. ~1.8× speedup at ~85% acceptance in DeepSeek-V3 serving.
