# Glossary additions — stubfill (training, post-training, inference)

Generated 2026-05-04 by the stubfill subagent. Per the subagent
playbook, the orchestrator merges these into `kb/glossary.md`.

## Mixed precision and training stability

- **Master weights** — A high-precision (FP32) copy of the model
  parameters maintained alongside a low-precision (FP16/BF16/FP8)
  working copy used in the forward/backward pass. Optimizer updates
  are applied to the master copy and cast down to the working copy.
  `[micikevicius2017-mixed-precision §2;
  kb/excerpts/micikevicius2017-mixed-precision#sec-2]`.
- **Loss scaling** — Multiplying the loss by a scale factor $S$
  before backprop, then dividing the resulting gradients by $S$
  before the master-weight update. Shifts the gradient distribution
  upward in the FP16 representable range to avoid underflow.
  `[micikevicius2017-mixed-precision §3;
  kb/excerpts/micikevicius2017-mixed-precision#sec-3]`.
- **BF16 (bfloat16)** — A 16-bit floating-point format with 8
  exponent bits and 7 mantissa bits. Same dynamic range as FP32;
  half the precision of FP16. The universal LLM pre-training default
  by 2023. `[kalamkar2019-bfloat16 §2;
  kb/excerpts/kalamkar2019-bfloat16#sec-2]`.
- **FP8 E4M3 / E5M2** — 8-bit float formats with 4-bit
  exponent + 3-bit mantissa (E4M3, range $\pm 448$) or 5-bit
  exponent + 2-bit mantissa (E5M2, larger range, lower precision).
  NVIDIA's hybrid recipe uses E4M3 forward and E5M2 backward;
  DeepSeek-V3 uses E4M3 everywhere with fine-grained tile scaling.
  `[deepseek-v3 §3.3.2;
  kb/excerpts/deepseek-v3-training#sec-3-3-2-mantissa]`.
- **Fine-grained quantization (tile/block scaling)** — Per-group
  scaling factors along inner dimensions of a GEMM, allowing each
  tile/block to choose its own scale and absorb local outliers.
  DeepSeek-V3's 1×128 (activations) and 128×128 (weights) scheme.
  `[deepseek-v3 §3.3.2;
  kb/excerpts/deepseek-v3-training#sec-3-3-2-finegrained]`.
- **Promotion to CUDA cores at $N_C$** — Periodic (every $N_C = 128$
  elements) copy of FP8 Tensor Core partial sums to FP32 registers
  on CUDA cores, where full-precision accumulation is performed.
  Recovers accumulation precision lost to Tensor Core's 14-bit
  internal accumulator. `[deepseek-v3 §3.3.2;
  kb/excerpts/deepseek-v3-training#sec-3-3-2-accum]`.
- **NVFP4** — 4-bit floating-point format (E2M1) with native
  hardware MXScale (32-element per-block scaling) on Blackwell-class
  GPUs. Production-stability for full pre-training is open as of
  2026-05.
- **Loss spike** — A sudden upward jump in pre-training loss, often
  caused by gradient-norm explosion, attention-logit divergence,
  embedding-norm growth, or FP8 underflow. May or may not recover.
  DeepSeek-V3 reports zero irrecoverable spikes over 14.8T tokens
  `[deepseek-v3 abstract;
  kb/excerpts/deepseek-v3-training#abstract]`.
- **QK-LayerNorm / QK-clip** — Stability machinery to prevent
  attention-logit blow-up: normalize $Q$ and $K$ before the dot
  product (QK-LayerNorm, Henry et al. 2020) or clamp $QK^\top$ before
  softmax (QK-clip).

## Adaptation, PEFT, and merging

- **LoRA (Low-Rank Adaptation)** — Parameterizes a frozen weight
  matrix's update as a rank-$r$ product $BA$ with $r \ll
  \min(d_{\text{in}}, d_{\text{out}})$; only $A, B$ are trained. Merges
  to a single weight at inference time with no latency cost.
  `[hu2021-lora §3, §4; kb/excerpts/hu2021-lora#sec-3]`.
- **QLoRA** — LoRA over a 4-bit-quantized (NF4) base model. Adds
  double-quantization and paged optimizers. 65B fine-tune on a single
  48GB GPU. `[dettmers2023-qlora §3;
  kb/excerpts/dettmers2023-qlora#sec-3-nf4]`.
- **NF4 (NormalFloat-4)** — 4-bit quantization with quantile-spaced
  bin centers matched to a standard normal distribution. Information-
  theoretically near-optimal for pretrained weights, which are
  empirically near-Gaussian per block.
  `[dettmers2023-qlora §3.1; kb/excerpts/dettmers2023-qlora#sec-3-nf4]`.
- **DoRA (Direction-Then-Magnitude LoRA)** — LoRA variant decomposing
  $W$ into magnitude and direction; LoRA the direction only. Cited
  via Liu et al. 2024. Often outperforms LoRA at similar parameter
  count.
- **Task vector** — A direction in weight space defined as
  $\tau_T = \theta_T - \theta_0$ where $\theta_0$ is a pretrained base
  and $\theta_T$ is the same base fine-tuned on task $T$. Arithmetic
  on task vectors (addition, negation, combination) steers model
  behavior. `[ilharco2022-task-vectors §2;
  kb/excerpts/ilharco2022-task-vectors#sec-2]`.
- **Model Soup** — A model produced by averaging the weights of $K$
  fine-tunes from a shared base. *Uniform soup* averages all; *greedy
  soup* iteratively includes only fine-tunes that don't decrease
  validation accuracy. `[wortsman2022-model-soups §3;
  kb/excerpts/wortsman2022-model-soups#sec-3]`.
- **TIES-Merging (Trim-Elect-Sign-Disjoint-Merge)** — Three-step model
  merge: trim each task vector to top-$k\%$, elect a sign per
  coordinate by magnitude-weighted majority, disjoint-merge the
  surviving same-sign entries. `[yadav2023-ties-merging §3;
  kb/excerpts/yadav2023-ties-merging#sec-3]`.
- **DARE (Drop And REscale)** — Bernoulli-drop $1-p$ of task-vector
  entries and rescale survivors by $1/p$ before TIES or Soup merging.
  Yu et al. 2024.
- **MergeKit** — Open-source toolkit for model merging supporting
  Linear / SLERP / TIES / DARE / passthrough merges with per-tensor
  configuration. `[mergekit2024]`.
- **Linear mode connectivity (LMC)** — The empirical property that
  fine-tunes of a shared base land in a connected loss basin: the
  path $\theta_0 + t(\theta_f - \theta_0)$, $t \in [0,1]$, stays in
  low-loss territory. Underpins why model merging works.
- **Frankenmerge / passthrough merge** — Layer-wise interleaving of
  layers from two or more models (e.g., copy layers 0–15 from model A
  and 16–31 from model B). Widely used in OSS practice; limited
  primary literature. [FORUM-SIGNAL].

## SFT and post-training data

- **Loss masking (in SFT)** — Setting per-token loss weight to 0 on
  prompt tokens and 1 on response tokens, so only response tokens
  contribute gradient. The standard convention; multi-turn variants
  train on all assistant turns or only the last. `[ouyang2022-
  instructgpt §3; kb/excerpts/ouyang2022-instructgpt#sec-3]`.
- **Rejection sampling (in post-training)** — Sample $N$ responses
  per prompt from the current policy, score with a reward model, keep
  the top-$k$, and SFT on those. The Llama-3 quality gate before each
  preference round. `[meta-llama3]`.
- **MAGPIE (alignment data synthesis from scratch)** — Procedure
  exploiting that an aligned LM, prompted with only its instruction-
  template prefix (no question), will fabricate the question and then
  answer it. Produces $(x,y)$ SFT pairs at zero human cost.
  `[magpie-2024]`.
- **Reasoning-distillation SFT** — Fine-tuning a small base model on
  $(x,y)$ traces sampled from a strong RL-trained reasoner (e.g.,
  R1-Distill: 800K samples from DeepSeek-R1; s1: 1K curated traces
  from Gemini Flash Thinking). Recovers most reasoning capability
  with no RL. `[deepseek-r1; s1-2025]`.
- **Thinking / non-thinking mode SFT** — Training a model to emit
  `<think>...</think>` blocks gated by a system prompt, switching
  reasoning behavior on/off. Qwen 3's 4-stage post-training innovation.
  `[qwen3]`.
- **FLAN (Finetuned Language Net)** — Original instruction-tuning
  recipe: ~1.8K NLP tasks reformatted as natural-language
  instructions, used as an SFT mixture. The pre-RLHF post-training
  baseline. `[wei2021-flan]`. FLAN-T5 `[chung2022-flan-t5]` and the
  FLAN Collection (Longpre et al. 2023, arXiv:2301.13688) extend it.

## RLAIF and Constitutional AI

- **RLAIF (Reinforcement Learning from AI Feedback)** — RLHF pipeline
  where preference labels come from an AI feedback model rather than
  human labelers. The reward model is trained on AI-generated
  preferences via standard Bradley–Terry. `[bai2022-cai §4;
  kb/excerpts/bai2022-cai#sec-4]`.
- **Constitution (in CAI)** — A list of natural-language principles
  ("the assistant should not...") used to condition the AI feedback
  model's preferences. The only human-authored alignment input in
  Constitutional AI. `[bai2022-cai §3;
  kb/excerpts/bai2022-cai#sec-3]`.
- **SL-CAI** — Stage 1 of Constitutional AI: sample a response from
  a helpful-only model, critique under a randomly-chosen principle,
  revise, and SFT on the revisions. Produces $\pi^{\mathrm{SL-CAI}}$.
  `[bai2022-cai §3; kb/excerpts/bai2022-cai#sec-3]`.
- **RL-CAI** — Stage 2 of Constitutional AI: sample response pairs
  from $\pi^{\mathrm{SL-CAI}}$, have an AI judge choose under a
  principle, train a reward model on AI preferences, run PPO with KL
  anchor. `[bai2022-cai §4; kb/excerpts/bai2022-cai#sec-4]`.
- **Non-evasion (CAI behavioral signature)** — A model trained with
  CAI engages with harmful prompts by stating its objections, rather
  than refusing flatly. The defining behavioral signature
  distinguishing CAI from helpful-only RLHF (which complies) and from
  naive safety RLHF (which refuses without explanation).
  `[bai2022-cai §6; kb/excerpts/bai2022-cai#sec-6]`.
- **R-CAI (Reverse CAI)** — Inverts the constitution and applies
  probability-clamped RLAIF to generate controlled adversarial / toxic
  data. `[r-cai-2026]`.
- **Recursive RLAIF** — Using a previously-trained RLAIF model as the
  feedback model for the next round of RLAIF. Risks model collapse in
  the constitution direction.

## Structured output / constrained decoding

- **Constrained decoding** — Generation procedure that, at each step,
  masks the logits to allow only tokens consistent with a target
  grammar (regex, JSON schema, GBNF, CFG). The sampling distribution
  is restricted to $\mathcal{V}_{\text{allowed}}(s_t)$, the FSM's
  allowed-token set in the current state. `[willard2023-outlines §3;
  kb/excerpts/willard2023-outlines#sec-3]`.
- **FSM-vocabulary index (Willard–Louf)** — Precomputed mapping from
  each FSM state to the set of allowed vocabulary tokens. Built once
  per grammar; reduces per-step constraint check from $O(V)$ to
  amortized $O(1)$. `[willard2023-outlines §3;
  kb/excerpts/willard2023-outlines#sec-3]`.
- **Context-independent vs context-dependent tokens** — Partition of
  the vocabulary by whether allowed/disallowed status depends only on
  the parser state (context-independent: precomputable bitmask) or
  also on stack contents (context-dependent: runtime check). XGrammar
  reports ~99% context-independent for typical JSON schemas.
  `[xgrammar2024 §3; kb/excerpts/xgrammar2024#sec-3]`.
- **Compressed FSM (SGLang)** — FSM transformation that merges chains
  of singular-transition edges into single edges, allowing multi-token
  decoding when the grammar deterministically forces a literal
  substring (e.g., `{"summary": "` in a JSON schema).
  `[zheng2024-sglang §4; kb/excerpts/zheng2024-sglang#sec-4]`.
- **GBNF (GGML BNF)** — llama.cpp's grammar specification format.
  An extended BNF used to define constrained-decoding grammars.
- **JSON mode / structured output API** — Vendor-facing API features
  (OpenAI `response_format`, Anthropic tool-use schemas, etc.) that
  guarantee output conforms to a schema. Implemented underneath via
  FSM-constrained decoding. Exact-conformance guarantees depend on
  whether the backend uses true FSM or only logit-bias.
- **JSONSchemaBench** — Benchmark of 10K real-world JSON schemas
  used to compare constrained-decoding engines. `[FORUM-SIGNAL:
  arXiv:2501.10868]`.
