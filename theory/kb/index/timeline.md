# LLM Architecture Timeline

Chronological progression of LLM theory and practice from the Transformer
through the 2026 frontier. Each entry cites one or more papers from
`kb/index/papers.json`. Entries are grouped by half-year for readability.

## Reading conventions

- **Year-Half:** entries clustered by chronological half (2017-H1, 2017-H2, …).
- **Shift:** one-line description of what changed.
- **Sources:** `[paper-key]` — these resolve to entries in
  `theory/kb/index/papers.json`.
- **Notes:** pointer(s) to the full digested treatment under
  `kb/notes/<area>/<topic>.md`.
- `[CONTRADICTION]` markers indicate items where sources disagree.
- The timeline is a **finding-aid**, not the authoritative narrative — the
  notes carry the math and the citations.

## Pre-2017 prequel

Encoder-decoder seq2seq with attention (Bahdanau 2014; Luong 2015) and
sub-word tokenization (Sennrich 2016 BPE) are the immediate prerequisites
for the Transformer. They are referenced from notes but not enumerated as
timeline entries because the timeline focuses on Transformer-era LLMs.

## 2017

### 2017-H2 — The Transformer

- **Shift:** Replace recurrence with attention; enable parallel sequence
  processing.
- **Sources:** `[vaswani2017]`
- **Notes:** `kb/notes/architecture/attention-mechanism.md`,
  `kb/notes/architecture/transformer-overview.md`,
  `kb/notes/architecture/position-encoding.md`,
  `kb/notes/architecture/ffn.md`,
  `kb/notes/architecture/normalization.md`.
- The encoder-decoder skeleton, scaled dot-product + multi-head attention,
  sinusoidal position encoding, and Pre-LN/Post-LN debate all originate here.

## 2018

### 2018-H1 — Decoder-only generative pre-training

- **Shift:** GPT-1 — pre-train a decoder-only Transformer with LM
  objective, fine-tune on downstream tasks.
- **Sources:** `[radford2018]`
- **Notes:** `kb/notes/architecture/transformer-overview.md`.

### 2018-H2 — Bidirectional encoder pre-training

- **Shift:** BERT — masked LM + next-sentence-prediction; encoder-only.
  Sets the encoder track parallel to the GPT decoder track.
- **Sources:** `[devlin2019-bert]`
- **Notes:** `kb/notes/architecture/transformer-overview.md`.

## 2019

### 2019-H1 — Scaling LM up

- **Shift:** GPT-2 — show 1.5B-param decoder-only LM has emergent zero-shot
  task transfer.
- **Sources:** `[radford2019-gpt2]`
- **Notes:** `kb/notes/architecture/transformer-overview.md`,
  `kb/notes/scaling/scaling-frontier.md`.

### 2019-H2 — Multi-Query Attention (MQA)

- **Shift:** Shazeer proposes sharing K/V across attention heads,
  identifying KV-cache reload as the dominant decoding cost.
- **Sources:** `[shazeer2019]`
- **Notes:** `kb/notes/architecture/attention-mechanism.md`,
  `kb/notes/inference/kv-cache-management.md`.

## 2020

### 2020-H1 — Kaplan scaling laws

- **Shift:** First systematic scaling-law paper. Power-law fit between
  loss, parameters, dataset size, compute. Recommends N ↑ faster than D.
- **Sources:** `[kaplan2020]`
- **Notes:** `kb/notes/scaling/kaplan-laws.md`.
- `[CONTRADICTION]`: Kaplan's recommendation is partially superseded by
  Chinchilla (2022), which showed Kaplan over-allocated to params at
  fixed compute. See `kaplan-laws.md §4`.

### 2020-H2 — GPT-3 and few-shot learning

- **Shift:** 175B parameter LM; few-shot in-context learning emerges
  without gradient updates.
- **Sources:** `[brown2020]`
- **Notes:** `kb/notes/architecture/transformer-overview.md`,
  `kb/notes/scaling/scaling-frontier.md`.

## 2021

### 2021-H1 — Rotary Position Embeddings (RoPE)

- **Shift:** Replace additive sinusoidal positions with rotation applied
  to Q/K vectors at every layer; relative-position encoded by inner
  product. Becomes the de-facto standard.
- **Sources:** `[su2021]`
- **Notes:** `kb/notes/architecture/position-encoding.md`,
  `kb/notes/architecture/long-context.md`.

### 2021-H2 — GLU activation variants

- **Shift:** Shazeer surveys GLU/GEGLU/SwiGLU variants for FFN; SwiGLU
  becomes the modern standard (LLaMA, etc.).
- **Sources:** `[shazeer2020]`
- **Notes:** `kb/notes/architecture/ffn.md`.

## 2022

### 2022-H1 — Chinchilla compute-optimal scaling

- **Shift:** Train Chinchilla 70B on 1.4T tokens — show compute-optimal
  is roughly D ≈ 20 N (params and tokens scale equally with compute),
  contradicting Kaplan.
- **Sources:** `[hoffmann2022-chinchilla]`
- **Notes:** `kb/notes/scaling/chinchilla.md`.

### 2022-H1 — Chain-of-Thought prompting

- **Shift:** Prompting LMs to "think step-by-step" elicits reasoning;
  emergent at scale.
- **Sources:** `[wei2022-cot]`, `[kojima2022]`
- **Notes:** `kb/notes/reasoning/chain-of-thought.md`.

### 2022-H1 — μP (μTransfer)

- **Shift:** Hyperparameter transfer: tune at small width, transfer
  optimal LR/init to wide model. Backed by Tensor Programs theory.
- **Sources:** `[yang2022-mup]`
- **Notes:** `kb/notes/scaling/mu-transfer.md`.

### 2022-H1 — InstructGPT and RLHF

- **Shift:** PPO-based RL from human preferences becomes the canonical
  alignment recipe. SFT → reward model → PPO three-stage pipeline.
- **Sources:** `[ouyang2022-instructgpt]`
- **Notes:** `kb/notes/post-training/rlhf.md`,
  `kb/notes/post-training/sft.md`.

### 2022-H1 — FlashAttention

- **Shift:** IO-aware exact attention via tiling + recomputation;
  changes the asymptotic HBM-access count from Θ(N²) to Θ(N²d²/M).
  Cuts wall-clock without changing the math.
- **Sources:** `[dao2022]`
- **Notes:** `kb/notes/architecture/attention-mechanism.md` §3.2.

### 2022-H2 — ChatGPT release; RLHF goes mainstream

- **Shift:** Productionization of InstructGPT-class models for general
  consumption; not a new paper, but the inflection that drives the
  2023-2024 alignment / post-training research wave.
- **Sources:** none (product release).

### 2022-H2 — Constitutional AI

- **Shift:** RLAIF — replace human preference labels with model-generated
  critiques against a written constitution.
- **Sources:** `[bai2022-cai]`
- **Notes:** `kb/notes/post-training/rlaif-and-constitutional.md`.

## 2023

### 2023-H1 — GPT-4

- **Shift:** Closed multimodal frontier model; technical report
  describes capabilities but withholds architecture.
- **Sources:** referenced via OpenAI tech report.
- **Notes:** `kb/notes/scaling/scaling-frontier.md`.

### 2023-H1 — LLaMA / LLaMA-2

- **Shift:** First broadly-available open-weights frontier-class
  models. Sets the open-source recipe (RoPE, RMSNorm, SwiGLU, GQA in
  LLaMA-2).
- **Sources:** `[touvron2023]`
- **Notes:** `kb/notes/architecture/transformer-overview.md`.

### 2023-H1 — Process Reward Models (PRM800k)

- **Shift:** Step-level supervision in mathematical reasoning; outcome-
  vs process-supervision distinction.
- **Sources:** `[lightman2023-prm800k]`
- **Notes:** `kb/notes/reasoning/process-supervision.md`.

### 2023-H2 — Grouped-Query Attention (GQA)

- **Shift:** Interpolate between MHA and MQA via query-head groups;
  becomes universal for decoder-only LLMs (LLaMA-2/3, Mistral, Qwen).
- **Sources:** `[ainslie2023]`
- **Notes:** `kb/notes/architecture/attention-mechanism.md` §3.1.

### 2023-H2 — Direct Preference Optimization (DPO)

- **Shift:** Cast preference learning as a closed-form policy update
  on the SFT reference; eliminate explicit reward model + PPO.
- **Sources:** `[rafailov2023-dpo]`
- **Notes:** `kb/notes/post-training/dpo-and-offline.md`.

### 2023-H2 — FlashAttention-2

- **Shift:** Re-partition FA1 work to maximize matmul time on tensor
  cores; ~2× over FA1 on A100.
- **Sources:** `[dao2023]`
- **Notes:** `kb/notes/architecture/attention-mechanism.md` §3.2.

### 2023-H2 — vLLM / PagedAttention

- **Shift:** OS-style virtual memory for the KV cache; eliminates
  fragmentation, enables high-throughput batched serving.
- **Sources:** `[kwon2023]`
- **Notes:** `kb/notes/inference/kv-cache-management.md`,
  `kb/notes/inference/serving-systems.md`.

### 2023-H2 — Watermarking and provenance

- **Shift:** Kirchenbauer et al. propose green-list/red-list logit-bias
  watermarking for LLM-generated text.
- **Sources:** `[kirchenbauer2023-watermark]`
- **Notes:** `kb/notes/alignment/watermarking-and-provenance.md`.

### 2023-H2 — GPTQ post-training quantization

- **Shift:** Layer-wise OBQ-style 4-bit quantization with Cholesky
  reformulation; preserves quality at 4-bit.
- **Sources:** `[frantar2022]`
- **Notes:** `kb/notes/inference/quantization.md`.

### 2023-H2 — Speculative decoding

- **Shift:** Use a small draft model to propose tokens; verify with
  the large model. Same output distribution, ~2-3× wall-clock speedup.
- **Sources:** `[leviathan2023]`
- **Notes:** `kb/notes/inference/speculative-decoding.md`.

## 2024

### 2024-H1 — Mamba and selective SSMs

- **Shift:** Selective state-space models match Transformer perplexity
  with linear-in-N inference. Hybrid attention+SSM models (Jamba, Zamba)
  follow.
- **Sources:** `[gu2023-mamba]`, `[mamba2]`
- **Notes:** `kb/notes/architecture/state-space-models.md`.

### 2024-H1 — Multi-Head Latent Attention (MLA)

- **Shift:** DeepSeek-V2 introduces low-rank KV joint compression with
  decoupled-RoPE; matches MHA quality at GQA-2-class cache size.
- **Sources:** `[deepseek-v2]`
- **Notes:** `kb/notes/architecture/attention-mechanism.md` §3.1.

### 2024-H1 — DeepSeekMoE / fine-grained + shared experts

- **Shift:** MoE design with many small experts plus a shared expert per
  layer; reduces routing collapse and enables fine-grained specialization.
- **Sources:** `[deepseek-v2]` (MoE part), `[deepseekmoe2024]`
- **Notes:** `kb/notes/architecture/moe.md`.

### 2024-H2 — Test-time compute scaling

- **Shift:** Snell et al. argue inference-time compute is a co-equal
  scaling axis with training compute. Empirically: small model + lots
  of search ≈ much larger model with greedy decoding.
- **Sources:** `[snell2024]`
- **Notes:** `kb/notes/scaling/inference-time-compute-scaling.md`,
  `kb/notes/reasoning/test-time-compute.md`.

### 2024-H2 — o1 / reasoning model paradigm

- **Shift:** OpenAI's o1 popularizes "thinking-first" generation: long
  chain-of-thought as a separate decoding phase before the answer.
- **Sources:** OpenAI o1 system card; tech-report-only.
- **Notes:** `kb/notes/architecture/reasoning-architectures.md`,
  `kb/notes/reasoning/reasoning-training.md`.

### 2024-H2 — JumpReLU SAEs and SAE scaling

- **Shift:** SAE quality and pareto-front improvements (TopK, JumpReLU);
  Anthropic Gemma-Scope releases per-layer SAE artifacts.
- **Sources:** `[rajamanoharan2024-jumprelu]`,
  `[gao2024-topk-saes]`,
  `[gemma-scope-2024]`.
- **Notes:** `kb/notes/interpretability/sparse-autoencoders.md`.

### 2024-H2 — FineWeb / data-quality classifiers

- **Shift:** 15T-token open dataset; FineWeb-Edu classifier gives a 3rd
  recipe for synthetic / filtered data alongside Phi (textbooks-style)
  and R1 (distillation). Global vs. individual-snapshot dedup
  `[CONTRADICTION]`.
- **Sources:** `[fineweb2024]`
- **Notes:** `kb/notes/training/pre-training-data.md`.

### 2024-H2 — Process Reward Models go mainstream

- **Shift:** PRM-based ranking + best-of-N sampling becomes the
  inference-time-search workhorse for math reasoning.
- **Sources:** `[lightman2023-prm800k]`, `[wang2024-mathshepherd]`,
  `[rstar-math2025]`.
- **Notes:** `kb/notes/reasoning/inference-time-search.md`.

### 2024-Q4 — Alignment-faking / scheming empirically demonstrated

- **Shift:** Anthropic's alignment-faking paper and Apollo's scheming
  evaluations demonstrate threat models that were previously theoretical.
- **Sources:** `[greenblatt2024-alignment-faking]`,
  `[meinke2024-apollo-scheming]`.
- **Notes:** `kb/notes/alignment/scheming-and-deceptive-alignment.md`.

## 2025

### 2025-H1 — DeepSeek-V3 / R1 / RLVR

- **Shift:** DeepSeek-V3 (671B-MoE; 2.788M H800-hr; $5.576M reported
  training cost) and R1 (RLVR — RL with verifiable rewards via GRPO,
  no human preferences in the inner loop). R1-Zero AIME 2024:
  15.6% → 77.9% (cons@64 = 86.7%) from base alone.
  `[CONTRADICTION]`: capability ceiling — Yue et al. argue RLVR cannot
  expand base capability; R1-Distill ablations suggest distillation
  carries reasoning beyond what RL reaches alone.
- **Sources:** `[deepseek-v3]`, `[deepseek-r1]`, `[shao2024]` (GRPO),
  `[dapo2025]`.
- **Notes:** `kb/notes/post-training/rlvr-and-grpo.md`,
  `kb/notes/architecture/reasoning-architectures.md`.

### 2025-H1 — FlashAttention-3 (H100)

- **Shift:** Asynchronous warp-specialized scheduling, FP8 support,
  H100 TMA engine. ~1.5–2× over FA2 on H100.
- **Sources:** `[shah2024]`
- **Notes:** `kb/notes/architecture/attention-mechanism.md` §3.2.

### 2025-H1 — Native Sparse Attention (NSA)

- **Shift:** DeepSeek team — sparsity is part of pre-training, not
  retrofitted. Hardware-aligned, end-to-end trainable. Targets long
  context.
- **Sources:** `[yuan2025]`
- **Notes:** `kb/notes/architecture/long-context.md`.

### 2025-H1 — Muon optimizer

- **Shift:** Newton-Schulz orthogonalization on momentum; Moonlight
  16B-MoE × 5.7T tokens claims ~52% fewer FLOPs vs AdamW at matched
  loss.
- **Sources:** `[muon-moonlight2025]`
- **Notes:** `kb/notes/training/optimization.md`.

### 2025-H1 — s1 / token-budget-forcing

- **Shift:** Show that simple test-time-compute interventions
  (truncate, append "Wait", append "Final Answer") on small models
  recover most of o1-class reasoning gains.
- **Sources:** `[s1-2025]`
- **Notes:** `kb/notes/reasoning/test-time-compute.md`.

### 2025-H2 — Frontier reasoning lineup turnover

- **Shift:** o3, Claude Opus 4 / Sonnet 4 / Haiku 4, Gemini 2.5,
  Qwen3 — dense reasoning-model release cadence. Earlier reasoning
  models (o1) considered superseded.
- **Sources:** vendor system cards.
- **Notes:** `kb/notes/scaling/scaling-frontier.md`.

### 2025-H2 — Benchmarks: GPQA Diamond saturation; AIME 2024 contamination

- **Shift:** Frontier crosses 90% on GPQA Diamond; AIME 2024 widely
  considered contaminated; AIME 2025 / MathArena are the current open
  targets. ARC-AGI-1 effectively solved by o3; ARC-AGI-2 succeeds.
- **Sources:** `[rein2023-gpqa]`, `[matharena-2025]`,
  `[arc-agi-2-2025]`.
- **Notes:** `kb/notes/evaluation/reasoning-benchmarks.md`.

### 2025-H2 — Anthropic circuit-tracing framework

- **Shift:** Attribution-graph framework for circuit tracing on
  production models (Haiku); operationally distinct from
  activation-patching.
- **Sources:** Anthropic transformer-circuits.pub posts (Mathematical
  Framework, Scaling Monosemanticity, Circuit Tracing, Biology).
- **Notes:** `kb/notes/interpretability/circuit-tracing.md`.

## 2026 (cutoff: 2026-05)

### 2026-H1 — Continued reasoning-model and inference-time-compute work

- **Shift:** R1-distill recipes proliferate; verifiable-reward
  workflows extend beyond math (code, agentic tools); RLVR limits
  debate continues. Differential Transformer
  (Ye et al. 2024 / ICLR 2025) gathers replications.
- **Sources:** `[differential-transformer-2024]` (if added),
  ongoing system cards.
- **Notes:** `kb/notes/architecture/attention-mechanism.md` §5,
  `kb/notes/post-training/rlvr-and-grpo.md`.

## Bridges (chronologically off-axis)

A few items don't fit neatly into one half-year because they span
multiple periods or were finalized in journal venues after their
arXiv preprint. They are documented in the relevant note rather than
re-listed here:

- **Layer-norm placement (Pre-LN vs Post-LN):** Xiong et al. 2020;
  Peri-LN rehabilitation 2024-2025. See `kb/notes/architecture/normalization.md`.
- **Mixture-of-Experts lineage:** Shazeer 2017 (sparse gating) →
  Switch Transformer 2021 → GShard → DeepSeekMoE 2024. See
  `kb/notes/architecture/moe.md`.
- **DPO offline-RL family:** DPO 2023 → IPO, KTO, ORPO, SimPO 2024.
  See `kb/notes/post-training/dpo-and-offline.md`.

## Maintenance

This timeline is updated:
- After each Phase 2 area subagent run (forward-fills new entries).
- During Phase 4 cross-link audits (back-fills bridges, fixes errors).
- When a stale-prior is corrected, the timeline entry gets a tag
  pointing at the correction note.
