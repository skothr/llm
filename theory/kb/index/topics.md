# KB Topic Graph (v1 — refined after Phase 1 landscape sweep)

Refined 2026-05-03 from the v0 hypothesis using Phase 1 findings (see
`theory/plans/2026-05-03-landscape-sweep-findings.md` for the full
synthesis). Phase 2 may further refine as note files are written.

## v0 → v1 changelog

- **Split** `architecture/ffn-and-moe` → `architecture/ffn` + `architecture/moe`
  (MoE is now its own subfield with routing, fine-grained experts, shared
  experts, load balancing variants — universal at frontier).
- **Split** `alignment/sycophancy-and-deception` → `alignment/sycophancy` +
  `alignment/scheming-and-deceptive-alignment` (distinct threat models;
  empirically demonstrated by Anthropic + Apollo Dec 2024).
- **Renamed** `training/continued-pretraining` → `training/adaptation-and-merging`
  (scope now equally covers CPT, domain adaptation, model merging).
- **Added** `architecture/reasoning-architectures` (thinking-token / hybrid
  think-non-think paradigm; architectural side of reasoning models).
- **Added** `architecture/multi-token-prediction` (DeepSeek V3 architectural
  feature; distinct from speculative decoding).
- **Added** `scaling/inference-time-compute-scaling` (Snell 2024 + R1
  paradigm; co-equal scaling axis with training compute).
- **Added** `interpretability/circuit-tracing` (Anthropic 2025 attribution-
  graph framework; operationally distinct from `activation-patching`).

**Net change:** 9 areas, 48 → 53 leaf topics (+5).

Held for Phase 2 review (not applied yet — may handle with sub-sections
inside the note rather than splitting): SAE → 3 sub-leaves, SSM →
pure+hybrid, agentic-benchmarks → 3, speculative-decoding → draft+model-free,
rlvr-and-grpo → foundations + variants.

## Status legend

- `pending` — not yet researched
- `stub` — note file created with placeholder
- `draft` — first pass written, not reviewed
- `reviewed` — claims spot-checked, citations verified
- `locked` — frozen for the current research cycle

## Areas

### architecture/

| Topic | Status | Note path |
|-------|--------|-----------|
| transformer-overview | draft | `kb/notes/architecture/transformer-overview.md` |
| tokenization | draft | `kb/notes/architecture/tokenization.md` |
| embeddings-and-tying | draft | `kb/notes/architecture/embeddings-and-tying.md` |
| position-encoding | draft | `kb/notes/architecture/position-encoding.md` |
| attention-mechanism | draft | `kb/notes/architecture/attention-mechanism.md` |
| ffn | draft | `kb/notes/architecture/ffn.md` |
| moe | draft | `kb/notes/architecture/moe.md` |
| normalization | draft | `kb/notes/architecture/normalization.md` |
| state-space-models | draft | `kb/notes/architecture/state-space-models.md` |
| long-context | draft | `kb/notes/architecture/long-context.md` |
| multimodal-llm-extensions | draft | `kb/notes/architecture/multimodal-llm-extensions.md` |
| reasoning-architectures | draft | `kb/notes/architecture/reasoning-architectures.md` |
| multi-token-prediction | draft | `kb/notes/architecture/multi-token-prediction.md` |

### training/

| Topic | Status | Note path |
|-------|--------|-----------|
| pre-training-data | draft | `kb/notes/training/pre-training-data.md` |
| synthetic-data-and-distillation | draft | `kb/notes/training/synthetic-data-and-distillation.md` |
| optimization | draft | `kb/notes/training/optimization.md` |
| distributed-training | draft | `kb/notes/training/distributed-training.md` |
| mixed-precision-and-stability | draft | `kb/notes/training/mixed-precision-and-stability.md` |
| adaptation-and-merging | draft | `kb/notes/training/adaptation-and-merging.md` |

### post-training/

| Topic | Status | Note path |
|-------|--------|-----------|
| sft | draft | `kb/notes/post-training/sft.md` |
| rlhf | draft | `kb/notes/post-training/rlhf.md` |
| dpo-and-offline | draft | `kb/notes/post-training/dpo-and-offline.md` |
| rlaif-and-constitutional | draft | `kb/notes/post-training/rlaif-and-constitutional.md` |
| rlvr-and-grpo | draft | `kb/notes/post-training/rlvr-and-grpo.md` |

### reasoning/

| Topic | Status | Note path |
|-------|--------|-----------|
| chain-of-thought | draft | `kb/notes/reasoning/chain-of-thought.md` |
| process-supervision | draft | `kb/notes/reasoning/process-supervision.md` |
| test-time-compute | draft | `kb/notes/reasoning/test-time-compute.md` |
| reasoning-training | draft | `kb/notes/reasoning/reasoning-training.md` |
| inference-time-search | draft | `kb/notes/reasoning/inference-time-search.md` |

### inference/

| Topic | Status | Note path |
|-------|--------|-----------|
| kv-cache-management | draft | `kb/notes/inference/kv-cache-management.md` |
| quantization | draft | `kb/notes/inference/quantization.md` |
| speculative-decoding | draft | `kb/notes/inference/speculative-decoding.md` |
| serving-systems | draft | `kb/notes/inference/serving-systems.md` |
| structured-output | draft | `kb/notes/inference/structured-output.md` |

### scaling/

| Topic | Status | Note path |
|-------|--------|-----------|
| kaplan-laws | draft | `kb/notes/scaling/kaplan-laws.md` |
| chinchilla | draft | `kb/notes/scaling/chinchilla.md` |
| mu-transfer | draft | `kb/notes/scaling/mu-transfer.md` |
| scaling-frontier | draft | `kb/notes/scaling/scaling-frontier.md` |
| inference-time-compute-scaling | draft | `kb/notes/scaling/inference-time-compute-scaling.md` |

### interpretability/

| Topic | Status | Note path |
|-------|--------|-----------|
| lens-techniques | draft | `kb/notes/interpretability/lens-techniques.md` |
| mechanistic-interpretability | draft | `kb/notes/interpretability/mechanistic-interpretability.md` |
| sparse-autoencoders | draft | `kb/notes/interpretability/sparse-autoencoders.md` |
| activation-patching | draft | `kb/notes/interpretability/activation-patching.md` |
| probing | draft | `kb/notes/interpretability/probing.md` |
| circuit-tracing | draft | `kb/notes/interpretability/circuit-tracing.md` |

### evaluation/

| Topic | Status | Note path |
|-------|--------|-----------|
| knowledge-benchmarks | draft | `kb/notes/evaluation/knowledge-benchmarks.md` |
| reasoning-benchmarks | draft | `kb/notes/evaluation/reasoning-benchmarks.md` |
| agentic-benchmarks | draft | `kb/notes/evaluation/agentic-benchmarks.md` |
| eval-methodology | draft | `kb/notes/evaluation/eval-methodology.md` |

### alignment/

| Topic | Status | Note path |
|-------|--------|-----------|
| safety-evaluation | draft | `kb/notes/alignment/safety-evaluation.md` |
| watermarking-and-provenance | draft | `kb/notes/alignment/watermarking-and-provenance.md` |
| oversight-and-scalable-alignment | draft | `kb/notes/alignment/oversight-and-scalable-alignment.md` |
| sycophancy | draft | `kb/notes/alignment/sycophancy.md` |
| scheming-and-deceptive-alignment | draft | `kb/notes/alignment/scheming-and-deceptive-alignment.md` |

## Total

- 9 areas
- 53 leaf topics

## Phase 2 input

Per-topic deep research (Phase 2) will populate the corresponding
`kb/notes/<area>/<topic>.md` for each leaf above. Candidate paper
inventory is in `theory/kb/index/papers.json` (expanded post-Phase 1)
and the per-area landscape reports under
`theory/plans/landscape-sweep/`.
