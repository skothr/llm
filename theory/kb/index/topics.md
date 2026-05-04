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
| transformer-overview | pending | `kb/notes/architecture/transformer-overview.md` |
| tokenization | pending | `kb/notes/architecture/tokenization.md` |
| embeddings-and-tying | pending | `kb/notes/architecture/embeddings-and-tying.md` |
| position-encoding | pending | `kb/notes/architecture/position-encoding.md` |
| attention-mechanism | draft | `kb/notes/architecture/attention-mechanism.md` |
| ffn | pending | `kb/notes/architecture/ffn.md` |
| moe | pending | `kb/notes/architecture/moe.md` |
| normalization | pending | `kb/notes/architecture/normalization.md` |
| state-space-models | pending | `kb/notes/architecture/state-space-models.md` |
| long-context | pending | `kb/notes/architecture/long-context.md` |
| multimodal-llm-extensions | pending | `kb/notes/architecture/multimodal-llm-extensions.md` |
| reasoning-architectures | pending | `kb/notes/architecture/reasoning-architectures.md` |
| multi-token-prediction | pending | `kb/notes/architecture/multi-token-prediction.md` |

### training/

| Topic | Status | Note path |
|-------|--------|-----------|
| pre-training-data | pending | `kb/notes/training/pre-training-data.md` |
| synthetic-data-and-distillation | pending | `kb/notes/training/synthetic-data-and-distillation.md` |
| optimization | pending | `kb/notes/training/optimization.md` |
| distributed-training | pending | `kb/notes/training/distributed-training.md` |
| mixed-precision-and-stability | pending | `kb/notes/training/mixed-precision-and-stability.md` |
| adaptation-and-merging | pending | `kb/notes/training/adaptation-and-merging.md` |

### post-training/

| Topic | Status | Note path |
|-------|--------|-----------|
| sft | pending | `kb/notes/post-training/sft.md` |
| rlhf | pending | `kb/notes/post-training/rlhf.md` |
| dpo-and-offline | pending | `kb/notes/post-training/dpo-and-offline.md` |
| rlaif-and-constitutional | pending | `kb/notes/post-training/rlaif-and-constitutional.md` |
| rlvr-and-grpo | pending | `kb/notes/post-training/rlvr-and-grpo.md` |

### reasoning/

| Topic | Status | Note path |
|-------|--------|-----------|
| chain-of-thought | pending | `kb/notes/reasoning/chain-of-thought.md` |
| process-supervision | pending | `kb/notes/reasoning/process-supervision.md` |
| test-time-compute | pending | `kb/notes/reasoning/test-time-compute.md` |
| reasoning-training | pending | `kb/notes/reasoning/reasoning-training.md` |
| inference-time-search | pending | `kb/notes/reasoning/inference-time-search.md` |

### inference/

| Topic | Status | Note path |
|-------|--------|-----------|
| kv-cache-management | pending | `kb/notes/inference/kv-cache-management.md` |
| quantization | pending | `kb/notes/inference/quantization.md` |
| speculative-decoding | pending | `kb/notes/inference/speculative-decoding.md` |
| serving-systems | pending | `kb/notes/inference/serving-systems.md` |
| structured-output | pending | `kb/notes/inference/structured-output.md` |

### scaling/

| Topic | Status | Note path |
|-------|--------|-----------|
| kaplan-laws | pending | `kb/notes/scaling/kaplan-laws.md` |
| chinchilla | pending | `kb/notes/scaling/chinchilla.md` |
| mu-transfer | pending | `kb/notes/scaling/mu-transfer.md` |
| scaling-frontier | pending | `kb/notes/scaling/scaling-frontier.md` |
| inference-time-compute-scaling | pending | `kb/notes/scaling/inference-time-compute-scaling.md` |

### interpretability/

| Topic | Status | Note path |
|-------|--------|-----------|
| lens-techniques | pending | `kb/notes/interpretability/lens-techniques.md` |
| mechanistic-interpretability | pending | `kb/notes/interpretability/mechanistic-interpretability.md` |
| sparse-autoencoders | pending | `kb/notes/interpretability/sparse-autoencoders.md` |
| activation-patching | pending | `kb/notes/interpretability/activation-patching.md` |
| probing | pending | `kb/notes/interpretability/probing.md` |
| circuit-tracing | pending | `kb/notes/interpretability/circuit-tracing.md` |

### evaluation/

| Topic | Status | Note path |
|-------|--------|-----------|
| knowledge-benchmarks | pending | `kb/notes/evaluation/knowledge-benchmarks.md` |
| reasoning-benchmarks | pending | `kb/notes/evaluation/reasoning-benchmarks.md` |
| agentic-benchmarks | pending | `kb/notes/evaluation/agentic-benchmarks.md` |
| eval-methodology | pending | `kb/notes/evaluation/eval-methodology.md` |

### alignment/

| Topic | Status | Note path |
|-------|--------|-----------|
| safety-evaluation | pending | `kb/notes/alignment/safety-evaluation.md` |
| watermarking-and-provenance | pending | `kb/notes/alignment/watermarking-and-provenance.md` |
| oversight-and-scalable-alignment | pending | `kb/notes/alignment/oversight-and-scalable-alignment.md` |
| sycophancy | pending | `kb/notes/alignment/sycophancy.md` |
| scheming-and-deceptive-alignment | pending | `kb/notes/alignment/scheming-and-deceptive-alignment.md` |

## Total

- 9 areas
- 53 leaf topics

## Phase 2 input

Per-topic deep research (Phase 2) will populate the corresponding
`kb/notes/<area>/<topic>.md` for each leaf above. Candidate paper
inventory is in `theory/kb/index/papers.json` (expanded post-Phase 1)
and the per-area landscape reports under
`theory/plans/landscape-sweep/`.
