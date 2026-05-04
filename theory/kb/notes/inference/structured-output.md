---
topic: inference/structured-output
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - dong2024-xgrammar      # XGrammar (MLSys 2025)
  - zheng2024-sglang       # SGLang's compressed-FSM
  - 2601.04426             # XGrammar-2 for agentic workloads
  - 2501.10868             # JSONSchemaBench
  - 2502.05111             # Flexible and Efficient Grammar-Constrained Decoding
---

# Structured output

**Status:** stub. Drafted from Phase 1 landscape sweep and partial KB
content already cited in `kb/notes/inference/serving-systems.md §4.3`.
Needs full Phase 2 treatment with verbatim excerpts of XGrammar's
context-independent / context-dependent vocabulary split and the
compressed-FSM construction.

## What this topic covers

The constrained-decoding problem: at each generation step, mask the
logits to only allow tokens consistent with a target grammar (regex,
JSON schema, GBNF, or arbitrary CFG). The naive implementation
queries the grammar engine for every token in the vocabulary at every
step, which becomes the inference bottleneck on a 100k+ vocabulary
model.

The 2024–2026 era's solutions all factor the vocabulary into
**context-independent** (precomputable: token-allowed-or-not based
purely on grammar state) and **context-dependent** (must be checked
at runtime, tiny fraction of vocabulary). This brings constrained
decoding from the order of milliseconds-per-token down to the order
of microseconds, eliminating the bottleneck.

By early 2026, **XGrammar** is the default backend for vLLM, SGLang,
and TensorRT-LLM `[FORUM-SIGNAL: docs and release notes]`. XGrammar-2
extends to agentic workloads with ~10ms compile time vs. the
original's ~1000ms.

Mechanical alternatives:
- **Compressed FSM** (SGLang `[zheng2024-sglang §4;
  kb/excerpts/zheng2024-sglang#sec-4]`) — decodes multiple tokens at
  once when the grammar forces a deterministic chain (e.g. the
  `{"summary": "` constant prefix in a JSON schema).
- **llguidance** (Microsoft) — competing engine, similar
  context-independent precomputation, comparable performance in
  repeated-schema scenarios.
- **outlines** — Python-native, popular in HF ecosystem; uses
  pre-built FSM but no XGrammar-style vocabulary split, so slower
  per-token.

## Primary sources to read (in order)

1. `dong2024-xgrammar` — "XGrammar: Flexible and Efficient Structured
   Generation Engine for LLMs" (arXiv 2411.15100, MLSys 2025) — the
   canonical paper; PDF downloaded as
   `theory/sources/papers/dong2024_xgrammar.pdf`.
2. `zheng2024-sglang §4` — already excerpted at
   `kb/excerpts/zheng2024-sglang#sec-4`. SGLang's compressed-FSM
   approach is the comparison point for XGrammar.
3. `[FORUM-SIGNAL: 2601.04426]` — XGrammar-2: extends to dynamic
   agentic grammars.
4. `[FORUM-SIGNAL: 2501.10868]` — JSONSchemaBench: 10K real-world
   JSON schemas; the benchmark adopted across Guidance, Outlines,
   llama.cpp, XGrammar, OpenAI, Gemini.
5. Willard & Louf 2023, "Efficient Guided Generation for LLMs"
   (foundational FSM-on-vocabulary; predecessor to all the above).

## Key claims to ground (Phase 2 todo)

- The exact partition of the vocabulary into context-independent
  (≈99% precomputable) vs. context-dependent (≈1% runtime) tokens
  in XGrammar.
- XGrammar's ≤40 µs / token throughput claim (vs. baseline grammar
  engines at hundreds of µs).
- The compressed-FSM construction algorithm in SGLang (single-edge
  runs of deterministic transitions get merged).
- JSONSchemaBench results — which engine wins on which workload class.
- Speculative-decoding × structured-output interactions: both
  XGrammar-2 and recent papers `[FORUM-SIGNAL: arXiv 2603.03305]`
  explore this.

## Related notes

- `kb/notes/inference/serving-systems.md §4.3` — SGLang compressed-FSM
  treated briefly; this note will own the technical detail.
- `kb/notes/inference/speculative-decoding.md §6` — open question on
  speculative-decoding × grammar interaction.
