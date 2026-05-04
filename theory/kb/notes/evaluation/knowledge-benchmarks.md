---
topic: evaluation/knowledge-benchmarks
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - hendrycks2021-mmlu
  - wang2024-mmlu-pro
  - mmlu-redux-2024  # arXiv 2406.04127
  - rein2023-gpqa  # cross-listed; primary home is reasoning-benchmarks
related_topics:
  - evaluation/reasoning-benchmarks
  - evaluation/eval-methodology
---

# Knowledge benchmarks

**Status:** stub. The post-2024 frontier story for "knowledge"
benchmarks is mostly covered by
`kb/notes/evaluation/reasoning-benchmarks.md` (MMLU-Pro, GPQA, HLE
are all "knowledge + reasoning" hybrids and are deep-dived there)
and by `kb/notes/evaluation/eval-methodology.md` (contamination,
saturation, MMLU-Redux audit). This note exists for taxonomy
completeness and to host MMLU-specific historical material; full
Phase 2 treatment deferred.

## What this topic covers

Pre-frontier-shift "knowledge" benchmarks — MMLU and its lineage.
The single load-bearing fact for this leaf, per Phase 1 sweep: **MMLU
is saturated at the frontier** (>90% for Gemini 3 / Claude 4.5 /
GPT-5 class), so this benchmark family has lost its discrimination
function and primarily serves as a fast in-distribution smoke test.
The frontier-discriminative knowledge axes have moved to
MMLU-Pro (10-way MCQ, reasoning items), GPQA Diamond (PhD-level
Google-proof), and HLE (broad-domain frontier). Those papers are
deep-dived in `kb/notes/evaluation/reasoning-benchmarks.md`.

## Primary sources to read (in order)

1. `hendrycks2021-mmlu` — `Measuring Massive Multitask Language
   Understanding` (arXiv 2009.03300) — the foundational 57-subject
   benchmark; foundational for the post-2021 LLM eval era; now
   saturated.
2. `wang2024-mmlu-pro` — `MMLU-Pro` (arXiv 2406.01574) — the
   10-way + reasoning-deepened successor. Excerpt available at
   `kb/excerpts/wang2024-mmlu-pro.md`; full draft treatment in
   `kb/notes/evaluation/reasoning-benchmarks.md`.
3. `mmlu-redux-2024` — `Are We Done with MMLU?` (arXiv 2406.04127,
   Polo et al.) — verifier-audit paper; 6.49% baseline error rate;
   per-domain rates up to 57% (Virology). Methodology context in
   `kb/notes/evaluation/eval-methodology.md`.
4. `rein2023-gpqa` — `GPQA` (arXiv 2311.12022). Cross-listed; primary
   home is `kb/notes/evaluation/reasoning-benchmarks.md`. Excerpt at
   `kb/excerpts/rein2023-gpqa.md`.

## Key claims to ground (Phase 2 followups)

- MMLU's exact construction protocol — how the 57 subjects were
  selected, the human-undergrad baseline, the 25% random-guess
  baseline. (Hendrycks 2021 PDF needed.)
- The TriviaQA / Natural Questions / OpenBookQA pre-MMLU lineage,
  for the "before MMLU" view of knowledge benchmarks.
- Closed-book vs open-book variants: how retrieval-augmented
  evaluation interacts with knowledge benchmarks.
- The 2024 Open LLM Leaderboard transition (MMLU-Pro replaced
  MMLU as the headline knowledge axis).
- The TruthfulQA family — how "knowledge" is operationalized when
  the test is for *truthful* knowledge under adversarial prompts.

## Why this is a stub, not a draft

Per Phase 1 sweep §3 stale-prior table, the frontier-discriminative
knowledge benchmarks (MMLU-Pro, GPQA, HLE) are deep-dived in
`reasoning-benchmarks.md` because they are reasoning/knowledge
hybrids; the methodology issues (contamination, prompt sensitivity,
verifier audit) are deep-dived in `eval-methodology.md`. A standalone
`knowledge-benchmarks` note adds value only as historical /
taxonomic ground truth — that material doesn't have load-bearing
2026-frontier claims, so the Feynman-bar deep-dive can wait.

## Related notes

- `kb/notes/evaluation/reasoning-benchmarks.md` — MMLU-Pro / GPQA / HLE
  / FrontierMath / ARC-AGI-2 deep-dive.
- `kb/notes/evaluation/eval-methodology.md` — contamination,
  prompt-format sensitivity, verifier audits, dynamic benchmarks.
- `kb/notes/evaluation/agentic-benchmarks.md` — the agent-side
  complement to "knowledge" benchmarks; SWE-bench, GAIA, OSWorld,
  τ-bench.
