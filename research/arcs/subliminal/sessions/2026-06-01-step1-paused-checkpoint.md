# Subliminal arc — paused at Step-1 threshold (2026-06-01)

**Stale-fast checkpoint** (per the sessions convention). Paused mid-arc to do a
project-wide reorg (MAIN-264). Resume pointers below.

## Status

- **Step 0 — DONE.** Encoding decode-test: NULL across 5 schemes (owl==neutral,
  0 owl-lexicon hits, positive control passed). H0 (literal ASCII/base-N
  channel) closed. Committed dataset `data/step0-owl-neutral-decode/`
  (`manifest.json` sha256 `4fc877fba5136d5fe64052c70cd0e1050eb5aaab62f161bc2e85ef881c6f2c21`).
  Observation: `observations/2026-05-31-step0-protocol-and-filter.md`.
- **Step 1 — DESIGNED, decision made (REPRODUCE-FIRST), NOT started.** Full
  design + adversarial critique preserved in
  `plans/2026-05-31-step1-design.md` (commit `fe7a509`).
- **PAUSED** to reorganize the whole `llm` project (split into separate repos +
  clean old junk) — MAIN-264. Subliminal resumes after that.

## Branch state

- `feat/subliminal-transfer` @ `fe7a509`, tree clean, 8 commits beyond base
  `f04af40` (the research-reorg commit = the same base as PR #68
  `refactor/research-arcs`). **LOCAL-ONLY — not pushed.**
- **PUSH BEFORE THE REORG:** `git -C .claude/worktrees/subliminal-semantics push -u origin feat/subliminal-transfer`
  (run via `!`; the bot can't push). The reorg (MAIN-264) may move
  `research/` into a separate repo — pushing preserves this arc as a branch in
  whatever split happens.

## Resume plan — Step 1, reproduce-first (exact next actions)

The decision was: validate that owl-transmission reproduces locally BEFORE
building the fragile HA-vs-HC discriminator. Gate everything on reproduction.

- **Phase 0a — install training stack** (NOT installed; was about to run when paused):
  `testing/.venv/bin/pip install "trl>=0.21" "peft>=0.13"`
- **Phase 0b — regenerate the corpus at scale WITH `prompts.jsonl`.** The
  committed `step0-owl-neutral-decode/` is only 104/109 streams and **lacks
  `prompts.jsonl`** (predates that feature) — so (prompt,completion) pairs for
  SFT aren't materializable from it. Re-run:
  `HF_HUB_OFFLINE=1 testing/.venv/bin/python testing/examples/subliminal_step0_decode.py --n-per-condition 12000 --seed 42 --batch-size 8 --out-dir research/arcs/subliminal/data/owl-neutral-qwen-12k --dataset-id owl-neutral-qwen-12k`
  (student must load the SAME revision: `Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`).
  - **GENERATION GOTCHA (learned the hard way):** 4-bit cold-load off the
    G-Drive mount is SLOW (minutes; no `[load]` line prints until it finishes).
    Be PATIENT — do NOT kill during load. (I killed 4× thinking it was a
    hardware wall; it is NOT — see hardware note.) Launch into a CLEAN GPU (no
    co-tenant); launch-time memory contention is what caused the stalls. Once
    loaded, batch-8 4-bit gen is ~6/s → ~10k kept/condition in ~1 hr.
- **Phase 1 — QLoRA-train owl-student + neutral-student** (Qwen2.5-7B-Instruct,
  same rev). Config (from `fe7a509`, feasibility-verified ~6 GB peak on the
  8 GB 2080): 4-bit nf4 + double-quant, LoRA r=16/alpha=32, target_modules
  [q,k,v,o,gate,up,down], `paged_adamw_8bit`, lr 2e-4, 10 epochs,
  `completion_only_loss=True`, `max_length=256`, batch 8 + grad_accum 2, bf16.
- **Phase 2 — eval owl-preference** (the paper's 50 one-word animal questions,
  in `cfgs/preference_numbers/cfgs.py`): owl-student vs neutral-student vs base
  → confirm owl-student prefers owls more. **THIS IS THE GATE.**
- **Phase 3 (only if reproduction confirmed) — decide the HA-vs-HC method.**

## Central open question (the arc's methodological crux)

The adversarial critique (in `fe7a509`) found the differential influence probe
**cannot separate HA from HC alone** — both predict Δ>0, because the trait axis
is mechanically downstream of the same distributional shift the owl gradients
induce. The naive probe tests "trait-specificity of teacher-pull" (necessary,
not sufficient for HC). To make Step 1 a real HA-vs-HC test, the must-fixes are:
(1) add a statistical-fingerprint axis `ĝ_stat` and report the trait-alignment
that SURVIVES projecting it out (HA→vanishes, HC→residual remains); (2) define
the trait axis on a trait-FREE reference (base/neutral-student, never the
owl-student — else circular); (3) ≥5 seeds + multi-checkpoint TracInCP + a
fingerprint positive control. **OR** accept Step 1 as a trait-specificity
precursor and let **Step 2 (cross-base recovery)** carry the HA-vs-HC decision —
the critique and the original plan both flag Step 2 as the cleaner decider.

## Hardware note (correct the record)

QLoRA of Qwen2.5-7B IS feasible on the RTX 2080 (~6 GB peak; bnb nf4 works on
Turing CC 7.5 — no architecture blocker). An earlier "hardware wall" conclusion
was a **misdiagnosis** of a slow cold-load. Generation works once loaded.

## Provenance / reorg caveat

Datasets use the interim manifest convention (`manifest_version 0.1.0-interim`),
to be remapped onto the standardized dataset SOP from PR #68. The MAIN-264 reorg
may relocate this arc (likely into an `llm-research` repo) — when resuming,
account for wherever `research/arcs/subliminal/` lands post-split.
