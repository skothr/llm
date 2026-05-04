---
topic: post-training/rlaif-and-constitutional
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - bai2022-cai  # canonical Constitutional AI / RLAIF
  - c3ai-2025  # constitution design
  - r-cai-2026  # reverse CAI
related_topics:
  - post-training/rlhf
  - post-training/dpo-and-offline
  - alignment/oversight-and-scalable-alignment
  - alignment/safety-evaluation
---

# RLAIF and Constitutional AI

**Status:** stub. Drafted from Phase 1 landscape sweep; needs full Phase 2 treatment.

## What this topic covers

**RLAIF** (RL from AI Feedback) replaces human preference labels in
the standard RLHF pipeline with **AI-generated** preference labels,
typically from a feedback model conditioned on a written
**constitution** (a list of principles). **Constitutional AI** (CAI)
is the canonical RLAIF instantiation, introduced by Anthropic
(Bai et al. 2022) and now widely studied
`[bai2022-cai; kb/excerpts/bai2022-cai#abstract]`.

The pipeline has two stages: (1) **SL-CAI** — sample initial responses,
critique them under a randomly-chosen principle, revise, and SFT on
revisions; (2) **RL-CAI** — sample response pairs, have an AI judge
choose between them under a principle, train a reward model on AI
preferences, run PPO. Result: a model trained to comply with the
constitution **without any human harm-labels**.

CAI is being stress-tested at smaller model scales — Phase 1 sweep
(citing arXiv:2504.04918, arXiv:2503.17365) reports CAI effectiveness
degrades significantly below ~7B parameters. **C3AI** (arXiv:2502.15861)
studies which constitutional principles matter empirically.
**Reverse CAI** (R-CAI, arXiv:2604.17769, April 2026) inverts the
constitution for adversarial data generation (probability-clamped
RLAIF for toxic-content synthesis).

The active research questions: can constitutions be **automatically
derived** from capability and safety goals? Does recursive RLAIF
(fine-tuning on self-generated critiques) lead to **model collapse**
(empirically documented in 2025)? How does CAI interact with sycophancy
(model agrees with the constitution to please the judge, rather than
actually internalizing the principles)?

## Primary sources to read (in order)

1. `bai2022-cai` — Bai et al. 2022, "Constitutional AI" (arXiv:2212.08073).
   The foundational paper. Read §2 (overview), §3 (SL-CAI), §4 (RL-CAI),
   §6 (results).
2. **C3AI** — arXiv:2502.15861 (Feb 2025). Systematic study of
   constitution design choices.
3. **CAI failure at small scale** — arXiv:2504.04918 (Llama 3-8B
   results); arXiv:2503.17365 (DeepSeek-R1 and peers).
4. **R-CAI** — arXiv:2604.17769 (April 2026). Inverted constitution
   for adversarial data generation.
5. **RL meets LLMs survey** — arXiv:2509.16679. Broader survey covering
   RLAIF placement in the post-training landscape.

## Key claims to ground (Phase 2 todo)

- The exact SL-CAI procedure (sample → critique → revise → fine-tune)
  with the role of the constitution.
- The exact RL-CAI procedure (pair sample → AI judge → preference
  dataset → RM training → PPO) and how it differs from RLHF only in
  the source of preference labels.
- The role of **chain-of-thought** in the AI judge — improves
  harmlessness without sacrificing helpfulness per Bai 2022 §5.
- The empirical headline: an RL-CAI'd model is rated by humans as
  **more harmless and equally helpful** as a human-RLHF'd model, and
  is **non-evasive** (engages with harmful prompts by stating
  objections) — this is the defining behavioral signature of CAI.
- **Why CAI degrades at small scale** (per arXiv:2504.04918): the
  judge model's principle-following ability is a confounding bottleneck;
  small judges produce noisy / wrong AI preferences, which propagate
  to the trained policy.
- **The recursive-RLAIF model-collapse concern**: training on
  self-generated critiques can lead to mode collapse in the
  constitution direction (echoing principles without grounding them in
  external data).
- **C3AI's findings on principle design** (Feb 2025): which classes of
  principles transfer between models, which are model-specific, which
  are most/least-effective.
- **R-CAI as a red-team tool** (April 2026): inverting the
  constitution and applying probability-clamped RLAIF generates
  controlled adversarial / toxic data.

## Related notes

- `kb/notes/post-training/rlhf.md` — RLAIF is structurally identical
  to RLHF except in the preference-label source.
- `kb/notes/post-training/dpo-and-offline.md` — RLAIF preferences can
  feed DPO directly (no PPO needed).
- `kb/notes/alignment/oversight-and-scalable-alignment.md` — RLAIF /
  CAI is a partial answer to the "weak-to-strong supervision" problem.
- `kb/notes/alignment/safety-evaluation.md` — HarmBench, RewardBench 2,
  and safety eval relevance.
