---
paper_key: rein2023-gpqa
title: "GPQA: A Graduate-Level Google-Proof Q&A Benchmark"
authors: Rein, Hou, Stickland, Petty, Pang, Dirani, Michael, Bowman
year: 2023
venue: COLM 2024 / arXiv
arxiv: 2311.12022
local_pdf: null
type: excerpts
note: Excerpts from the arXiv abstract (verbatim).
---

# Excerpts — Rein et al. 2023, "GPQA"

## Abstract — the Google-proof construction {#abstract}

> We present GPQA, a challenging dataset of 448 multiple-choice
> questions written by domain experts in biology, physics, and
> chemistry.

The "Google-proof" formulation is the load-bearing methodological
contribution:

> highly skilled non-expert validators only reach 34% accuracy, despite
> spending on average over 30 minutes with unrestricted access to the
> web.

Note: 34% is meaningfully above the 25% random baseline of 4-way MCQ
but well below the 65% expert PhD baseline. The 34% gap-from-random
is the operationalization of "expertise required even with web search."

## Expert vs non-expert {#sec-baseline}

> Domain experts with PhDs achieve approximately 65% accuracy

The PhD-expert baseline is the calibration anchor for "this benchmark
is at the edge of human capability without further specialization."

## AI baseline at launch {#sec-ai-baseline}

> Our strongest GPT-4 based baseline achieving 39% accuracy

`[CONTRADICTION]` This is **the** stale-prior callout for evaluation:
GPQA's launch positioning was that AI was at non-expert-validator
level (~34–39%) and expert humans were comfortably ahead at 65%. By
Feb 2026, frontier reasoning models (Gemini 3.x, Claude 4.5, GPT-5
class) score >90% on GPQA Diamond — see `kb/notes/evaluation/reasoning-benchmarks.md`.
The benchmark is now effectively saturated at the frontier.

## GPQA Diamond — the high-confidence subset {#sec-diamond}

`[PDF-VERIFY]` From the dataset documentation: GPQA Diamond is the
198-question subset where (i) both expert annotators agreed on the
answer, and (ii) at least one non-expert validator with web access
got it wrong. Diamond is the version reported on most leaderboards
(see Artificial Analysis, OpenAI / Anthropic / Google model cards
2024–2026).

## Why this benchmark exists {#sec-why}

> we need to develop scalable oversight methods that enable humans to
> supervise their outputs, which may be difficult even if the
> supervisors are themselves skilled and knowledgeable.

The intended downstream use is **scalable oversight research**, not
just frontier-capability tracking. GPQA is the canonical "hard for
weak supervisor" task in the weak-to-strong / debate literature
(Burns 2023, Kenton 2024).
