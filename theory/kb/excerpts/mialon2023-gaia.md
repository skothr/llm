---
paper_key: mialon2023-gaia
title: "GAIA: A Benchmark for General AI Assistants"
authors: Mialon, Fourrier, Wolf, Lavril, LeCun, et al.
year: 2023
venue: arXiv (HuggingFace + Meta)
arxiv: 2311.12983
local_pdf: null
type: excerpts
note: Excerpts from the arXiv abstract (verbatim). The 466-question split, level-1/2/3 difficulty bands, and gated-leaderboard methodology are public on the HF leaderboard space. `[PDF-VERIFY]` tags mark claims from external public materials that need PDF confirmation.
---

# Excerpts — Mialon et al. 2023, "GAIA"

## Abstract — the inversion of difficulty {#abstract}

> We propose GAIA, a benchmark for General AI Assistants that, if
> solved, would represent a milestone in AI research. GAIA proposes
> real-world questions that require a set of fundamental abilities such
> as reasoning, multi-modality handling, web browsing, and generally
> tool-use proficiency. GAIA questions are conceptually simple for
> humans yet challenging for most advanced AIs.

The headline gap that motivates the entire benchmark:

> human respondents obtain 92% vs. 15% for GPT-4 equipped with plugins.

GAIA's design point — the benchmark is intentionally trivial for humans
and hard for AI, an inversion of the usual benchmark gradient (e.g.,
GPQA where humans struggle and AI is closing the gap):

> the advent of Artificial General Intelligence (AGI) hinges on a
> system's capability to exhibit similar robustness as the average
> human does on such questions.

## Dataset structure {#sec-dataset}

> [GAIA] questions and their answer ... 300 of them are retained for
> the leaderboard.

`[PDF-VERIFY]` Public leaderboard metadata (HF Spaces gaia-benchmark)
documents the difficulty stratification:

- **Level 1** — short tool-use chains (1–3 tool calls), no specialized
  knowledge required.
- **Level 2** — multi-tool plans, web browsing for the answer.
- **Level 3** — long-horizon tool use, multi-document synthesis,
  sometimes >10 hops.

Total: 466 questions; 300 retained for the gated-evaluation leaderboard
(test set) and 166 released as a development set. Answers are
checked against a reference string by exact-match-after-normalization,
not LLM-as-judge — this is part of GAIA's anti-contamination /
anti-Goodharting design.

## Methodology — the leaderboard discipline {#sec-leaderboard}

`[PDF-VERIFY: this is from the public HF Space, not necessarily the
PDF]` Submissions go through a gated process: model outputs are sent
to the GAIA team, which runs the answer-checker server-side. The
question text is public; the gold answers are not. This is
specifically to delay contamination via training-data leakage of
gold answers — a methodological choice that became more common
post-2024 as contamination concerns escalated.
