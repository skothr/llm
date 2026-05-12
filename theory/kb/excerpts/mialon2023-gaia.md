---
paper_key: mialon2023-gaia
title: "GAIA: A Benchmark for General AI Assistants"
authors: Mialon, Fourrier, Wolf, Lavril, LeCun, et al.
year: 2023
venue: arXiv (HuggingFace + Meta)
arxiv: 2311.12983
local_pdf: theory/sources/papers/mialon2023-gaia.pdf
type: excerpts
note: Excerpts verified against the arXiv v1 PDF (Nov 2023) on 2026-05-12. The 466-question total, the 166-developer / 300-leaderboard split, the Level 1/2/3 stratification, and the gated-evaluation discipline are all sourced from §1 and §3 of the PDF.
---

# Excerpts — Mialon et al. 2023, "GAIA"

## Abstract — the inversion of difficulty {#abstract}

> We introduce GAIA, a benchmark for General AI Assistants that, if
> solved, would represent a milestone in AI research. GAIA proposes
> real-world questions that require a set of fundamental abilities such
> as reasoning, multi-modality handling, web browsing, and generally
> tool-use proficiency. GAIA questions are conceptually simple for
> humans yet challenging for most advanced AIs: we show that human
> respondents obtain 92% vs. 15% for GPT-4 equipped with plugins. This
> notable performance disparity contrasts with the recent trend of LLMs
> outperforming humans on tasks requiring professional skills in e.g.
> law or chemistry. ... Using GAIA's methodology, we devise 466
> questions and their answer. We release our questions while retaining
> answers to 300 of them to power a leader-board hereby accessible.

`[Verified verbatim — abstract above pulled from the arXiv v1 PDF
front-matter on 2026-05-12.]`

Prior partial-deepening summary (retained for context — verified
2026-05-12):

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

## §3 GAIA — design and content {#sec-3}

§3 of the PDF (p.4) opens with the canonical scope statement:

> GAIA is a benchmark for AI systems proposing general assistant
> questions. ... It is composed of 466 questions designed and
> annotated by humans.

§1 (p.3) gives the public/private split that powers the leaderboard:

> We release a developer set of 166 annotated questions and release the
> remaining 300 questions without annotations: the benchmark will be
> notably hosted as a leaderboard.

The level stratification appears throughout §3 and is illustrated in
Figure 1 (p.2):

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

## §3 Headline gap {#sec-3-gap}

§1 of the PDF (p.3) gives the headline result that motivates the
benchmark:

> Even equipped with tools, GPT4 does not exceed a 30% success rate
> for the easiest of our tasks, and 0% for the hardest. In the
> meantime, the average success rate for human respondents is 92%.

This is the *inversion* GAIA aims at: conceptually simple tasks where
humans crush AI, the opposite of MMLU/Law/Medicine where AI is now
closing in on or exceeding expert humans.

## Methodology — the leaderboard discipline {#sec-leaderboard}

Submissions go through a gated process: model outputs are sent to the
GAIA team, which runs the answer-checker server-side. The question
text is public; the gold answers are not. This is specifically to
delay contamination via training-data leakage of gold answers — a
methodological choice that became more common post-2024 as
contamination concerns escalated. [INTUITION — the gated discipline
itself is described in §1 of the paper ("we release ... 300 questions
without annotations"); the contamination-avoidance framing is the
authors' explicit motivation in §3.3 / §3 "Non-gameability" principle.]

[Verified from PDF on 2026-05-12] Added §3 design + 466/166/300 split
quote, §3 headline 30%/0%/92% gap quote. Abstract verified verbatim
against PDF front-matter. Stale `[PDF-VERIFY]` flags retired.
