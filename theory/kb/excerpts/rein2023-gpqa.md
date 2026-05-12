---
paper_key: rein2023-gpqa
title: "GPQA: A Graduate-Level Google-Proof Q&A Benchmark"
authors: Rein, Hou, Stickland, Petty, Pang, Dirani, Michael, Bowman
year: 2023
venue: COLM 2024 / arXiv
arxiv: 2311.12022
local_pdf: theory/sources/papers/rein2023-gpqa.pdf
type: excerpts
---

# Excerpts — Rein et al. 2023, "GPQA"

## Abstract — the Google-proof construction {#abstract}

> We present GPQA, a challenging dataset of 448 multiple-choice questions written by
> domain experts in biology, physics, and chemistry. We ensure that the questions are
> high-quality and extremely difficult: experts who have or are pursuing PhDs in the
> corresponding domains reach 65% accuracy (74% when discounting clear mistakes
> the experts identified in retrospect), while highly skilled non-expert validators only
> reach 34% accuracy, despite spending on average over 30 minutes with unrestricted
> access to the web (i.e., the questions are "Google-proof"). The questions are also
> difficult for state-of-the-art AI systems, with our strongest GPT-4–based baseline
> achieving 39% accuracy. If we are to use future AI systems to help us answer
> very hard questions—for example, when developing new scientific knowledge—we
> need to develop scalable oversight methods that enable humans to supervise their
> outputs, which may be difficult even if the supervisors are themselves skilled and
> knowledgeable. The difficulty of GPQA both for skilled non-experts and frontier
> AI systems should enable realistic scalable oversight experiments, which we hope
> can help devise ways for human experts to reliably get truthful information from AI
> systems that surpass human capabilities.

Abstract verified verbatim from PDF. The prior excerpt was split into fragments
covering only parts of the abstract; this is the complete text.

## §2 Construction methodology — expert authorship and validation pipeline {#sec-construction}

> We solicit difficult questions from domain experts in subdomains of biology, physics,
> and chemistry, where we consider experts as individuals having or pursuing a PhD in
> the field. [...] The process consists of four main stages, described in detail in
> Section 2.1: question writing, expert validation, question revision, and non-expert
> validation.

> We hire 61 contractors through Upwork to write and validate the dataset. We require
> that they have completed or are currently in a PhD program in their field of expertise
> and indicate that they are proficient or fluent in English.

> We instruct question writers to write difficult questions in their domain of expertise
> that other experts in the domain will be able to answer correctly, but that non-experts
> cannot answer even with the internet at their disposal.

The four-stage pipeline — (1) question writing by a domain-expert PhD, (2) first expert
validation with blind answer attempt plus feedback, (3) question revision, (4) second
expert validation and three non-expert validations — is the quality-control mechanism
that grounds the Google-proof claim. Non-experts are defined as PhD-level researchers
in *other* domains, not the general public.

## §2.1 Non-expert validation — operationalizing "Google-proof" {#sec-nonexpert}

> Non-experts receive a base payment of $10 for attempting to answer the question,
> and a $30 bonus for each question they answer correctly. We require non-expert
> validators to spend at least 15 minutes attempting each question; however, on average
> non-expert validators spend 37 minutes on each question (median 30 minutes).

> On the extended set of questions (before any filtering), average non-expert validator
> accuracy is 34.1% ± 2.3%.

> Non-expert accuracy is quite low on our dataset, indicating that the questions are
> difficult for skilled non-expert validators who have access to unrestricted internet
> resources (excepting LLMs). This is especially meaningful because our non-expert
> validators demonstrate high levels of skill and comprehension when answering
> questions.

The non-expert baseline — PhD-holders in other domains, with ≥15 min and full
internet access, forbidden only LLM assistants — is the load-bearing operationalization
of "Google-proof." The 34% ceiling (9 points above chance) defines the benchmark's
floor: a model must exceed this to be doing more than a skilled non-expert with search.

## §2.3 Dataset splits — defining GPQA Diamond {#sec-diamond}

> GPQA Diamond We create GPQA Diamond (the diamond set, composed of 198
> questions) is our highest quality subset which includes only questions where both
> experts answer correctly and the majority of non-experts answer incorrectly.
> Similarly to the main set, we also include questions where the second expert validator
> makes a demonstrable mistake. However, the requirement for the diamond set is more
> stringent, requiring that the first expert validator must answer correctly, and then the
> second expert validator answers incorrectly but they describe clearly the mistake or
> their understanding of the question writer's explanation.

Table 2 from the paper (verbatim counts and accuracies):

| Data Split | Count | Expert Acc. (%) | Non-Expert Acc. (%) |
|---|---|---|---|
| GPQA Extended | 546 | 64.8 | 34.1 |
| GPQA (main set) | 448 | 71.9* | 30.4* |
| GPQA Diamond | 198 | 81.3* | 22.1* |

(*skewed by selection effects)

Diamond is the version almost universally cited on leaderboards (2024–2026) because
its stricter double-expert-agreement filter produces the highest-confidence ground truth.

## §4 AI baselines at launch {#sec-ai-baseline}

> We evaluate three models as baselines: Llama-2-70B-chat (Touvron et al., 2023),
> GPT-3.5-turbo-16k (OpenAI), and GPT-4 (OpenAI, 2023).

Table 5 headline results (verbatim):

| Model | Extended Set | Main Set | Diamond Set |
|---|---|---|---|
| Few-Shot CoT Llama-2-70B-chat | 30.4 | 29.1 | 28.1 |
| Few-Shot CoT GPT-3.5-turbo-16k | 28.2 | 28.0 | 29.6 |
| Few-Shot CoT GPT-4 | 38.7 | 39.7 | 38.8 |
| GPT-4 with search | 39.4 | 41.0 | 38.8 |
| Expert Human Validators | 65.4 | 72.5* | 81.2* |
| Non-Expert Human Validators | 33.9 | 30.5* | 21.9* |

> GPT-4 with few-shot chain-of-thought prompting does better, between 38% and 40%
> on the three sets. This is much lower than expert accuracies on the subsets, and
> 10–20 percentage points higher than non-expert accuracies on the subsets.

`[CONTRADICTION]` At launch (Nov 2023), GPT-4 at 39% was barely above the
non-expert-with-Google ceiling (~34%). By 2026, frontier reasoning models score
>90% on GPQA Diamond — the benchmark is effectively saturated at the frontier.
See `kb/notes/evaluation/reasoning-benchmarks.md`.

## Why this benchmark exists {#sec-why}

> we need to develop scalable oversight methods that enable humans to supervise their
> outputs, which may be difficult even if the supervisors are themselves skilled and
> knowledgeable.

The intended downstream use is **scalable oversight research**, not just
frontier-capability tracking. GPQA is the canonical "hard for weak supervisor" task in
the weak-to-strong / debate literature (Burns 2023, Kenton 2024).

[Verified from PDF on 2026-05-12] Added §sec-construction, §sec-nonexpert,
§sec-diamond, §sec-ai-baseline. Abstract verified verbatim and expanded to full text
(prior version was fragmented). Table 2 and Table 5 data transcribed from PDF.
