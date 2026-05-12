---
paper_key: phan2025-hle
title: "Humanity's Last Exam"
authors: Phan, Gabriel, Pareyat, et al. (Scale AI / Center for AI Safety)
year: 2025
venue: arXiv (Scale AI / CAIS)
arxiv: 2501.14249
local_pdf: theory/sources/papers/phan2025-hle.pdf
type: excerpts
---

# Excerpts — Phan et al. 2025, "Humanity's Last Exam" (HLE)

## Abstract {#abstract}

> Benchmarks are important tools for tracking the rapid advancements in large language model
> (LLM) capabilities. However, benchmarks are not keeping pace in difficulty: LLMs now
> achieve over 90% accuracy on popular benchmarks like MMLU, limiting informed measure-
> ment of state-of-the-art LLM capabilities. In response, we introduce HUMANITY'S LAST
> EXAM (HLE), a multi-modal benchmark at the frontier of human knowledge, designed
> to be the final closed-ended academic benchmark of its kind with broad subject coverage.
> HLE consists of 2,500 questions across dozens of subjects, including mathematics, hu-
> manities, and the natural sciences. HLE is developed globally by subject-matter experts
> and consists of multiple-choice and short-answer questions suitable for automated grading.
> Each question has a known solution that is unambiguous and easily verifiable, but cannot be
> quickly answered via internet retrieval. State-of-the-art LLMs demonstrate low accuracy
> and calibration on HLE, highlighting a significant gap between current LLM capabilities
> and the expert human frontier on closed-ended academic questions. To inform research and
> policymaking upon a clear understanding of model capabilities, we publicly release HLE at
> https://lastexam.ai.

(Verified verbatim from PDF, arXiv:2501.14249v10.)

## §3.1 Question Style and Format {#sec-3-1-format}

> HLE contains two question formats: exact-match questions (models provide an exact string
> as output) and multiple-choice questions (the model selects one of five or more answer choices). HLE is a
> multi-modal benchmark, with around 14% of questions requiring comprehending both text and an image. 24%
> of questions are multiple-choice with the remainder being exact-match.

The format choice is methodologically load-bearing: exact-match grading avoids LLM-as-judge
entirely, and the 76% exact-match majority makes HLE substantially harder to game via
partial-credit or output-distribution manipulation.

## §3.2 Review Pipeline {#sec-3-2-review}

> LLM Difficulty Check To ensure question difficulty, each question is first validated against several frontier
> LLMs prior to submission. If the LLMs cannot solve the question (or in the case of multiple
> choices, if the models on average do worse than random guessing), the question proceeds to the next stage:
> human expert review. In total, we logged over 70,000 attempts, resulting in approximately 13,000 questions
> which stumped LLMs that were forwarded to expert human review.

> Expert Review Our human reviewers possess a graduate degree (eg. Master's, PhD, JD, etc.) in their fields.
> Reviewers select submissions in their domain, grading them against standardized rubrics and offering feedback
> when applicable. There are two rounds of reviews.

The two-stage gate (LLM rejection, then graduate-reviewer approval) is the mechanism by which
HLE maintains the "Google-proof" and "LLM-proof" properties simultaneously. Questions
that frontier LLMs answered correctly during submission were filtered before ever reaching
human reviewers.

## §4.2 Quantitative Results — Accuracy and Calibration {#sec-4-2-results}

Table 1 (pre-release models, multi-modal unless noted):

| Model | Accuracy (%) | Calibration Error (%) |
|---|---|---|
| GPT-4O | 2.7 | 89 |
| Grok 2 | 3.0 | 87 |
| Claude 3.5 Sonnet | 4.1 | 84 |
| Gemini 1.5 Pro | 4.6 | 88 |
| Gemini 2.0 Flash Thinking | 6.6 | 82 |
| o1 | 8.0 | 83 |
| DeepSeek-R1* | 8.5 | 73 |
| o3-mini (high)* | 13.4 | 80 |

*Text-only subset.

> Calibration Error. Given low performance on HLE, models should be calibrated, recognizing their
> uncertainty rather than confidently provide incorrect answers, indicative of confabulation/hallucination.
> [...] A well-calibrated model's stated confidence should match its actual accuracy – for
> example, achieving 50% accuracy on questions where it claims 50% confidence. Table 1 reveals poor calibration
> across all models, reflected in high RMS calibration error scores. Models frequently provide incorrect answers
> with high confidence on HLE, failing to recognize when questions exceed their capabilities.

The calibration error is RMS (root-mean-square) over confidence bins, following the implementation
from Hendrycks et al. [24]. High calibration error combined with near-floor accuracy is the
operational signature of hallucination under uncertainty: models assert false answers confidently
rather than abstaining.

[Verified from PDF on 2026-05-12] Added §3.1-format, §3.2-review, §4.2-results. Abstract verified verbatim.
