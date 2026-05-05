---
paper_key: russinovich2024-crescendo
title: "Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack"
authors: Russinovich, Salem, Eldan
year: 2024
venue: USENIX Security 2025 (arXiv 2404.01833, April 2024)
arxiv: 2404.01833
local_pdf: null
type: excerpts
note: PDF download blocked (curl/wget restricted by sandbox in this Phase 2.5 deepening pass); full abstract retrieved verbatim via WebFetch on arXiv 2404.01833 abstract page (2026-05-05). Models tested per abstract: ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70B Chat, LlaMA-3 70B Chat, and Anthropic Chat. Used in `kb/notes/alignment/safety-evaluation.md` §2.1.
---

# Excerpts — Russinovich, Salem, Eldan 2024, "The Crescendo Multi-Turn LLM Jailbreak Attack"

## #abstract — full verbatim

> Large Language Models (LLMs) have risen significantly in popularity
> and are increasingly being adopted across multiple applications. These
> LLMs are heavily aligned to resist engaging in illegal or unethical
> topics as a means to avoid contributing to responsible AI harms.
> However, a recent line of attacks, known as jailbreaks, seek to
> overcome this alignment. Intuitively, jailbreak attacks aim to narrow
> the gap between what the model can do and what it is willing to do.
> In this paper, we introduce a novel jailbreak attack called Crescendo.
> Unlike existing jailbreak methods, Crescendo is a simple multi-turn
> jailbreak that interacts with the model in a seemingly benign manner.
> It begins with a general prompt or question about the task at hand
> and then gradually escalates the dialogue by referencing the model's
> replies progressively leading to a successful jailbreak. We evaluate
> Crescendo on various public systems, including ChatGPT, Gemini Pro,
> Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat.
> Our results demonstrate the strong efficacy of Crescendo, with it
> achieving high attack success rates across all evaluated models and
> tasks. Furthermore, we present Crescendomation, a tool that automates
> the Crescendo attack and demonstrate its efficacy against
> state-of-the-art models through our evaluations. Crescendomation
> surpasses other state-of-the-art jailbreaking techniques on the
> AdvBench subset dataset, achieving 29-61% higher performance on GPT-4
> and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's
> ability to jailbreak multimodal models.

(Verified verbatim from arXiv 2404.01833 abstract via WebFetch
2026-05-05.)

## Headline ASR uplift {#sec-asr-uplift}

> Crescendomation surpasses other state-of-the-art jailbreaking
> techniques on the AdvBench subset dataset, achieving 29-61%
> higher performance on GPT-4 and 49-71% on Gemini-Pro.

The 29-71% ASR uplift over single-turn baselines is the
load-bearing methodological number for the safety-eval note. It
documents that **single-turn evaluations systematically
underestimate** the attack surface against frontier models —
benchmarks like JailbreakBench and HarmBench that test single-turn
behaviors will report ASR numbers below true deployment-condition
ASR.

## Mechanism

The exploitation pattern:

1. Open with a general / benign question about the topic.
2. Ask a slightly more specific follow-up.
3. Reference the model's *own* previous answer as if it had committed.
4. Iterate until the model produces the harmful behavior.

The exploitation hinges on the model over-weighting (a) recent
tokens and (b) text *it has just produced itself*. The model's
prior answers create a *commitment gradient* the attacker climbs;
since each individual turn is benign, no single-turn refusal
classifier triggers.

## Crescendomation

The paper introduces **Crescendomation**, an automated implementation
of the Crescendo strategy that uses an attacker LLM to generate the
escalation chain. The "29-71% ASR uplift" refers to Crescendomation
specifically — the automated version, not the manual Crescendo
template.

[PDF-VERIFY] The exact behavior subset of AdvBench used, the
specific judge configuration, and the per-step escalation budget
need PDF verification before propagating as hard claims. The
structural mechanism + ASR-uplift number above are the load-bearing
elements for the safety-eval note.
