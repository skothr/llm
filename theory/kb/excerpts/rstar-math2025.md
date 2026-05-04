---
paper_key: rstar-math2025
title: "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking"
authors: Guan, Zhang, Wang, Liu, et al. (Microsoft)
year: 2025
venue: arXiv
arxiv: 2501.04519
local_pdf: null
type: excerpts
note: |
  Verbatim abstract from the arXiv abstract page (v2, Jan 2025).
  PDF was not downloaded during this Phase 2 pass (sandbox/curl
  restriction); body-section quotes are pending PDF acquisition.
  Section anchors are stable for follow-up population.
---

# Excerpts — Guan et al. 2025, "rStar-Math"

## Abstract — verbatim {#abstract}

> We present rStar-Math to demonstrate that small language models
> (SLMs) can rival or even surpass the math reasoning capability of
> OpenAI o1, without distillation from superior models. rStar-Math
> achieves this by exercising "deep thinking" through Monte Carlo Tree
> Search (MCTS), where a math policy SLM performs test-time search
> guided by an SLM-based process reward model. rStar-Math introduces
> three innovations to tackle the challenges in training the two SLMs:
> (1) a novel code-augmented CoT data sythesis [sic] method, which
> performs extensive MCTS rollouts to generate step-by-step verified
> reasoning trajectories used to train the policy SLM; (2) a novel
> process reward model training method that avoids naïve step-level
> score annotation, yielding a more effective process preference model
> (PPM); (3) a self-evolution recipe in which the policy SLM and PPM
> are built from scratch and iteratively evolved to improve reasoning
> capabilities. Through 4 rounds of self-evolution with millions of
> synthesized solutions for 747k math problems, rStar-Math boosts
> SLMs' math reasoning to state-of-the-art levels.

## Headline numbers {#headline}

- **Qwen2.5-Math-7B**: 58.8% → 90.0% on MATH benchmark (after 4
  evolution rounds).
- **Phi3-mini-3.8B**: 41.4% → 86.4% on MATH.
- Surpasses o1-preview on MATH by **+4.5%** (Qwen2.5-Math-7B) and
  **+0.9%** (Phi3-mini-3.8B).
- AIME 2024: **53.3%** (8/15), top 20% of high-school students.
- Training corpus: **747k math problems**, millions of synthesised
  solutions across 4 self-evolution rounds.

## §3 Method — three innovations {#sec-3}

[NOTE — pdf-not-available] Targets: the three innovations enumerated
in the abstract:

1. **Code-augmented CoT synthesis**: MCTS rollouts where reasoning
   steps include executable Python that validates intermediate
   results before continuing — only steps whose code-checks pass are
   retained.
2. **Process Preference Model (PPM)**: a process reward model trained
   from preferences over MCTS-discovered step pairs, rather than
   from naïve step-level scores.
3. **Self-evolution**: a 4-round loop in which the policy SLM, the
   PPM, and the synthesised data co-evolve.

## §3.x MCTS details {#sec-3-mcts}

[NOTE — pdf-not-available] Targets: the exact UCT formula used for
selection; expansion branching factor; rollout depth; how PPM scores
serve as the simulation value vs terminal verification.

## §4 Empirical results {#sec-4}

[NOTE — pdf-not-available] Targets: the per-benchmark and per-base-
model gains tables; comparison to o1-preview / o1-mini / DeepSeek-R1
(if available at submission time); the self-evolution-round-by-round
progression chart.

## Notes for the populator

- The "PPM" terminology (process *preference* model) is a deliberate
  contrast with PRM (process *reward* model); the training objective
  differs (preferences vs absolute labels). Note distinction when
  populating §3.
- The term "deep thinking" used in the title is informal; the
  mechanism is MCTS over reasoning steps with PPM-guided value
  estimates.
- Download command: `curl -sL -o theory/sources/papers/rstar-math2025
  .pdf "https://arxiv.org/pdf/2501.04519"`.
