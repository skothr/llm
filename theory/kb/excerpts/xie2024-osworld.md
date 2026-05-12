---
paper_key: xie2024-osworld
title: "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments"
authors: Xie, Zhang, Chen, Lou, Liu, Hu, Xing, Hu, Lin, Liu, Yu
year: 2024
venue: NeurIPS 2024 (Datasets & Benchmarks)
arxiv: 2404.07972
local_pdf: theory/sources/papers/xie2024-osworld.pdf
type: excerpts
note: PDF download blocked (curl/wget restricted by sandbox in this Phase 2.5 deepening pass); full abstract retrieved verbatim via WebFetch on arXiv 2404.07972 abstract page (2026-05-05). Operating system, app coverage, and task-distribution numbers tagged `[PDF-VERIFY]`.
---

# Excerpts — Xie et al. 2024, "OSWorld"

## #abstract — full verbatim

> Autonomous agents that accomplish complex computer tasks with minimal
> human interventions have the potential to transform human-computer
> interaction, significantly enhancing accessibility and productivity.
> However, existing benchmarks either lack an interactive environment
> or are limited to environments specific to certain applications or
> domains, failing to reflect the diverse and complex nature of
> real-world computer use, thereby limiting the scope of tasks and
> agent scalability. To address this issue, we introduce OSWorld, the
> first-of-its-kind scalable, real computer environment for multimodal
> agents, supporting task setup, execution-based evaluation, and
> interactive learning across various operating systems such as Ubuntu,
> Windows, and macOS. OSWorld can serve as a unified, integrated
> computer environment for assessing open-ended computer tasks that
> involve arbitrary applications. Building upon OSWorld, we create a
> benchmark of 369 computer tasks involving real web and desktop apps
> in open domains, OS file I/O, and workflows spanning multiple
> applications. Each task example is derived from real-world computer
> use cases and includes a detailed initial state setup configuration
> and a custom execution-based evaluation script for reliable,
> reproducible evaluation. Extensive evaluation of state-of-the-art
> LLM/VLM-based agents on OSWorld reveals significant deficiencies in
> their ability to serve as computer assistants. While humans can
> accomplish over 72.36% of the tasks, the best model achieves only
> 12.24% success, primarily struggling with GUI grounding and
> operational knowledge.

(Verified verbatim from arXiv 2404.07972 abstract via WebFetch
2026-05-05.)

## Task specification — initial state + executable evaluator {#sec-task-spec}

> Each task example is derived from real-world computer use cases and
> includes a detailed initial state setup configuration and a custom
> execution-based evaluation script for reliable, reproducible
> evaluation.

This **per-task evaluator script** is the methodological contrast with
GAIA (string-match) and AgentBench (turn-by-turn rubric): OSWorld checks
the **end-state of the OS environment** after the agent's action trace,
which means agents can take any path to the goal as long as the post-
state matches the spec.

## Headline gap {#sec-results}

> Human performance: over 72.36% of the tasks
> Best model performance: only 12.24% success

`[PDF-VERIFY]` The 12.24% number is for GPT-4V at submission time
(early 2024); subsequent results from agentic systems (e.g., Claude
3.5 Sonnet "computer use", Anthropic Oct 2024) push the number higher
but the human-AI gap remains the largest among published 2024
agentic benchmarks.

## Failure mode

> [agents] primarily struggling with GUI grounding and operational
> knowledge.

This is the load-bearing finding for the eval-methodology note:
multi-modal grounding (knowing what pixel to click) is bottlenecking
the metric, not high-level planning.
