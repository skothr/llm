---
paper_key: xie2024-osworld
title: "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments"
authors: Xie, Zhang, Chen, Lou, Liu, Hu, Xing, Hu, Lin, Liu, Yu
year: 2024
venue: NeurIPS 2024 (Datasets & Benchmarks)
arxiv: 2404.07972
local_pdf: null
type: excerpts
note: Excerpts from the arXiv abstract page (verbatim). Operating system, app coverage, and task-distribution numbers tagged `[PDF-VERIFY]`.
---

# Excerpts — Xie et al. 2024, "OSWorld"

## Abstract — what's new {#abstract}

> [autonomous agents that] accomplish complex computer tasks with minimal
> human interventions have the potential to transform human-computer
> interaction.

Scope and environments:

> OSWorld [is] a benchmark of 369 computer tasks involving real web and
> desktop apps in open domains, OS file I/O, and workflows spanning
> multiple applications.

> [OSWorld supports] various operating systems such as Ubuntu, Windows,
> and macOS.

## Task specification — initial state + executable evaluator {#sec-task-spec}

> Tasks derive from authentic use cases, with each including a detailed
> initial state setup configuration and a custom execution-based
> evaluation script for reliable, reproducible evaluation.

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
