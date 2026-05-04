---
paper_key: phan2025-hle
title: "Humanity's Last Exam"
authors: Phan, Gabriel, Pareyat, et al. (Scale AI / Center for AI Safety)
year: 2025
venue: arXiv (Scale AI / CAIS)
arxiv: 2501.14249
local_pdf: null
type: excerpts
note: Excerpts from arXiv abstract (verbatim). Multimodal subset percentages tagged `[PDF-VERIFY]`.
---

# Excerpts — Phan et al. 2025, "Humanity's Last Exam" (HLE)

## Abstract — the saturation framing {#abstract}

The motivation paragraph names the gap that HLE fills:

> LLMs now achieve over 90% accuracy on popular benchmarks like MMLU,
> limiting informed measurement of state-of-the-art LLM capabilities.

Scale and scope:

> HLE consists of 2,500 questions across dozens of subjects, including
> mathematics, humanities, and the natural sciences.

The framing as **a deliberate frontier-difficulty target**:

> the final closed-ended academic benchmark of its kind with broad
> subject coverage.

## Question format and verifier discipline {#sec-format}

> a known solution that is unambiguous and easily verifiable, but
> cannot be quickly answered via internet retrieval.

This is the load-bearing methodological claim — the benchmark
shares GPQA's "Google-proof" property by construction, applied across
many more domains than GPQA's bio/phys/chem.

The format combines multiple-choice and short-answer items, with
multimodal items (images) for some questions; the verifier is
either choice-equality or string-equality-after-normalization,
*not* LLM-as-judge.

## Capability gap {#sec-capability-gap}

> State-of-the-art LLMs demonstrate low accuracy and calibration on
> HLE, highlighting a significant gap between current LLM capabilities
> and the expert human frontier on closed-ended academic questions.

`[PDF-VERIFY]` Specific launch-time numerical gap to expert humans
not enumerated in the abstract section sourced here; Phase 1 sweep
notes top score ~65% as of Feb 2026.
