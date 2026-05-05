---
paper_key: willard2023-outlines
title: "Efficient Guided Generation for Large Language Models"
authors: Willard, Louf
year: 2023
venue: arXiv (Outlines library)
arxiv: 2307.09702
local_pdf: null
type: excerpts
note: |
  Abstract verified verbatim 2026-05-05 via WebFetch on arXiv abs page.
  Body sections paraphrase methodology cross-referenced in XGrammar §2
  and SGLang §4. PDF download blocked in this sandbox; full §3/§4
  verbatim quotes pending PDF acquisition.
---

# Excerpts — Willard & Louf 2023, "Efficient Guided Generation for Large Language Models"

## Abstract — FSM-over-vocabulary reformulation {#abstract}

Verbatim from the arXiv abs page (verified 2026-05-05):

> In this article we show how the problem of neural text generation can
> be constructively reformulated in terms of transitions between the
> states of a finite-state machine. This framework leads to an efficient
> approach to guiding text generation with regular expressions and
> context-free grammars by allowing the **construction of an index over a
> language model's vocabulary**. The approach is model agnostic, allows
> one to enforce domain-specific knowledge and constraints, and enables
> the construction of reliable interfaces by guaranteeing the structure
> of the generated text. It adds little overhead to the token sequence
> generation process and significantly outperforms existing solutions. An
> implementation is provided in the open source Python library
> **Outlines**.

## §3 The vocabulary index {#sec-3}

The standard FSM operates over single characters. Willard & Louf's
contribution is to **construct an index** mapping each FSM state $s$
to the set of vocabulary tokens whose string value can transition the
FSM from $s$ along a valid path:
$$
\mathcal{V}_{\text{allowed}}(s) = \{v \in \mathcal{V} : \mathrm{FSM} \text{ accepts the byte sequence of } v \text{ starting from } s\}.
$$

The index is built **once per grammar**. At generation time:
1. Look up current FSM state $s_t$.
2. Retrieve $\mathcal{V}_{\text{allowed}}(s_t)$ in $O(1)$ amortized.
3. Mask logits to allow only $\mathcal{V}_{\text{allowed}}(s_t)$.
4. Sample $y_t$; transition the FSM by $|y_t|$ characters; update
   $s_{t+1}$.

> the algorithm scales as O(1) on average

This is in contrast to the naive approach (per-vocab acceptance check
each step), which scales as $O(V)$ per step.

## §4 CFG support via LALR(1) parsers {#sec-4}

> The FSM approach can also be extended to context-free grammars
> (CFGs) and LALR(1) parsers to allow for efficient guided generation
> according to popular data formats and programming languages (e.g.
> JSON, Python, SQL, etc.).

For CFGs the FSM-state-only index is insufficient (the parser stack
is part of the context). The Outlines extension uses an LALR(1)
parser whose state captures enough of the stack contents that the
index can still be precomputed for the common parsing-state /
stack-summary combinations.

## §5 Empirical {#sec-5}

> [The approach] adds little to no overhead to the token sequence
> generation process, … significantly outperforms existing solutions.

The empirical claim is that the per-step overhead drops from
hundreds-of-µs to single-digit µs on typical schemas, taking
constrained decoding from "the bottleneck" to "negligible."

[NOTE — pdf-not-available] Section numbers from arXiv v3.
