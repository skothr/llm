---
paper_key: willard2023-outlines
title: "Efficient Guided Generation for Large Language Models"
authors: Willard, Louf
year: 2023
venue: arXiv (Outlines library)
arxiv: 2307.09702
local_pdf: null
type: excerpts
note: "Excerpts from arXiv abstract page (verbatim) plus methodology summary reproduced in subsequent constrained-decoding papers (XGrammar §2; SGLang §4 references). PDF not yet downloaded."
---

# Excerpts — Willard & Louf 2023, "Efficient Guided Generation for Large Language Models"

## Abstract — FSM-over-vocabulary reformulation {#abstract}

> In this article we describe an efficient approach to guiding
> language model text generation with regular expressions and
> context-free grammars. Our approach adds little to no overhead to
> the token sequence generation process, and makes guided generation
> feasible in practice. An implementation is provided in the open
> source Python library **Outlines**.
>
> Drawing parallels with the (random variate) sampling literature, we
> reformulate the problem of neural text generation in terms of
> transitions between the states of a finite-state machine. This
> framework leads to an efficient approach to guiding text generation
> with regular expressions and context-free grammars by allowing the
> construction of an **index over a language model's vocabulary**.
> The approach is **model agnostic**, allows one to enforce
> domain-specific knowledge and constraints, and enables the
> construction of reliable interfaces by guaranteeing the structure
> of the generated text.

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
