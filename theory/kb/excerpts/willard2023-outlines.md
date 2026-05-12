---
paper_key: willard2023-outlines
title: "Efficient Guided Generation for Large Language Models"
authors: Willard, Louf
year: 2023
venue: arXiv (Outlines library)
arxiv: 2307.09702
local_pdf: theory/sources/papers/willard2023-outlines.pdf
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

> our indexing approach can also be extended to CFGs
> and LALR(1) parsers to allow for efficient guided generation
> according to popular data formats and programming languages (e.g.
> JSON, Python, SQL, etc.).

For CFGs the FSM-state-only index is insufficient (the parser stack
is part of the context). The Outlines extension uses an LALR(1)
parser whose state captures enough of the stack contents that the
index can still be precomputed for the common parsing-state /
stack-summary combinations.

## §5 Discussion {#sec-5}

> [It adds little overhead to the token sequence generation process
> and] significantly outperforms existing solutions.

PDF §5 is titled "Discussion," not "Empirical." The quantitative
comparison (Outlines vs. Guidance latency) is in §3.2 "Comparison
with current methods," which shows runtime plots; §5 is qualitative
discussion of memory/trade-offs and broader implications. The phrase
"hundreds-of-µs to single-digit µs" does not appear in the PDF and
has been removed. The overhead/outperforms claim is from the Abstract.

[Verified from PDF on 2026-05-12: arXiv v4 (2307.09702v4). Abstract
verbatim confirmed. §3 = "Iterative FSM Processing and Indexing"
(vocabulary index map σ : Q → P(V), O(1) retrieval). §4 = "Extensions
to Iterative Parsing" / §4.1 "Pushdown Automata Formulation" (PDA
6-tuple, LALR(1) extension). §5 = "Discussion" (memory trade-offs,
no raw latency numbers). Corrected: (1) §4 quoted text was "The FSM
approach" — PDF says "our indexing approach"; fixed to verbatim. (2)
§5 section title was "Empirical" — PDF §5 is "Discussion"; the
empirical benchmark is §3.2. (3) "hundreds-of-µs to single-digit µs"
claim not present in PDF; removed.]
