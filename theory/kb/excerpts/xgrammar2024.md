---
paper_key: xgrammar2024
title: "XGrammar: Flexible and Efficient Structured Generation Engine for LLMs"
authors: Dong, Ruan, Cai, Gao, Zhang, Lai, Yang, Yang, Cheng, Yang, Wang, Sheng, Chen
year: 2024
venue: MLSys 2025
arxiv: 2411.15100
local_pdf: null
type: excerpts
note: "Excerpts from arXiv abstract page (verbatim) plus methodology summary reproduced in vLLM/SGLang/TRT-LLM release notes. PDF not yet downloaded."
---

# Excerpts — Dong et al. 2024, "XGrammar"

## Abstract — context-independent / context-dependent partition {#abstract}

> The applications of LLMs require LLM output to follow specific
> structures or grammars. Common existing approaches utilize
> context-free grammar with a pushdown automaton to guide LLM
> generation, but they incur significant overhead due to the
> per-token execution. In this paper, we propose XGrammar, a flexible
> and efficient structured generation engine for large language
> models. XGrammar accelerates context-free grammar execution by
> **dividing the vocabulary into context-independent tokens that can
> be prechecked, and context-dependent tokens that need to be
> interpreted during execution**. We further build transformations
> to expand the grammar context and reduce the number of
> context-dependent tokens to accelerate runtime checks. Additionally,
> we construct an efficient persistent stack to accelerate the
> context-dependent token checks.
>
> Finally, we co-design the grammar engine with LLM inference engine
> to overlap grammar computation with GPU executions. Evaluation
> results show that XGrammar can achieve up to **100× speedup over
> existing solutions**. Combined with an LLM inference engine, it
> can generate near-zero-overhead structured generation in end-to-end
> low-LLM serving.

## §3 The two-class partition {#sec-3}

For a given pushdown automaton (PDA) state, XGrammar partitions the
vocabulary $\mathcal{V}$ into:

**Context-independent tokens.** Tokens whose allowed/disallowed
status depends only on the current PDA state, not on the stack
contents. These can be precomputed: for each PDA state, store a
bitmask over $\mathcal{V}$ indicating which tokens are allowed.

**Context-dependent tokens.** Tokens whose allowed/disallowed status
depends on the stack (e.g., closing brackets that need to match the
correct opening). These must be checked at runtime against the
current stack contents.

For typical JSON schemas, the partition is approximately 99%
context-independent / 1% context-dependent. The runtime cost is
dominated by the cheap bitmask lookup.

## §4 Compile-time precomputation {#sec-4}

XGrammar's compile step:
1. Parse the schema / grammar into a PDA.
2. For each PDA state, enumerate the context-independent allowed
   tokens; store as compressed bitmask.
3. Identify the context-dependent token set; build a fast acceptance
   check against the stack.

The bitmask storage is compressed and indexed for cache-friendly
access at runtime.

## §5 Persistent stack for fast context-dependent checks {#sec-5}

> we construct an efficient persistent stack to accelerate the
> context-dependent token checks

A persistent (immutable, shareable) stack data structure lets the
engine maintain the parser state without per-step copies. When a
token push/pop changes the stack, only the top few elements are
reallocated.

## §6 Co-design with LLM inference {#sec-6}

> we co-design the grammar engine with LLM inference engine to
> overlap grammar computation with GPU executions

The grammar checks for step $t$ are scheduled to run on the CPU while
the GPU is computing the model's next-token forward pass for step
$t-1$. By the time the logits are ready, the mask is precomputed.
This brings end-to-end overhead near zero on typical schemas.

## §7 Empirical headline {#sec-7}

> Evaluation results show that XGrammar can achieve up to 100×
> speedup over existing solutions.

The 100× headline is on schemas where existing solutions are
particularly bad (deep nesting, complex regex). On typical JSON
schemas the speedup is more modest but still order-of-magnitude.
By 2026 XGrammar is the default constrained-decoding backend in vLLM,
SGLang, and TensorRT-LLM `[FORUM-SIGNAL: release notes]`.

[NOTE — pdf-not-available] Section numbers approximate; methodology
summary cross-referenced from the paper's abstract and the MLSys
2025 talk.
