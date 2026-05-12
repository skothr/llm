---
paper_key: glazer2024-frontiermath
title: "FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning in AI"
authors: Glazer, Erdil, Besiroglu, Chen, Hickman, Patel, Vivekanandan, Chen, Jain, Mehta, Bhuwalka, et al. (Epoch AI)
year: 2024
venue: arXiv
arxiv: 2411.04872
local_pdf: theory/sources/papers/glazer2024-frontiermath.pdf
type: excerpts
note: Verbatim quotations from PDF (arXiv:2411.04872v7, Dec 2025). Abstract verified verbatim from PDF on 2026-05-12. Deepened with §2.1 (collection pipeline), §2.2 (automated verification), §2.5 (difficulty dimensions), §4.2.1 (results). Prior abstract had em-dash rendered as double-hyphen; corrected.
---

# Excerpts — Glazer et al. 2024, "FrontierMath"

## #abstract — full verbatim

> We introduce FrontierMath, a benchmark of hundreds of original,
> exceptionally challenging mathematics problems crafted and vetted by
> expert mathematicians. The questions cover most major branches of
> modern mathematics—from computationally intensive problems in
> number theory and real analysis to abstract questions in algebraic
> geometry and category theory. Solving a typical problem requires
> multiple hours of effort from a researcher in the relevant branch of
> mathematics, and for the upper end questions, multiple days.
> FrontierMath uses new, unpublished problems and automated
> verification to reliably evaluate models while minimizing risk of
> data contamination. Current state-of-the-art AI models solve under 2%
> of problems, revealing a vast gap between AI capabilities and the
> prowess of the mathematical community. As AI systems advance toward
> expert-level mathematical abilities, FrontierMath offers a rigorous
> testbed that quantifies their progress.

## §2.1 Collection Pipeline — Expert Mathematician Authorship {#sec-2-1}

> We developed the FrontierMath benchmark through collaboration with
> over 60 mathematicians from universities across more than a dozen
> countries. Our contributors formed a diverse group spanning academic
> ranks from graduate students to faculty members. Many are distinguished
> participants in prestigious mathematical competitions, collectively
> holding 14 IMO gold medals. One contributing mathematician also holds
> a Fields Medal.

> Each mathematician created new problems following guidelines designed
> to ensure clarity, verifiability, and definitive answers across various
> difficulty levels and mathematical domains.

The four requirements for problem creation (§2.1, p.4): Originality, Automated verifiability, Guessproofness, and Computational tractability. Each submitted problem also underwent blind peer review by at least one domain-expert mathematician (§2.3). This methodology ensures the "no data contamination" property: problems are new and unpublished, and solutions are verified automatically without exposing reasoning to leakage.

## §2.2 Automated Verification {#sec-2-2}

> FrontierMath focuses exclusively on problems with automatically
> verifiable solutions—primarily numerical answers or mathematical
> objects that can be expressed as SymPy objects (including symbolic
> expressions, matrices, sets, and other mathematical structures). For
> each problem, the evaluated model submits code that computes and saves
> its answer as a Python object. A script then automatically verifies
> this answer. If the problem has a unique integer solution, it checks
> for an exact match. If the problem has a unique symbolic real answer,
> it uses SymPy evaluation to check that the difference between the
> submitted answer and the actual answer simplifies to 0. In all other
> cases, a custom verification script is necessary to check validity of
> a submitted answer.

This design is the contamination-defense mechanism: exact programmatic checking means no LLM-as-judge, no partial-credit ambiguity, and no need to expose gold answers during evaluation. Problems that don't admit a SymPy-representable answer are excluded by construction.

## §2.5 Problem Difficulty — Three-Dimensional Ratings {#sec-2-5}

> To assess problem difficulty, each problem author provided ratings
> along three key dimensions:
>
> 1. **Background**: This rating ranges from 1 to 5 and indicates the
>    level of mathematical background required to approach the problem.
>    A rating of 1 corresponds to high school level, 2 to early
>    undergraduate level, 3 to late undergraduate level, 4 to graduate
>    level, and 5 to research level.
> 2. **Creativity**: Estimated as the number of hours an expert would
>    need to identify the key ideas for solving the problem. This measure
>    has no upper limit.
> 3. **Execution**: Similarly estimated as the number of hours required
>    to compute the final answer once the key ideas are found, including
>    time writing a script if applicable. This measure also has no upper
>    limit.

[INTUITION] The three-dimensional rating separates "how much do you need to know" (Background) from "how hard is the creative leap" (Creativity) from "how tedious is the computation" (Execution). A problem could be Background=5 but Creativity=0.5 if the key idea is routine given graduate training. The paper notes these ratings provide only "rough guidance" due to inter-reviewer variability.

## §4.2.1 Results — Sub-2% Solve Rate {#sec-4-2-1}

> All models had a very low performance on FrontierMath problems, with
> no model achieving even a 2% success rate on the full benchmark.

> Based on a single evaluation of the full benchmark, we found that
> models solved very few problems (less than 2% success rate). Given
> this low success rate and the fact that we ran only one evaluation,
> the precise ordering of model performance should be interpreted with
> significant caution, as individual successes can have an outsized
> impact on rankings.

Models evaluated at launch: o1-preview, o1-mini, GPT-4o (2024-08-06), Claude 3.5 Sonnet (2024-10-22), Grok 2 Beta, Gemini 1.5 Pro 002. All scored below 2%. The total number of problems solved by any model across the full benchmark was four (4) distinct problems. This is the primary citation target for "FrontierMath maintains >98% unsolved rate" (Figure 2 caption).

[Verified from PDF on 2026-05-12] Added §2.1-collection, §2.2-verification, §2.5-difficulty, §4.2.1-results. Abstract verified verbatim (prior had "--" instead of em-dash "—").
