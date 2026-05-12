---
paper_key: jimenez2024-swebench
title: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"
authors: Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan
year: 2024
venue: ICLR 2024
arxiv: 2310.06770
local_pdf: theory/sources/papers/jimenez2024-swebench.pdf
type: excerpts
note: Excerpts compiled from the arXiv abstract page (verbatim) plus widely cited details about the evaluation protocol from the public companion site (https://www.swebench.com/) and the README of https://github.com/princeton-nlp/SWE-bench. PDF not yet verified — claims tagged below as `[PDF-VERIFY]` should be confirmed against the v3 PDF before propagating into LaTeX series.
---

# Excerpts — Jimenez et al. 2024, "SWE-bench"

## Abstract — the framing problem {#abstract}

> Language models have outpaced our ability to evaluate them effectively,
> but for their future development it is essential to study the frontier
> of their capabilities. We find real-world software engineering to be a
> rich, sustainable, and challenging testbed for evaluating the next
> generation of language models.

Headline scale:

> SWE-bench [is] an evaluation framework consisting of 2,294 software
> engineering problems drawn from real GitHub issues and corresponding
> pull requests across 12 popular Python repositories.

## Task formulation — what the model sees, what it produces {#sec-task}

> Given a codebase along with a description of an issue to be resolved,
> a language model is tasked with editing the codebase to address the
> issue.

Difficulty driver:

> Resolving issues in SWE-bench frequently requires understanding and
> coordinating changes across multiple functions, classes, and even files
> simultaneously, calling for models to interact with execution environments,
> process extremely long contexts and perform complex reasoning that goes
> far beyond traditional code generation tasks.

## Evaluation protocol — `FAIL_TO_PASS` and `PASS_TO_PASS` {#sec-eval-protocol}

`[PDF-VERIFY]` The following description is the established public
protocol used by every SWE-bench leaderboard submission and described
in the dataset README at github.com/princeton-nlp/SWE-bench:

For each task instance, the dataset ships two test sets derived from
the canonical PR's test diff:

- **`FAIL_TO_PASS`** — tests that **fail on the pre-PR codebase** and
  must **pass after the candidate patch is applied**. Empty FAIL_TO_PASS
  is rejected during construction (the PR did not introduce a regression
  test).
- **`PASS_TO_PASS`** — tests that **pass on the pre-PR codebase** and
  must **continue to pass after the patch is applied**. Guards against
  patches that fix the bug by breaking unrelated functionality.

A task instance is **resolved** iff (i) the patch applies cleanly, and
(ii) every FAIL_TO_PASS test transitions $\text{fail} \to \text{pass}$
**and** every PASS_TO_PASS test stays $\text{pass} \to \text{pass}$.
This is the basis of the `% resolved` headline number on the leaderboard.

Two reported quantities:

- `% applied` = fraction of generated patches that apply cleanly with
  `git apply` (a strict prerequisite for the resolved check).
- `% resolved` = fraction of task instances that pass the full
  FAIL_TO_PASS + PASS_TO_PASS regression check.

## Headline result at launch {#sec-headline}

`[PDF-VERIFY]` From the v3 paper / website at submission time:

> Claude 2 ... resolves only 1.96% of the issues.

This number — single-digit pass rate at the original launch — is the
key reason SWE-bench was treated as the post-MMLU frontier-eval candidate
for coding agents.

## SWE-bench Verified — derivation

`[PDF-VERIFY: this is the OpenAI follow-up, not the original ICLR paper]`
OpenAI Aug 2024 release filtered the original 2,294 task pool down to
**500 human-validated instances** by having professional software engineers
label each issue for: (i) whether the issue specification was unambiguous
(many original issues were under-specified), (ii) whether the
FAIL_TO_PASS test set was a fair specification of the bug fix, and (iii)
whether environment setup was reproducible. Approximately one-third of
original tasks were rejected at this filter. SWE-bench Verified is the
de-facto leaderboard target as of 2025; original SWE-bench is treated as
"developer testing only."
