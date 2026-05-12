---
paper_key: jimenez2024-swebench
title: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"
authors: Jimenez, Yang, Wettig, Yao, Pei, Press, Narasimhan
year: 2024
venue: ICLR 2024
arxiv: 2310.06770
local_pdf: theory/sources/papers/jimenez2024-swebench.pdf
type: excerpts
---

# Excerpts — Jimenez et al. 2024, "SWE-bench"

## Abstract {#abstract}

> Language models have outpaced our ability to evaluate them effectively,
> but for their future development it is essential to study the frontier
> of their capabilities. We find real-world software engineering to be a
> rich, sustainable, and challenging testbed for evaluating the next
> generation of language models. To this end, we introduce SWE-bench, an
> evaluation framework consisting of 2,294 software engineering problems
> drawn from real GitHub issues and corresponding pull requests across 12
> popular Python repositories. Given a codebase along with a description
> of an issue to be resolved, a language model is tasked with editing the
> codebase to address the issue. Resolving issues in SWE-bench frequently
> requires understanding and coordinating changes across multiple
> functions, classes, and even files simultaneously, calling for models
> to interact with execution environments, process extremely long contexts
> and perform complex reasoning that goes far beyond traditional code
> generation tasks. Our evaluations show that both state-of-the-art
> proprietary models and our fine-tuned model SWE-Llama can resolve only
> the simplest issues. The best-performing model, Claude 2, is able to
> solve a mere 1.96% of the issues. Advances on SWE-bench represent steps
> towards LMs that are more practical, intelligent, and autonomous.

## §2.1 Benchmark construction — the 3-stage pipeline {#sec-2-1-construction}

> SWE-bench is a benchmark featuring GitHub issues from popular
> repositories that report bugs or request new features, and pull
> requests that make changes to the repository to resolve these issues.
> The task is to generate a pull request that addresses a given issue
> and passes tests related to the issue.

The construction pipeline (§2.1, verbatim stage descriptions):

> **Stage I: Repo selection and data scraping.** We start by collecting
> pull requests (PRs) from 12 popular open-source Python repositories on
> GitHub, producing about ~90,000 PRs in total. We focus on popular
> repositories as they tend be better maintained, have clear contributor
> guidelines, and have better test coverage. Each PR has an associated
> codebase specified by it's base commit.
>
> **Stage II: Attribute-based filtering.** We create candidate tasks by
> selecting the merged PRs that (1) resolve a GitHub issue and (2) make
> changes to the test files of the repository, which indicates that the
> user likely contributed tests to check whether the issue has been
> resolved.
>
> **Stage III: Execution-based filtering.** For each candidate task, we
> apply the PR's test content, and log the associated test results before
> and after the PR's other content is applied. We filter out task
> instances without at least one test where its status changes from a
> fail to pass (henceforth referred to as fail-to-pass test). We also
> filter out instances that result in installation or runtime errors.

> Through these stages of filtering, the original 90,000 PRs are
> filtered down to the 2,294 task instances which comprise SWE-bench.

## §2.2 Task formulation and evaluation metrics {#sec-2-2-eval}

> **Model input.** A model is given an issue text description and a
> complete codebase. The model is then tasked to make an edit to the
> codebase to resolve the issue. In practice, we represent edits as
> patch files, which specify which lines in the codebase to modify in
> order to resolve the issue.
>
> **Evaluation metrics.** To evaluate a proposed solution, we apply the
> generated patch, using unix's patch program, to the codebase and then
> execute the unit and system tests associated with the task instance. If
> the patch applies successfully and all of these tests pass we consider
> the proposed solution to have successfully resolved the issue. The
> metric for our benchmark is the percentage of task instances that are
> resolved.

The test-set structure that underlies the evaluation (from §2.3):

> For each task instance, there is at least one fail-to-pass test which
> was used to test the reference solution, and 40% of instances have at
> least two fail-to-pass tests. These tests evaluate whether the model
> addressed the problem in the issue. In addition, a median of 51
> additional tests run to check whether prior functionality is properly
> maintained.

[INTUITION: "fail-to-pass" tests are the ground truth that the bug is
fixed; "pass-to-pass" tests are the regression guard that no working
feature was broken. A task is resolved iff both sets pass after patching.]

## §5 Headline results — frontier models at single-digit resolution {#sec-5-results}

> We summarize models' performance using BM25 retrieval in Table 5.
> Across the board, models struggle significantly to resolve issues. The
> best performing model, Claude 2, is only able to resolve 1.96% of the
> issues.

Full result table from Table 5 (BM25 retrieval setting, verbatim):

| Model | % Resolved (SWE-bench) | % Apply | % Resolved (Lite) | % Apply (Lite) |
|---|---|---|---|---|
| Claude 3 Opus | 3.79 | 46.56 | 4.33 | 51.67 |
| Claude 2 | 1.97 | 43.07 | 3.00 | 33.00 |
| ChatGPT-3.5 | 0.17 | 26.33 | 0.33 | 10.00 |
| GPT-4-turbo | 1.31 | 26.90 | 2.67 | 29.67 |
| SWE-Llama 7b | 0.70 | 51.74 | 1.33 | 38.00 |
| SWE-Llama 13b | 0.70 | 53.62 | 1.00 | 38.00 |

Context-length finding (§5, verbatim):

> [A]s total context length increases, Claude 2's performance drops
> considerably; behavior that is also observed in other models. In our
> evaluation settings, models see a lot of code that may not be directly
> related to solving the issue at hand, and they seem to frequently
> struggle with localizing problematic code needing to be updated.

## SWE-bench Verified — derivation

`[SPECULATION: this is the OpenAI follow-up (Aug 2024), not in the
original ICLR paper]` OpenAI filtered the original 2,294 task pool down
to 500 human-validated instances. SWE-bench Verified is the de-facto
leaderboard target as of 2025; original SWE-bench is treated as
"developer testing only."

---

[Verified from PDF on 2026-05-12] Added §2.1 (3-stage construction
pipeline, #sec-2-1-construction), §2.2 (task formulation + evaluation
metrics, #sec-2-2-eval), §5 (headline results table + context finding,
#sec-5-results). Abstract replaced with verbatim PDF text (prior version
was a partial excerpt missing the full abstract). `note:` front-matter
field removed; `local_pdf:` was already set.
