---
paper_key: lm-eval-harness
title: "EleutherAI Language Model Evaluation Harness"
authors: Gao, Tow, Abbasi, Biderman, Black, et al. (EleutherAI)
year: 2021–ongoing
venue: GitHub (open-source infrastructure)
arxiv: null
local_pdf: null
url: https://github.com/EleutherAI/lm-evaluation-harness
type: excerpts
note: Excerpts from the GitHub project README (verbatim, treated as Tier-B authoritative for the project itself; software documentation, not peer-reviewed paper).
---

# Excerpts — lm-evaluation-harness (EleutherAI)

## Project framing {#sec-framing}

> [the harness is] a unified framework to test generative language
> models on a large number of different evaluation tasks

Scale signal:

> over 60 standard academic benchmarks ... with hundreds of subtasks
> and variants

## Reproducibility discipline {#sec-reproducibility}

> evaluation with publicly available prompts [guarantees] reproducibility
> and comparability between papers.

This is the load-bearing methodological pitch: HELM-style standardized
prompts, but as open-source infrastructure rather than as a curated
benchmark. Tasks are YAML files with Jinja2 templates — anyone can
add a task, anyone can re-run someone else's eval.

## Request type taxonomy — the API surface {#sec-request-types}

> loglikelihood, loglikelihood_rolling [for models providing logprobs]
> generate_until [for all models]

This is the load-bearing structural claim: every benchmark is reducible
to one of three request types:

1. **`loglikelihood(context, continuation)`** → returns
   $\log P(\text{continuation} \mid \text{context})$. Used for
   multiple-choice scoring (compute logprob of each option, take
   argmax; this is the standard MMLU evaluation method).
2. **`loglikelihood_rolling(text)`** → perplexity over a long sequence.
   Used for language modeling benchmarks (WikiText, etc.).
3. **`generate_until(context, stop_seqs)`** → free-form generation.
   Used for benchmarks where the answer is parsed from generated text
   (math word problems, code generation).

Critical methodological consequence: a benchmark scored via (1) and
the same benchmark scored via (3) can give **substantially different
numbers** on the same model — see `kb/notes/evaluation/eval-methodology.md`
for the prompt-format/answer-extraction sensitivity.

## Adoption signal {#sec-adoption}

> the backend for Hugging Face's Open LLM Leaderboard

> adopted in hundreds of academic papers, and is used internally by
> organizations including NVIDIA, Cohere, BigScience, and Mosaic ML

This is the de-facto "standard" the field has converged on for
open-model leaderboard evaluation. Closed labs (OpenAI, Anthropic,
Google) maintain their own internal eval harnesses, which is a
methodological asymmetry — closed-model numbers are not always
comparable to open-model numbers because the harness differs.
