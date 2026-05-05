---
paper_key: chao2024-jailbreakbench
title: "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models"
authors: Chao, Debenedetti, Robey, Andriushchenko, Croce, Sehwag, Dobriban, Flammarion, Pappas, Tramer, Hassani, Wong
year: 2024
venue: NeurIPS 2024 D&B
arxiv: 2404.01318
local_pdf: null
type: excerpts
note: PDF not yet downloaded; excerpts from arXiv 2404.01318 abstract (verified via WebSearch 2026-05-04). Used in `kb/notes/alignment/safety-evaluation.md` §2.1.
---

# Excerpts — Chao et al. 2024, "JailbreakBench"

## Abstract — full text {#abstract}

> Jailbreak attacks cause large language models (LLMs) to generate
> harmful, unethical, or otherwise objectionable content. Evaluating
> these attacks presents a number of challenges, which the current
> collection of benchmarks and evaluation techniques do not
> adequately address. First, there is no clear standard of practice
> regarding jailbreaking evaluation. Second, existing works compute
> costs and success rates in incomparable ways. And third, numerous
> works are not reproducible, as they withhold adversarial prompts,
> involve closed-source code, or rely on evolving proprietary APIs.
>
> To address these challenges, we introduce JailbreakBench, an
> open-sourced benchmark with the following components:
> (1) an evolving repository of state-of-the-art adversarial prompts,
> which we refer to as *jailbreak artifacts*;
> (2) a jailbreaking dataset comprising 100 behaviors — both original
> and sourced from prior work — which align with OpenAI's usage
> policies, along with a standardized evaluation framework and a
> leaderboard that tracks the performance of attacks and defenses for
> various LLMs.

(Verified via WebSearch 2026-05-04 against arXiv 2404.01318 abstract.)

## Structural commitments {#sec-structure}

- **100 behaviors** (original + sourced from prior work), aligned
  with OpenAI's usage policies.
- **Evolving repository** of jailbreak artifacts (the actual
  adversarial prompts).
- **Standardized judge** with published evaluation prompts (LLM-as-
  judge using a frontier model).
- **Public leaderboard** tracking attacks and defenses across LLMs.

GitHub: https://github.com/JailbreakBench/jailbreakbench

## Comparison with HarmBench

HarmBench has broader behavior coverage (510 vs. 100) and an attack-
method bake-off across 18 methods. JailbreakBench has deeper
*artifact reproducibility* (publishing the actual successful
jailbreak prompts) and a public leaderboard. The two are
complementary: HarmBench for breadth + attack/defense codevelopment;
JailbreakBench for reproducibility + community leaderboard.

## Methodological contribution

JailbreakBench's primary contribution is **reproducibility
infrastructure**, not a new attack or defense. The "evolving
repository of jailbreak artifacts" is in some sense an open-source
analog of an adversarial-attack CVE database — when a new jailbreak
prompt is found, it gets added to the repo with provenance,
evaluation conditions, and judge verdict, allowing later defenses to
be tested directly against the historical attack record.

[PDF-VERIFY] The exact judge-LLM identity (e.g., GPT-4 family),
the specific judge-prompt phrasing, and the leaderboard pinning
methodology require PDF verification. The structural commitments
above are sufficient to ground the §2.1 references in
`kb/notes/alignment/safety-evaluation.md`.
