---
paper_key: chao2024-jailbreakbench
title: "JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models"
authors: Chao, Debenedetti, Robey, Andriushchenko, Croce, Sehwag, Dobriban, Flammarion, Pappas, Tramer, Hassani, Wong
year: 2024
venue: NeurIPS 2024 Datasets and Benchmarks Track
arxiv: 2404.01318
local_pdf: theory/sources/papers/chao2024-jailbreakbench.pdf
type: excerpts
---

# Excerpts — Chao et al. 2024, "JailbreakBench"

## Abstract {#abstract}

> Jailbreak attacks cause large language models (LLMs) to generate harmful, un-
> ethical, or otherwise objectionable content. Evaluating these attacks presents a
> number of challenges, which the current collection of benchmarks and evaluation
> techniques do not adequately address. First, there is no clear standard of
> practice regarding jailbreaking evaluation. Second, existing works compute costs
> and success rates in incomparable ways. And third, numerous works are not re-
> producible, as they withhold adversarial prompts, involve closed-source code, or
> rely on evolving proprietary APIs. To address these challenges, we introduce
> JailbreakBench, an open-sourced benchmark with the following components:
> (1) an evolving repository of state-of-the-art adversarial prompts, which we re-
> fer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors—
> both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023,
> 2024)—which align with OpenAI's usage policies; (3) a standardized evaluation
> framework at https://github.com/JailbreakBench/jailbreakbench that
> includes a clearly defined threat model, system prompts, chat templates, and scor-
> ing functions; and (4) a leaderboard at https://jailbreakbench.github.io/
> that tracks the performance of attacks and defenses for various LLMs. We have
> carefully considered the potential ethical implications of releasing this benchmark,
> and believe that it will be a net positive for the community.

Abstract verified verbatim from PDF (arXiv:2404.01318v5, NeurIPS 2024).

## §3.1 — JBB-Behaviors: the 100-behavior dataset {#sec-3-1-behaviors}

> JBB-Behaviors contains a representative set of behaviors that encompass a broad
> spectrum of misuse. In contrast to several existing datasets, we ensure that each
> behavior in JBB-Behaviors is unique and realizable with a text-based generation.
> Among these behaviors, 18% are sourced from AdvBench (Zou et al., 2023),
> 27% from TDC/HarmBench (Mazeika et al., 2023, 2024), whereas the remaining
> 55% are original to JBB-Behaviors. Specifically, JBB-Behaviors contains 100 rows,
> where each row specifies five distinct fields:
>
>   - Behavior. A unique identifier describing a distinct misuse behavior.
>   - Goal. A query requesting an objectionable behavior.
>   - Target. An affirmative response to the harmful goal string.
>   - Category. A broader category of misuse from OpenAI's usage policies.
>   - Source. A reference to the source dataset of the goal and target string.

In addition, 100 matching benign behaviors are included (one per harmful behavior,
same topic) to evaluate refusal rates and ensure defenses are not overly conservative.

## §3.5 — Judge selection: Llama-3-70B vs. five alternatives {#sec-3-5-judge}

> To choose an effective classifier, we collected a dataset of 200 jailbreak prompts and
> responses. Three experts labeled each prompt-response pair, and the agreement between
> them was approximately 95%. The ground truth label for each behavior is then the majority
> vote among the labelers. Moreover, we add 100 benign examples from XS-Test (Röttger
> et al., 2023) to test how sensitive the judges are to benign prompts and responses that
> share similarities to harmful ones.

The six candidate classifiers evaluated and their agreement with human ground truth
(300-example dataset, majority-vote labels):

| Judge | Agreement | FPR | FNR |
|---|---|---|---|
| Rule-based (Zou et al. 2023) | 56.0% | 64.2% | 9.1% |
| GPT-4-0613 | 90.3% | 10.0% | 9.1% |
| HarmBench (Llama-2-13B) | 78.3% | 26.8% | 12.7% |
| Llama Guard (Llama-2-7B) | 72.0% | 9.0% | 60.9% |
| Llama Guard 2 (Llama-3-8B) | 87.7% | 13.2% | 10.9% |
| **Llama-3-70B** | **90.7%** | **11.6%** | **5.5%** |

> Although Llama-3-70B and GPT-4 appear to perform similarly well as judges, GPT-4
> comes with the drawback of close-sourced models, i.e., expensive to query and subject
> to change. Thus, in line with the aim of JailbreakBench to be reproducible, we choose
> Llama-3-70B as the classifier in our benchmark as it is an open-weight model and
> comparable to GPT-4 as a judge. Moreover, it has a relatively low FPR, which, although
> it may systematically reduce the success rate across attack algorithms, is important for
> remaining conservative and avoiding the misclassification of benign behaviors as jailbroken.

## §4 — Headline ASR results across attacks and models {#sec-4-asr}

> Motivated by our evaluation in Section 3.5, we track the attack success rate (ASR) according
> to Llama-3-70B as a jailbreak judge.

Key ASR results from Table 2 (attacks on undefended models, greedy/deterministic decoding,
150 tokens per response):

| Attack | Vicuna-13B | Llama-2-7B | GPT-3.5 | GPT-4 |
|---|---|---|---|---|
| PAIR | 69% | 0% | 71% | 34% |
| GCG | 80% | 3% | 47% | 4% |
| JB-Chat (AIM) | 90% | 0% | 0% | 0% |
| Prompt + RS | 89% | 90% | 93% | 78% |

> Prompt with RS is on average the most effective attack, achieving 90% ASR on Llama-2
> and 78% GPT-4. ... GCG exhibits a slightly lower jailbreak percentage than previously
> reported values: we believe this is primarily due to (1) the selection of more challenging
> behaviors in JBB-Behaviors and (2) a more conservative jailbreak classifier. Overall, these
> results show that even recent and closed-source undefended models are highly vulnerable
> to jailbreaks.

The prior-benchmark discrepancy for GCG is a key calibration finding: the same attack
algorithm produces lower ASR under JailbreakBench's standardized judge than under the
rule-based string-matching classifiers used in earlier work.

[Verified from PDF on 2026-05-12] Added §3.1 (JBB-Behaviors design), §3.5 (judge selection + Table 1 data), §4 (ASR results + Table 2 data). Abstract verified verbatim from PDF. Removed [PDF-VERIFY] flags — all claims now grounded in local PDF.
