---
paper_key: snell2024
title: Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters
authors: Snell, Lee, Xu, Kumar (UC Berkeley / Google DeepMind)
year: 2024
venue: arXiv (Aug 2024)
arxiv: 2408.03314
local_pdf: theory/sources/papers/snell2024_test_time_compute.pdf
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (Aug 2024). The headline result is the FLOPs-matched comparison showing a smaller model + test-time compute can outperform a 14× larger model on easy/medium reasoning. Anchors point to section + equation numbers.
---

# Excerpts — Snell et al. 2024, "Scaling LLM Test-Time Compute Optimally..."

## Abstract — the question and the headline result {#abstract}

> Enabling LLMs to improve their outputs by using more test-time
> computation is a critical step towards building generally
> self-improving agents that can operate on open-ended natural language.
> In this paper, we study the scaling of inference-time computation in
> LLMs, with a focus on answering the question: if an LLM is allowed to
> use a fixed but non-trivial amount of inference-time compute, how much
> can it improve its performance on a challenging prompt? Answering this
> question has implications not only on the achievable performance of
> LLMs, but also on the future of LLM pretraining and how one should
> tradeoff inference-time and pre-training compute. … In this work, we
> analyze two primary mechanisms to scale test-time computation: (1)
> searching against dense, process-based verifier reward models; and (2)
> updating the model's distribution over a response adaptively, given the
> prompt at test time. We find that in both cases, the effectiveness of
> different approaches to scaling test-time compute critically varies
> depending on the difficulty of the prompt. This observation motivates
> applying a "compute-optimal" scaling strategy, which acts to most
> effectively allocate test-time compute adaptively per prompt. Using
> this compute-optimal strategy, we can improve the efficiency of
> test-time compute scaling by more than 4x compared to a best-of-N
> baseline. Additionally, in a FLOPs-matched evaluation, we find that on
> problems where a smaller base model attains somewhat non-trivial
> success rates, test-time compute can be used to outperform a 14x larger
> model.

## §1 Introduction — the unified framing {#sec-1-headline}

> [The compute-optimal strategy] illustrates **the need to deploy an
> adaptive "compute-optimal" strategy for scaling test-time compute**,
> wherein the specific approach for utilizing test-time compute is
> selected depending on the prompt, so as to make the best use of
> additional computation. We also show that a notion of *question
> difficulty* (Section 4) from the perspective of the base LLM can be
> used to predict the efficacy of test-time computation, enabling us to
> practically instantiate this 'compute-optimal' strategy given a prompt.
> By appropriately allocating test-time compute in this way, we are able
> to greatly improve test-time compute scaling, surpassing the
> performance of a best-of-N baseline while only using about **4× less
> computation** with both revisions and search (Sections 5 and 6).

## §1 The FLOPs-matched comparison framing {#sec-1-flops-matched}

> Using our improved test-time compute scaling strategy, we then aim to
> understand to what extent test-time computation can effectively
> substitute for additional pretraining. We conduct a FLOPs-matched
> comparison between a smaller model with additional test-time compute
> and pretraining a **14× larger model**. We find that on easy and
> intermediate questions, and even hard questions (depending on the
> specific conditions on the pretraining and inference workload),
> additional test-time compute is often preferable to scaling
> pretraining. … *in some settings it is be more effective to pretrain
> smaller models with less compute, and then apply test-time compute to
> improve model outputs*. That said, with the most challenging
> questions, we observe very little benefits from scaling up test-time
> compute. Instead, we find that on these questions, it is more
> effective to make progress by applying additional pretraining compute,
> demonstrating that current approaches to scaling test-time compute may
> not be 1-to-1 exchangeable with scaling pretraining.

## §2 Unified perspective: proposer + verifier {#sec-2-proposer-verifier}

> First, we view the use of additional test-time compute through the
> lens of modifying the model's predicted distribution *adaptively* at
> test-time, conditioned on a given prompt. Ideally, test-time compute
> should modify the distribution so as to generate better outputs than
> naïvely sampling from the LLM itself would. In general, there are two
> knobs to induce modifications to an LLM's distribution: **(1) at the
> input level**: by augmenting the given prompt with an additional set of
> tokens that the LLM conditions on to obtain the modified distribution,
> or **(2) at the output level**: by sampling multiple candidates from
> the standard LM and performing surgery on these candidates. In other
> words, we could either modify the **proposal distribution** induced by
> the LLM itself such that it is an improvement over naïvely conditioning
> on the prompt or we could use some post-hoc **verifiers or scorers** to
> perform output modifications.

> **Modifying the proposal distribution.** … techniques such as
> self-critique enable the model itself to improve its own proposal
> distribution at test time by instructing it to critique and revise its
> own outputs in an iterative fashion. Since prompting off-the-shelf
> models is not effective at enabling effective revisions at test time,
> we specifically finetune models to iteratively revise their answers in
> complex reasoning-based settings.

> **Optimizing the verifier.** … the most canonical way to use such a
> verifier is by applying best-of-N sampling, wherein we sample N
> complete solutions and then select the best one according to a
> verifier. However, this approach can be further improved by training
> a process-based verifier, or a process reward model (PRM), which
> produces a prediction of the correctness of each intermediate step in
> a solution, rather than just the final answer. We can then utilize
> these per-step predictions to perform tree search over the space of
> solutions, enabling a potentially more efficient and effective way to
> search against a verifier, compared to naïve best-of-N.

## §3.1 Test-Time Compute-Optimal Scaling Strategy — formal definition {#sec-3-1}

> In general, we would therefore like to select the *optimal* allocation
> of our test-time compute budget for a given problem. To this end, for
> any given approach of utilizing test-time compute (e.g., revisions and
> search against a verifier in this paper, various other methods
> elsewhere), we define the **"test-time compute-optimal scaling
> strategy"** as the strategy that chooses hyperparameters corresponding
> to a given test-time strategy for maximal performance benefits on a
> given prompt at test time. Formally, define $\mathrm{Target}(\theta, N,
> q)$ as the distribution over natural language output tokens induced by
> the model for a given prompt $q$, using test-time compute
> hyper-parameters $\theta$, and a compute budget of $N$. We would like
> to select the hyper-parameters $\theta$ which maximize the accuracy of
> the target distribution for a given problem. We express this formally
> as:
>
> $$\theta^\ast_{q, a^\ast(q)}(N) = \arg\max_\theta \Big( \mathbb{E}_{y \sim \mathrm{Target}(\theta, N, q)}\big[ \mathbb{1}_{y = y^\ast(q)} \big] \Big), \tag{1}$$
>
> where $y^\ast(q)$ denotes the ground-truth correct response for $q$,
> and $\theta^\ast_{q, y^\ast(q)}(N)$ represents the test-time
> compute-optimal scaling strategy for the problem $q$ with compute
> budget $N$.

## §3.2 Question difficulty as a sufficient statistic {#sec-3-2-difficulty}

> In order to effectively analyze the test-time scaling properties of
> the different mechanisms discussed in Section 2 (e.g. the proposal
> distribution and the verifier), we will prescribe an approximation to
> this optimal strategy $\theta^\ast_{q, y^\ast(q)}(N)$ as a function of
> a statistic of a given prompt. This statistic estimates a notion of
> *difficulty* for a given prompt. The compute-optimal strategy is
> defined as a function of the difficulty of this prompt. Despite being
> only an approximate solution to the problem shown in Equation 1, we
> find that it can still yield substantial improvements in performance
> over a baseline strategy of allocating this inference-time compute in
> an ad-hoc or uniformly-sampled manner.

> Our estimate of the question difficulty assigns a given question to
> one of **five difficulty levels**. … **Defining difficulty of a
> problem.** Following the approach of Lightman et al., we define
> question difficulty as a function of a given base LLM. Specifically,
> we bin the model's pass@1 rate – estimated from 2048 samples – on each
> question in the test set into five quantiles, each corresponding to
> increasing difficulty levels. We found this notion of model-specific
> difficulty bins to be more predictive of the efficacy of using
> test-time compute in contrast to the hand-labeled difficulty bins in
> the MATH dataset.

> [W]e approximate the problem's difficulty via a **model-predicted
> notion of difficulty**, which performs the same binning procedure over
> the the averaged final answer score from a learned verifier (and not
> groundtruth answer correctness checks) on the same set of 2048 samples
> per problem. We refer to this setting as **model-predicted difficulty**
> and the setting which relies on the ground-truth correctness as
> **oracle difficulty**.

## §4 Experimental setup {#sec-4-setup}

> **Datasets.** … we focus on the MATH benchmark, which consists of
> high-school competition level math problems with a range of difficulty
> levels. For all experiments, we use the dataset split consisting of
> 12k train and 500 test questions, used in Lightman et al.

> **Models.** We conduct our analysis using the **PaLM 2-S\* (Codey)**
> base model. We believe this model is representative of the
> capabilities of many contemporary LLMs, and therefore think that our
> findings likely transfer to similar models.

## §5.2 Search methods against a PRM {#sec-5-2-search}

> We optimize the PRM at test time via search methods. We study three
> search approaches that sample outputs from a few-shot prompted base
> LLM. … **Best-of-N weighted.** We sample N answers independently from
> the base LLM and then select the best answer according to the PRM's
> final answer judgement. **Beam search.** Beam search optimizes the PRM
> by searching over its per-step predictions. Our implementation is
> similar to BFS-V. Concretely, we consider a fixed number of beams $N$
> and a beam width $M$. We then run the following steps: 1. sample $N$
> initial predictions for the first step in the solution; 2. score the
> generated steps according to the PRM's predicted step-wise
> reward-to-go estimate; 3. filter for only the top $\frac{N}{M}$
> highest scoring steps; 4. now from each candidate, sample $M$
> proposals from the next step, resulting in a total of $N/M \times M$
> candidate prefixes again. Then repeat steps 2-4 again.

A third search approach (lookahead search) is described after this in
§5.2.

## Why this matters — the new scaling axis {#why-matters}

Snell et al. is the **theoretical formalization** of inference-time
compute as a scaling axis distinct from training compute. The two key
empirical claims that drove the field:

1. **Compute-optimal allocation matters.** Adaptively choosing among
   {best-of-N, beam search, revision} based on a difficulty estimate
   beats any single fixed strategy by ~4×.

2. **Test-time compute can substitute for pretraining.** A smaller base
   model with sufficient test-time budget can match a 14× larger base
   model on easy/medium MATH questions. This is **not** universal — for
   the hardest questions, more pretraining still wins.

This paper landed Aug 2024; DeepSeek-R1 (Jan 2025) and the o1/o3 family
made the scaling axis empirically dominant in production. The "thinking
budget" parameter in Gemini 2.5 (`thinkingBudget` API) is the
productionized form of the same idea.
