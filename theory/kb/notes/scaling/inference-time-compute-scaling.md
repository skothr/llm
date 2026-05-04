---
topic: scaling/inference-time-compute-scaling
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - snell2024
  - deepseek-r1
  - s1-2025
secondary_sources:
  - wei2022-cot
  - rstar-math2025
  - thinkprm2025
  - gemini2-5      # productionized "thinkingBudget" API
related_topics:
  - scaling/chinchilla
  - scaling/scaling-frontier
  - reasoning/test-time-compute
  - reasoning/chain-of-thought
  - reasoning/process-supervision
  - reasoning/inference-time-search
  - post-training/rlvr-and-grpo
---

# Inference-time compute scaling

A second scaling axis, **distinct from training compute**, that was
formalized by Snell et al. (2024) and proven dominant in production by
DeepSeek-R1 (2025) and the o1/o3/Gemini-Thinking family. The headline:
holding a base model fixed, the loss/accuracy of its outputs improves
predictably with the amount of *test-time* compute spent searching,
revising, or sampling — and on a wide class of problems, this is more
cost-effective than training a larger model.

This note treats inference-time compute scaling as a quantitative
*scaling law* analog to Kaplan/Chinchilla. Mechanism details (CoT,
PRMs, search algorithms, RLVR training) are linked from §3 to their
own notes.

## 1. Formal definition

### 1.1 The compute-optimal test-time strategy

Following Snell et al.
`[snell2024 §3.1 Eq.(1); kb/excerpts/snell2024#sec-3-1]`,
let $\mathrm{Target}(\theta, N, q)$ be the distribution over output
tokens induced by a fixed base LLM on prompt $q$, with test-time
hyperparameters $\theta$ and a compute budget of $N$ inference FLOPs
(or, equivalently, generated tokens). Let $y^\ast(q)$ be the
ground-truth correct response. The compute-optimal strategy is:

$$\theta^\ast_{q, y^\ast(q)}(N) \;=\; \arg\max_\theta\; \mathbb{E}_{y \sim \mathrm{Target}(\theta, N, q)}\!\left[\mathbb{1}_{y = y^\ast(q)}\right]. \tag{1}$$

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $q$ | the prompt / question |
| $y^\ast(q)$ | ground-truth answer |
| $N$ | test-time compute budget (FLOPs, or token count) |
| $\theta$ | hyperparameters of a chosen test-time strategy (best-of-N count, beam width, revision depth, etc.) |
| $\mathrm{Target}(\theta, N, q)$ | the modified output distribution induced by spending $N$ FLOPs with strategy $\theta$ |
| $\mathbb{1}$ | indicator (1 if equal, 0 else) |

The key technical move: the strategy $\theta^\ast$ is a function of the
**prompt**, not the model. Different difficulty levels of $q$ are best
served by different $\theta$.

### 1.2 The proposer/verifier decomposition

Snell et al. unify all known test-time-compute strategies under two
levers `[snell2024 §2; kb/excerpts/snell2024#sec-2-proposer-verifier]`:

1. **Modify the proposal distribution.** Get the LLM to produce a
   *better* distribution at test time by feeding it more tokens (CoT,
   self-critique, revision-finetuned models that iteratively edit).
2. **Apply a verifier on outputs.** Sample $N$ candidates from the
   base distribution and select among them using a learned scorer:
   best-of-N (full-sequence verifier), beam search / lookahead against
   a process reward model (PRM, per-step verifier), tree search.

[ANALOGY] This is **MCMC sampling**: you have a hard target
distribution (correct answers); you can either build a better proposal
(modify the LM) or apply a Metropolis-style accept/reject (the
verifier). Snell make this analogy explicit. The canonical symbolic
form is Eq. (1): the goal is to maximize
$\mathbb{E}_{y \sim \mathrm{Target}}\!\left[\mathbb{1}_{y=y^\ast}\right]$,
and both proposer and verifier just shape $\mathrm{Target}$ in
different ways.

### 1.3 Question-difficulty-conditional strategy

The compute-optimal $\theta^\ast$ depends on prompt difficulty
`[snell2024 §3.2; kb/excerpts/snell2024#sec-3-2-difficulty]`. Snell
operationalize *difficulty* by binning the base LLM's pass@1 rate
(estimated from 2048 samples) on each test-set prompt into 5 quantiles.
Then for each difficulty bin, they pick the strategy ($\theta$) that
maximizes accuracy on a validation fold. Two variants:

- **Oracle difficulty** — use ground-truth correctness to bin (only
  feasible during method development).
- **Model-predicted difficulty** — use the verifier's averaged final-
  answer score as a proxy for pass@1 (deployable, costs the same as
  one strategy run).

Empirically: easy prompts benefit most from **revision** (modify
proposal); hard prompts benefit most from **search** (best-of-N or
beam against PRM); the very hardest fail to benefit at all from current
test-time-compute strategies and require *more pretraining*
`[snell2024 §1; kb/excerpts/snell2024#sec-1-flops-matched]`.

## 2. Mechanism — the empirical scaling curves

### 2.1 The two main mechanisms

**Search against a PRM.** A process reward model assigns a per-step
correctness score; a search procedure (best-of-N-weighted, beam, or
lookahead) selects among candidates
`[snell2024 §5.2; kb/excerpts/snell2024#sec-5-2-search]`. Compute
scales with number of candidates × depth × verifier cost.

**Revision (modify proposal).** A model is finetuned to iteratively
revise its previous answer; at test time, sample $N$ revisions in a
chain, optionally selecting via best-of-N at the end. Compute scales
linearly with chain depth.

A third class — **majority voting / self-consistency** (Wang et al.
2022) — is the cheap baseline: sample $N$ independent CoTs at $T > 0$
and majority-vote the answers. No verifier needed. Compute scales with
$N$. Often outperformed by best-of-N + verifier but occasionally
competitive on math.

### 2.2 The compute-equivalence claim

Snell's Section 6 FLOPs-matched experiment compares:

| Configuration | FLOPs |
|---|---|
| Smaller base model + test-time compute (revisions, search) | $C_{\text{small}}^{\text{train}} + C^{\text{test}}$ |
| 14× larger model, same total inference budget | $C_{\text{large}}^{\text{train}} + C_{\text{large}}^{\text{1-pass-inference}}$ |

with $C_{\text{small}}^{\text{train}} + C^{\text{test}} \approx
C_{\text{large}}^{\text{train}} + C_{\text{large}}^{\text{1-pass-inference}}$.

The headline finding `[snell2024 §1; kb/excerpts/snell2024#sec-1-flops-matched]`:

> On easy and intermediate questions, and even hard questions
> (depending on the specific conditions on the pretraining and
> inference workload), additional test-time compute is often
> preferable to scaling pretraining. … in some settings it is be more
> effective to pretrain smaller models with less compute, and then
> apply test-time compute to improve model outputs. That said, with
> the most challenging questions, we observe very little benefits from
> scaling up test-time compute. Instead, we find that on these
> questions, it is more effective to make progress by applying
> additional pretraining compute, demonstrating that current
> approaches to scaling test-time compute may not be 1-to-1
> exchangeable with scaling pretraining.

So the trade-off is **conditional on difficulty regime**. For the easy
and medium tail (which is most prompts in deployment), test-time
compute substitutes for ~14× more pretraining FLOPs. For the hardest
tail, the substitute breaks down.

### 2.3 The 4× compute-optimal vs. naïve gain

Within test-time compute itself, picking the right strategy *per
prompt* (compute-optimal in the §1.1 sense) gives ~**4× efficiency
over fixed best-of-N**, holding accuracy constant
`[snell2024 abstract; kb/excerpts/snell2024#abstract]`. The biggest
single source of gain is using revision on easy and search on hard.

## 3. Variants and lineage

### 3.1 The chain-of-thought lineage

| Year | Method | Mechanism | Compute scaling |
|---|---|---|---|
| 2022 | CoT prompting `[wei2022-cot]` | Few-shot demos with reasoning steps | Linear in chain length |
| 2022 | Zero-shot CoT (Kojima) | "Let's think step by step" trigger | Linear in chain length |
| 2022 | Self-consistency (Wang) | $N$-sample CoT, majority vote | Linear in $N$ |
| 2023 | Tree of Thoughts (Yao) | Branching CoT, BFS/DFS over states | Polynomial in branch × depth |
| 2024 | OpenAI o1 (Sept 2024) | RL-trained "thinking" before answering | Linear in thinking-token budget |
| 2024 | Snell et al. `[snell2024]` | Compute-optimal allocation among above | Strategy-conditional |
| 2025 | DeepSeek-R1 `[deepseek-r1]` | GRPO + verifiable rewards; emergent CoT | Linear in thinking-token budget |
| 2025 | s1 `[s1-2025]` | "Budget forcing" — append `Wait` to extend thinking | Linear, with explicit budget control |
| 2025 | rStar-Math `[rstar-math2025]` | MCTS + PRM, self-evolving | Polynomial in MCTS budget |

### 3.2 Reasoning-trained vs. inference-only

Two operationally distinct regimes:

- **Inference-only test-time compute.** Apply one of the §3.1 methods
  to a base or instruction-tuned model that was *not* trained for
  long-form reasoning. Returns are limited; o1-style models confirm
  that without specific RL training, models plateau quickly.
- **Reasoning-trained models.** Models like o1, DeepSeek-R1, Qwen-3
  thinking-mode, Gemini 2.5 thinking-mode are RL-trained (typically
  with verifiable rewards, GRPO-family) on long-form CoT, then deploy
  test-time compute scaling natively. These models have qualitatively
  different scaling curves — accuracy continues to climb with thinking
  budget far longer than untrained models do.

The DeepSeek-R1 paper `[deepseek-r1; kb/index/papers.json#deepseek-r1]`
trained Qwen-32B-base with GRPO + verifiable binary rewards (math,
code) and reached AIME 2024 71% (from base 15.6%) — comparable to
o1-preview at ~70% lower API cost. This was the empirical proof point
that inference-time compute is now a deployable scaling axis, not just
an academic finding.

[CONTRADICTION] On whether RLVR *creates* new reasoning ability: Yue
et al. 2025 (`rlvr-limits-2025`) argue RLVR shifts the pass@1
distribution within a base model's existing competency — it improves
*sampling efficiency* on problems the base model can already
sometimes solve, but does **not** expand the capability frontier. The
"reasoning ability ceiling is set at pretraining" view contradicts the
DeepSeek-R1 narrative of RLVR-elicited reasoning. As of early 2026, the
field has not converged.

### 3.3 Productionized — the "thinking budget" API

Gemini 2.5 `[gemini2-5; kb/index/papers.json#gemini2-5]` exposes a
configurable `thinkingBudget` parameter that maps to a hard cap on
thinking-token spend per request. Anthropic's extended-thinking mode
(2025) and OpenAI's reasoning-effort levels are similar productized
forms. The API surface is now:

- Cheap mode: small thinking budget, low latency, low cost — most
  consumer queries.
- Expensive mode: large thinking budget, multi-second latency,
  proportionally higher cost — hard reasoning, agentic loops.

This is the productionization of Snell's compute-optimal idea: route
queries to budgets based on difficulty estimates.

## 4. Intuitions and analogies

[INTUITION] **Not all problems benefit equally.** A prompt the model
already nearly solves is best served by *refining* (revision); a
prompt the model occasionally solves is best served by *sampling
many* (best-of-N or beam against PRM); a prompt the model essentially
*never* solves is best served by *training a better model* — no amount
of inference compute will rescue it. The scaling curves are flat at
both extremes (free at the easy end, futile at the hard end) and
steep in the middle. This is the empirical content of Snell's "five
difficulty levels" framework.

[ANALOGY] **Chinchilla : pre-training :: Snell : inference.** Both are
"how to allocate a compute budget for best loss/accuracy" results.
Chinchilla allocates between $N$ (model size) and $D$ (training
tokens) at fixed $C_{\text{train}}$. Snell allocates among $\theta$
(strategy choice) at fixed $C_{\text{test}}$. The structural parallel:
both are constrained-optimization problems with empirically-fitted
exchange rates between dimensions. The canonical math is Eq. (1)
above — strict argmax over $\theta$ — and the empirical content is
the FLOPs-matched comparison curves.

[INTUITION] **Why test-time compute can substitute for parameters.**
A larger model has more "implicit" search capacity per single forward
pass — it can explore more circuits internally. A smaller model with
more explicit search (best-of-N + verifier) trades implicit for
explicit: it samples more candidates, each cheaper. If the verifier
is good enough to discriminate, the explicit search wins on FLOPs.
The 14× substitution ratio Snell finds is a *measurement* of how much
implicit search you can replace with explicit — for the kinds of
problems where verifiers work well. For problems where no good
verifier exists (open-ended generation, novel research-level
reasoning), the substitution breaks down.

## 5. Frontier and open questions (as of 2026-05)

- **What's the functional form of the test-time scaling law?** Snell
  show empirical curves but do not commit to a Kaplan/Chinchilla-style
  closed-form $L(\theta, N)$. s1 (2025) shows roughly logarithmic
  returns to thinking-token count for some tasks. Whether power-law,
  log, or something else is the right functional form is open.
- **The "compute paradox" / inverse scaling.** On some tasks (open-
  ended dialogue, certain creative tasks), more thinking *hurts*
  accuracy or quality. This contradicts naive expectations from CoT.
  An active 2025–2026 research thread.
- **Can verifiers scale further than the model?** PRM and ORM
  performance is bounded by the verifier's training data quality. If
  a verifier is roughly as capable as the proposer, search compounds
  errors rather than discriminating them. Whether you can train a
  verifier that meaningfully exceeds its base model is open.
- **RLVR base-model dependence.** [CONTRADICTION] DeepSeek-R1 vs
  Yue 2025: does RL with verifiable rewards expand the capability
  frontier or only refine the pass@1 distribution within a base
  model's existing competency? See
  `kb/notes/post-training/rlvr-and-grpo.md`.
- **Compute distribution over time.** Industry projections (Epoch AI,
  2025) put inference at 75% of total AI compute by 2030, displacing
  training as the dominant spend. This is a structural economic shift
  that follows directly from inference-time compute scaling becoming
  deployable.
- **Combination with training-time scaling.** Llama 4's "iRoPE" + long
  context + reasoning-trained variants are early data on whether
  training and inference scaling axes compound multiplicatively or
  trade off. As of 2026 there is no clean theoretical model of the
  joint $(C_{\text{train}}, C_{\text{test}})$ frontier.

## 6. See also

- `kb/notes/scaling/chinchilla.md` — the *training-time* analog and
  conceptual parent.
- `kb/notes/scaling/scaling-frontier.md` — current frontier-lab
  practice; how the two axes are combined.
- `kb/notes/reasoning/test-time-compute.md` — operational details of
  CoT, self-consistency, ToT, and the budget-forcing recipe.
- `kb/notes/reasoning/process-supervision.md` — PRM training (the
  verifier side of Snell §2).
- `kb/notes/reasoning/inference-time-search.md` — beam search, MCTS,
  the search algorithms used against PRMs.
- `kb/notes/post-training/rlvr-and-grpo.md` — the training pipeline
  that creates reasoning-trained models (DeepSeek-R1, Qwen-3 thinking,
  o1-style).
- `kb/notes/reasoning/chain-of-thought.md` — the foundational prompting
  technique that all of test-time-compute scaling builds on.
