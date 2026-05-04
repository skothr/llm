---
topic: reasoning/test-time-compute
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - snell2024
  - s1-2025
  - deepseek-r1
  - wang2022-self-consistency
secondary_sources:
  - lightman2023-prm800k
  - gemini2-5
  - rstar-math2025
related_topics:
  - reasoning/chain-of-thought
  - reasoning/process-supervision
  - reasoning/reasoning-training
  - reasoning/inference-time-search
  - scaling/inference-time-compute-scaling
---

# Test-time compute

**Test-time compute (TTC)** is the additional inference-side compute
spent on a single problem instance to improve answer quality, beyond a
single greedy forward pass. It is the second of two scaling axes
recognised by the post-2024 literature — co-equal to training compute
`[snell2024 §1]`. This note covers the formal framing, the three
mechanism families (sampling, search, sequential), the s1 minimum recipe,
and the DeepSeek-R1 demonstration that TTC scaling can be elicited
through training rather than purely test-time scaffolding.

> **PDF-fetch caveat (2026-05-04):** primary PDFs for `snell2024`,
> `s1-2025`, and `deepseek-r1` were not downloaded in this Phase 2
> pass — see the area report. Section anchors below refer to the
> canonical arXiv-version structure of each paper. Verbatim excerpt
> files are skeleton-only and must be populated before any LaTeX
> propagation.

## 1. Formal definition

Let $x$ be a problem instance, $y^*$ its ground-truth answer, and
$\mathcal{V}: y \to \{0, 1\}$ a verifier. Let $C(\theta)$ denote the
total inference compute (FLOPs, tokens, or wall-clock) used by a model
$\theta$ on $x$. **Test-time compute scaling** asks: as $C \to \infty$,
how does the expected verifier score
$\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x; C)} [\mathcal{V}(y)]$
behave?

A core empirical claim of Snell et al. 2024 is that, for a fixed model
$\theta$, this expectation is a **smooth, monotonic, concave** function
of $\log C$ on math-reasoning tasks, comparable in shape to training-
compute scaling laws `[snell2024 §3, fig.1]`. They further claim that on
medium-difficulty problems, TTC scaling on a smaller model can dominate
parameter scaling on a larger model at matched FLOPs `[snell2024 §4]`.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $C$ | inference compute budget (in tokens, samples, or FLOPs) |
| $N$ | number of independent samples (sampling-based TTC) |
| $B$ | beam width / branching factor (search-based TTC) |
| $L$ | trace length (sequential TTC: extended thinking) |
| $\pi_\theta$ | the inference distribution of model $\theta$ — depends on the TTC strategy |
| $\mathcal{V}$ | verifier (rule-based, learned, or self-consistency-based) |
| $\hat y$ | selected answer after TTC |

## 2. Mechanism — three TTC families

The 2025 literature converged on three mechanically-distinct families,
which can be combined.

### 2.1 Sampling-based (parallel)

Sample $N$ independent traces and aggregate:

$$\hat y = \mathrm{aggregate}\bigl(y^{(1)}, \ldots, y^{(N)}\bigr) \tag{1}$$

Three aggregation rules dominate:

1. **Self-consistency / majority vote** (Wang 2022): unweighted
   plurality over extracted answers `[wang2022-self-consistency §3]`.
   Cheap, requires no verifier, requires low-cardinality answer space.
2. **Best-of-$N$ with outcome-reward verifier** (ORM): pick
   $\hat y = y^{(n^*)}$ where $n^* = \arg\max_n R_\phi(x, z^{(n)},
   y^{(n)})$. The ORM $R_\phi$ is trained on (prompt, complete
   solution, outcome) triples `[lightman2023-prm800k §2]`.
3. **Best-of-$N$ with process-reward verifier** (PRM): score each
   *step* of the trace and aggregate (min, product, last).
   Substantially outperforms ORM at fixed $N$ on MATH, mainly because
   PRM tolerates a single bad step among many
   `[lightman2023-prm800k §4, fig.4]`. See
   `kb/notes/reasoning/process-supervision.md`.

Compute cost: $C \propto N$, embarrassingly parallel across samples.

### 2.2 Search-based (tree)

Treat decoding as expanding a tree of partial reasoning prefixes; use a
verifier to direct expansion:

- **Best-of-$N$ over partial steps / beam search.** Generate $B$ next-
  step candidates per node, score each, retain top-$k$, expand.
- **Lookahead / step-level beam.** Generate full traces from each
  candidate, score the resulting outcome, retain best `[snell2024 §4
  procedure]`.
- **MCTS over reasoning steps.** Selection + expansion + simulation +
  backup, with a step-level value function (often a PRM)
  `[rstar-math2025 §3]`. See `kb/notes/reasoning/inference-time-
  search.md` for the full taxonomy.

Compute cost: $C \propto B \times L$ (or $\propto$ MCTS rollout count).

### 2.3 Sequential-compute (extended thinking)

Hold $N = 1$ but let the model emit a *much longer* single trace
$z$ before committing to $y$:

$$z \sim p_\theta(\cdot \mid x; \text{length budget}\ L), \quad y \sim p_\theta(\cdot \mid x, z) \tag{2}$$

The compute is spent re-reading prior trace tokens via the KV cache and
emitting refinements, branches considered and rejected, error-corrections,
and verification within $z$ itself. This is the o1 / R1 / QwQ /
Gemini-2.5-thinking paradigm `[deepseek-r1 §2.3]`.

Two ways to control $L$:

1. **API-level thinking budgets.** Gemini 2.5 exposes a
   `thinkingBudget` integer; o-series uses `effort` levels; Anthropic
   exposes a `thinking` block with token caps `[gemini2-5 §3]`.
2. **Budget forcing (s1 2025).** A lightweight test-time technique:
   when the model emits its end-of-thinking token before the budget
   is exhausted, replace it with a continuation cue ("Wait", or just
   the suppression of `</think>`) to force more decoding
   `[s1-2025 §3.2]`. Conversely, hard-truncate to enforce shorter
   budgets.

Compute cost: $C \propto L \times $ (per-token decode cost). Per-token
cost is itself $O(L)$ in the prefix-attention term, so total cost grows
super-linearly in budget.

## 3. The s1 minimum recipe

s1 is the load-bearing 2025 demonstration that "thinking-model"
behaviour is reachable with very little post-training data
`[s1-2025 §3, §4]`:

1. **Curated SFT data:** 1,000 high-quality (problem, long-CoT trace,
   answer) examples — single-digit thousands of tokens each — selected
   for difficulty and reasoning-trace quality.
2. **SFT only.** No RL, no reward model.
3. **Budget forcing at test time.** Force the model to think longer by
   suppressing the end-of-thinking token; force it to stop earlier by
   inserting an end-of-thinking token.

The resulting s1-32B reproduces o1-preview-style test-time scaling
curves on AIME / MATH / GPQA at a small fraction of o1's training cost
`[s1-2025 §4 table 1]`.

[CONTRADICTION] s1's claim that 1K examples + budget forcing **suffices**
sits in tension with DeepSeek-R1's claim that **RL on top of SFT** is
required to achieve the strongest performance `[deepseek-r1 §2.2.4
ablation]`. The clearest reconciliation in the 2025 literature: SFT on
curated traces gets the model into the right regime cheaply; RL refines
within that regime but cannot create the regime from scratch on a base
that lacks the latent capability `[rlvr-limits-2025 §4]`.

## 4. The DeepSeek-R1 demonstration

R1 establishes that **TTC scaling can be elicited by training**, not
just by test-time scaffolding `[deepseek-r1 §2.2]`:

- R1-Zero: GRPO with rule-based verifier rewards (correct numeric
  answer for math, passing tests for code), starting from a base
  model. The trained model spontaneously develops longer traces, with
  training-time average response length growing from $\sim 200$ to
  $\sim 10{,}000$ tokens, accompanied by AIME 2024 score growing from
  $15.6\% \to 71\%$ `[deepseek-r1 §2.2.2 fig.3]`.
- R1: cold-start SFT on a small set of long-CoT exemplars before GRPO,
  fixing readability problems of R1-Zero. Same TTC-scaling property is
  preserved `[deepseek-r1 §2.3]`.
- Distillation: SFT on R1-generated traces transfers reasoning to
  smaller models (1.5B–32B) at a fraction of the cost. The R1-distill
  models retain o1-mini-class reasoning at 7B–32B `[deepseek-r1
  §3.2.2]`.

Operationally, R1's contribution to the TTC literature is: the long
trace is no longer "prompted in" or "test-time-scaffolded in" — it is
the model's natural output distribution. Inference-time scaling then
just modulates the trace length under that learned distribution.

## 5. Variants and lineage

| Approach | Year | Family | Compute axis | Quality regime |
|---|---|---|---|---|
| Greedy CoT | 2022 | sequential ($L$) | trace length, no scaling | baseline; emergent at scale `[wei2022-cot §3.3]` |
| Self-consistency | 2022 | sampling ($N$) | $N$ samples + majority | strong at low cost `[wang2022-self-consistency]` |
| Best-of-$N$ + ORM | 2023 | sampling ($N$) + verifier | $N$ samples scored by ORM | dominated by PRM at same $N$ `[lightman2023-prm800k]` |
| Best-of-$N$ + PRM | 2023 | sampling ($N$) + step verifier | $N$ samples, step-scored | strong on math `[lightman2023-prm800k §4]` |
| Snell 2024 compute-optimal mix | 2024 | adaptive sampling/search | shifts $N$ vs $L$ by difficulty | proves TTC ≥ params on medium-difficulty `[snell2024 §6]` |
| Tree search (rStar-Math) | 2025 | search + PRM | MCTS rollouts | 90% MATH at 7B `[rstar-math2025 §4]` |
| s1 (1K + budget force) | 2025 | sequential, light SFT | trace length $L$ via forcing | reproduces o1-preview curves `[s1-2025 §4]` |
| R1 (RL elicits TTC) | 2025 | sequential, trained | trace length $L$ from training | o1-class on math/code `[deepseek-r1 §3.1]` |
| Gemini-2.5 thinkingBudget | 2025 | API-level $L$ control | configurable per call | productized `[gemini2-5 §3]` |

### 5.1 The compute-optimal-TTC argument

Snell et al. 2024 frame TTC selection as an optimisation: at fixed
total inference compute $C$, choose the strategy
$s \in \{$sampling-$N$, beam-$B$, sequential-$L$, mixtures$\}$ that
maximises expected verifier score on a held-out distribution
`[snell2024 §3]`. Two empirical headlines:

1. The optimal mixture is **problem-difficulty-dependent**: easy
   problems are best served by short single traces; medium-difficulty
   problems by sampling-based TTC; hard problems by search + verifier
   `[snell2024 §5]`.
2. On medium-difficulty MATH problems, a 14× smaller model with TTC
   beats the larger model at matched inference FLOPs
   `[snell2024 §6.1]`. This is the canonical "TTC > scale" result.

[INTUITION] The "shape" of the TTC-vs-params trade-off is determined by
how much of the answer's information is recoverable by re-using the
existing weights with more compute, versus needing larger weights. For
search-decomposable problems (math proofs, code with unit tests),
verification signal is cheap and search wins. For problems that hinge
on knowledge breadth, more parameters win. The canonical formalisation
of this is open as of 2026.

## 6. Frontier and open questions (as of 2026-05)

- **Does TTC scaling have a ceiling on current models?** [CONTRADICTION]
  `[arXiv 2502.12215]` reports that for R1- and QwQ-class models,
  extending solution length past a problem-dependent point monotonically
  degrades accuracy — the model loops, hallucinates, or commits to a
  wrong answer. Snell 2024's smooth curves were measured below this
  ceiling. The ceiling is presumably a function of architecture (long-
  context attention quality), training (whether the model was trained
  to stop), and problem structure.
- **What is the right cost metric?** TTC papers vary between using
  FLOPs, output-tokens, or wall-clock as the $C$ axis. The exchange
  rates differ by deployment (KV-cache reuse changes per-token cost
  super-linearly in trace length). Cross-paper comparison is fragile.
- **Adaptive / controllable TTC.** The L1-controllable / L2-adaptive
  taxonomy `[arXiv 2507.02076]` distinguishes between user-set budgets
  and model-decided budgets. Production systems need both, but the
  joint optimisation is unsolved.
- **Verifier-free TTC.** Most search-based TTC requires a PRM or rule-
  based verifier. For domains without verifiable rewards (creative
  writing, open-ended QA), the equivalent of "majority vote with
  PRM" is unclear; current practice falls back on self-consistency or
  LLM-as-judge.
- **Scaling laws for TTC.** Snell 2024's curves are empirical, not
  derived from a Chinchilla-style closed-form. Whether there is a
  universal $C_{\text{train}}$ vs $C_{\text{infer}}$ optimum is open
  `[scaling/inference-time-compute-scaling]`.

## 7. Intuitions and analogies

[ANALOGY] TTC families as **CPU operations**: sampling is *parallel
threads*, search is *branch prediction*, sequential is *deeper pipeline*.
The analogy returns to canonical form via the cost expressions of §2:
parallel threads are linear in $N$, branching is multiplicative across
levels, deeper pipelines are linear in $L$ but interact with KV-cache
re-attention costs.

[INTUITION] R1's training-time grows the trace from 200 to 10 000
tokens, but this is **not** "the model thinking faster"; it is the
model's optimisation pressure under verifiable rewards finding that
longer traces achieve higher reward, plus enough supervision in
pretraining (math, code, scientific text) to make long-CoT continuations
sensible. Whether this is "real reasoning" or "sampled-from-the-right-
distribution" is the same question §4.1 of the CoT note attacks under
the faithfulness label.

[INTUITION] s1's 1K-example minimality argues that **the test-time
behaviour was latent in the base model**; SFT only needed to surface
it. R1's RL finding argues the same in stronger form: GRPO from a
base model produces the trace-lengthening behaviour without explicit
long-CoT exemplars `[deepseek-r1 §2.2.2]`. Together these support the
"latent capability" reading of pretraining
`[rlvr-limits-2025]`, not the "RL teaches new skills" reading.
[CONTRADICTION] This interpretive question is contested.

## 8. See also

- `kb/notes/reasoning/chain-of-thought.md` — substrate of all sequential
  TTC; faithfulness considerations.
- `kb/notes/reasoning/process-supervision.md` — PRM-based verifiers
  used in best-of-$N$ and search.
- `kb/notes/reasoning/inference-time-search.md` — MCTS, beam, BFS over
  reasoning steps.
- `kb/notes/reasoning/reasoning-training.md` — training side of R1 /
  o1 / QwQ; how TTC scaling is elicited.
- `kb/notes/scaling/inference-time-compute-scaling.md` — TTC as a
  scaling axis comparable to Chinchilla / Kaplan.
- `kb/notes/architecture/reasoning-architectures.md` — architectural
  features that make long-trace inference efficient (KV-cache reuse,
  long-context attention).
