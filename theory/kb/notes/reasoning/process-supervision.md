---
topic: reasoning/process-supervision
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - lightman2023-prm800k
  - thinkprm2025
secondary_sources:
  - rstar-math2025
  - cao2025-edu-prm
  - she2025-r-prm
  - zheng2025-prm-survey
related_topics:
  - reasoning/test-time-compute
  - reasoning/inference-time-search
  - reasoning/reasoning-training
  - reasoning/chain-of-thought
  - post-training/rlvr-and-grpo
---

# Process supervision

**Process supervision** is the practice of training and / or test-time
verification of LLM reasoning at the level of *individual reasoning
steps*, rather than only the final outcome. The canonical artefacts are
**Process Reward Models (PRMs)**: scoring functions that estimate the
correctness of each step in a chain-of-thought trace. PRMs were
introduced as a contrast to **Outcome Reward Models (ORMs)**, which
score only the final answer. This note covers the PRM training data,
the PRM800K dataset, the PRM vs ORM ablation, the 2025 generative-PRM
shift, and the use of PRMs both at test time (best-of-$N$, search) and
in training loops.

> **PDF-fetch caveat (2026-05-04):** primary PDFs for `lightman2023-
> prm800k`, `thinkprm2025`, and the 2025 PRM survey were not downloaded
> in this Phase 2 pass — see the area report. Section anchors below
> refer to the canonical arXiv-version structure. Verbatim excerpt
> files are skeleton-only.

## 1. Formal definition

Let a reasoning trace be decomposed into steps
$z = (s_1, s_2, \ldots, s_T)$, with $s_t$ a contiguous span of the
trace (a sentence, an equation line, etc.). Two scoring families:

### 1.1 Outcome Reward Model (ORM)

A function $R^O_\phi: (x, z, y) \to [0, 1]$ trained on examples where
the label is the binary correctness of the final answer
`[lightman2023-prm800k §2.1]`:

$$\mathcal{L}_\text{ORM} = - \mathbb{E}_{(x, z, y, c) \sim \mathcal{D}} \bigl[c \log R^O_\phi(x, z, y) + (1-c) \log(1 - R^O_\phi(x, z, y))\bigr] \tag{1}$$

with $c \in \{0, 1\}$ the outcome correctness label.

### 1.2 Process Reward Model (PRM)

A function $R^P_\phi: (x, s_1, \ldots, s_t) \to [0, 1]$ trained on per-
step labels `[lightman2023-prm800k §2.2]`:

$$\mathcal{L}_\text{PRM} = - \mathbb{E}_{(x, s_{1:t}, c_t) \sim \mathcal{D}} \bigl[c_t \log R^P_\phi(x, s_{1:t}) + (1-c_t) \log(1 - R^P_\phi(x, s_{1:t}))\bigr] \tag{2}$$

with $c_t \in \{0, 1\}$ the per-step correctness label, conditioned
only on the prefix $s_1, \ldots, s_t$.

### 1.3 Trace-level aggregation

To compare a PRM and an ORM at fixed cost, the per-step PRM scores must
be aggregated to a trace score. Three rules dominate
`[lightman2023-prm800k §3.1]`:

- **Min**: $R^P(z) = \min_t R^P_\phi(x, s_{1:t})$ — interpret a single
  bad step as fatal.
- **Product / mean of log**: $R^P(z) = \prod_t R^P_\phi(x, s_{1:t})$
  — interpret each step as conditionally independent.
- **Last-step**: $R^P(z) = R^P_\phi(x, s_{1:T})$ — score the whole
  trace at its final state.

OpenAI's original ablation found that **min** is the strongest
aggregator on PRM800K-grade verifiers `[lightman2023-prm800k §4.2]`.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $T$ | number of steps in trace $z$ |
| $s_t$ | the $t$-th step (a span of trace tokens) |
| $c_t$ | per-step correctness label (PRM) |
| $c$ | outcome correctness label (ORM) |
| $R^P_\phi$, $R^O_\phi$ | PRM, ORM (parameters $\phi$) |
| PRM800K | OpenAI's released dataset of $\approx 800$K human step-labels |

## 2. Mechanism — PRM800K and the canonical experiment

### 2.1 The PRM800K dataset

OpenAI collected $\approx 800{,}000$ human step-correctness labels on
GPT-4-generated solutions to MATH problems
`[lightman2023-prm800k §3]`:

- Each problem is sampled from the held-out training portion of MATH.
- A generator (a GPT-4 variant) produces several candidate solutions
  per problem.
- Human annotators label each step as positive / negative / neutral.
- The dataset is released openly.

PRM800K is the foundational reference dataset; subsequent open-source
PRMs (Math-Shepherd, ThinkPRM, R-PRM) all benchmark against it.

### 2.2 The headline ablation: PRM > ORM at matched $N$

Lightman et al. compare ORM-best-of-$N$ vs PRM-best-of-$N$ on MATH
test, fixing $N$ across both. The PRM achieves substantially higher
$\mathrm{best}@N$ at the same $N$ across the full $N \in [1, 1024]$
range `[lightman2023-prm800k §4 fig.4]`.

The proposed mechanism: PRMs catch local errors that the ORM only sees
indirectly through their effect on the final answer. When the
generator's per-step error rate is non-trivial and steps are not
independent, an ORM under-uses signal `[lightman2023-prm800k §5
discussion]`.

### 2.3 Test-time uses of a PRM

Three canonical use patterns:

1. **Best-of-$N$ reranking.** Generate $N$ traces, score each with the
   PRM (typically with `min` aggregation), select the highest. Section
   2.2's headline result.
2. **PRM-guided search.** Use the PRM as the value function in a tree
   search over partial traces. Pruning expansions of low-PRM partial
   prefixes; rStar-Math is the canonical instantiation
   `[rstar-math2025 §3]`. See `kb/notes/reasoning/inference-time-
   search.md`.
3. **PRM-weighted self-consistency.** Combine self-consistency
   majority vote with PRM scores: weight each trace's vote by its PRM
   score before majority. Improves over plain self-consistency on
   competition math `[lightman2023-prm800k §4.3]`.

### 2.4 Training-loop uses of a PRM

The 2025 literature increasingly uses PRMs as a **training signal** in
addition to / in place of outcome verifiers:

- **PRM-shaped GRPO.** Replace or augment the outcome reward $r_i$
  with a per-step or per-trace PRM score. Used by some R1-replication
  efforts `[arXiv 2505.00551]`. Tradeoff: PRM-shaping helps when the
  PRM is well-aligned, but a misaligned PRM can encourage steps that
  *look* correct without producing the right answer (process
  hacking).
- **MCTS data generation.** rStar-Math uses the PRM as the MCTS value
  function; the trees discovered guide both inference and policy SFT
  data generation `[rstar-math2025 §3]`.

## 3. Variants and lineage

### 3.1 Discriminative PRMs

The original PRM lineage `[lightman2023-prm800k]`. The PRM is a
classifier that, given prefix tokens, outputs a probability of step
correctness. Architectures: a small head on top of a frozen / fine-
tuned LLM backbone; cheaper than the policy model.

| PRM | Year | Training data | Notable property |
|---|---|---|---|
| OpenAI PRM | 2023 | PRM800K (human labels) | Foundational benchmark `[lightman2023-prm800k]` |
| Math-Shepherd | 2024 | Automatic labels via outcome supervision | Removes need for human step-labels at scale `[arXiv 2312.08935]` |
| OmegaPRM | 2024 | Large-scale automated PRM data | Scales `[Rewarding Progress, 2024]` |
| R-PRM | 2025 | Adds reasoning step before scoring | Generalises better `[she2025-r-prm, arXiv 2503.21295]` |
| EDU-PRM | 2025 | Entropy-driven step boundaries; 1.5% data | Beats Math-Shepherd at small budget `[cao2025-edu-prm, arXiv 2503.22233]` |

### 3.2 Generative PRMs (the 2025 shift)

The 2025 standout: ThinkPRM `[thinkprm2025 §3]`. Rather than a
classifier, the verifier is itself a small reasoning LLM that:

1. Reads the prefix $s_{1:t}$.
2. Emits its own short reasoning trace evaluating the prefix.
3. Outputs a verdict token.

ThinkPRM is trained on **1% of PRM800K** plus self-generated synthetic
verification traces, and outperforms full-data discriminative PRMs on
ProcessBench and AIME 2024 `[thinkprm2025 §4]`. The mechanism: the
verifier's own CoT lets it apply checks (algebraic verification, type
checking, dimensional analysis) that a feed-forward classifier cannot
internalise from labelled examples.

[INTUITION] Generative PRMs are to discriminative PRMs what reasoning-
trained LLMs are to direct-answer LLMs: a token-budget for
intermediate computation buys quality, at the cost of being slower
to evaluate. The cost asymmetry is real — best-of-$N$ with a
generative PRM costs $O(N \times L_\text{verify})$ tokens; with a
discriminative PRM, $O(N)$ classifier passes.

[CONTRADICTION] The 2025 PRM survey `[zheng2025-prm-survey, arXiv
2510.08049]` argues the discriminative-vs-generative split is
problem-dependent: generative PRMs win on competition math where each
step has a verifiable certificate; discriminative PRMs are competitive
on broader domains where step-level checks are not formal. ProcessBench
and AIME 2024 both fall in the "verifiable certificate" camp, plausibly
favouring generative PRMs.

### 3.3 Formally-verified training data

A separate 2025 thread `[arXiv 2505.15960]` uses formal proof checkers
(Lean, Coq) to generate provably-correct step-level training labels.
PRMs trained on such data inherit the verifier's correctness
guarantees on the labelled training set. Currently restricted to math
domains where formalisation is available.

### 3.4 ProcessBench

ProcessBench `[arXiv 2412.06559]` is the standardised benchmark for
PRM step-error detection: given a partial trace, identify the first
incorrect step (or note that all are correct). Used to compare PRMs
since late 2024.

## 4. Interaction with the broader reasoning stack

### 4.1 PRMs vs trained long-CoT models

The 2025 literature reports a tension: as base models get better at
long-CoT (post-R1), the marginal value of an external PRM at test
time decreases — the model's own internal verification approaches the
PRM's signal `[thinkprm2025 §5 discussion]`, a tension also surfaced by
the 2025 PRM survey `[zheng2025-prm-survey]`.

[CONTRADICTION] Whether PRMs remain useful for the strongest
reasoning-trained models is contested. Some 2026 measurements show
PRM-best-of-$N$ over R1-distill outputs gives gains; others find
diminishing returns. Likely depends on the base model's verifier-
fidelity gap.

### 4.2 PRMs as RL signal vs human preference signal

Standard RLHF uses a learned preference reward; RLVR uses a verifiable
outcome reward; PRM-shaped RLVR uses a step-level reward. The
spectrum:

```
RLHF                RLVR (outcome)        PRM-shaped RLVR
↑ noisy, dense     ↑ clean, very sparse  ↑ less noisy than RLHF, denser than RLVR
```

The "PRM-shaped RLVR" middle ground is theoretically attractive but
practically risky: PRMs themselves are imperfect, and rewarding a
misaligned PRM produces *process hacking* (steps that look correct
without yielding correct answers) — a step-level analogue of the
outcome-hacking problem `[arXiv 2505.00551]`.

## 5. Frontier and open questions (as of 2026-05)

- **Discriminative vs generative PRM scope.** [CONTRADICTION] Open as
  of 2026 `[zheng2025-prm-survey]`. Strongest discriminative PRMs trained on
  1.5% data (EDU-PRM) compete with generative ThinkPRM on
  ProcessBench; the comparison flips on harder distributions.
- **Self-PRM.** Can a single LLM serve as both policy and verifier
  (`generate then verify` with the same weights, possibly with a
  prompt switch)? Several 2026 papers `[arXiv 2505.00551 §4 review]`
  attempt this; results are mixed. The risk: a model's own evaluator
  inherits its own blind spots.
- **PRM faithfulness.** A PRM can score a step highly without that
  score being grounded in the step's correctness — analogous to the
  CoT faithfulness question. Generative PRMs are more interpretable
  but also more capable of producing rationalising verdicts.
- **PRMs beyond math.** Step-level signals are easy in math (each
  equation step is checkable) and code (each function is testable).
  In open-domain QA, scientific reasoning, planning — what is a
  "step"? What is a "step-correctness label"? Open.
- **Reward / process hacking under PRM-shaped RL.** The known failure
  mode is policies that produce verbose, locally-plausible steps that
  the PRM scores well but that do not lead to correct answers
  `[arXiv 2505.00551]`. Mitigations: ORM + PRM combined rewards,
  adversarial PRM training, KL constraints on policy.

## 6. Intuitions and analogies

[ANALOGY] An ORM is a **thumbs-up at the end of the meal**; a PRM is
a **review after every course**. The analogy returns to canonical form
through Eq. (1) vs Eq. (2): the supervisory signal is concentrated at
$T = $ end vs distributed across $t = 1, \ldots, T$. With dense per-
step signal, the credit-assignment problem is easier — a noisy step
is identified locally rather than as a credit-share of a final
failure.

[INTUITION] Why min-aggregation works. If steps are heterogeneous and
even one wrong step can derail the answer, the trace's value is
bounded above by its weakest step. Mean / product would let many
strong steps mask one critical error; min surfaces it. The empirical
finding `[lightman2023-prm800k §4.2]` is consistent with this picture.

[INTUITION] Generative PRMs are **chain-of-thought verifiers**. They
are the same trick as CoT for the policy — let the verifier think
before deciding — applied to the verifier role. Why this works under a
small training budget: the verifier task is structurally easier than
the policy task (verify > generate), so a smaller model that thinks
can match a larger model that classifies in one shot
`[thinkprm2025 §4]`.

[ANALOGY] PRMs as **type-checkers for reasoning**. A type-checker
emits per-line errors; a runtime tester emits a single pass / fail.
Process supervision is to outcome supervision what static analysis is
to integration testing. The analogy is informal — types are formal,
PRM signals are statistical — but the gradient-of-locality intuition
returns to canonical form via the per-step labels of Eq. (2).

## 7. See also

- `kb/notes/reasoning/test-time-compute.md` — best-of-$N$ + PRM is the
  highest-leverage TTC strategy on math.
- `kb/notes/reasoning/inference-time-search.md` — MCTS / beam over
  reasoning steps uses a PRM as the value function.
- `kb/notes/reasoning/reasoning-training.md` — when PRMs serve as
  training signal (rStar-Math, PRM-shaped GRPO).
- `kb/notes/reasoning/chain-of-thought.md` — the substrate that PRMs
  evaluate.
- `kb/notes/post-training/rlvr-and-grpo.md` — relation between
  outcome rewards (RLVR) and process rewards.
