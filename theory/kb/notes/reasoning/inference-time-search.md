---
topic: reasoning/inference-time-search
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - rstar-math2025
  - lightman2023-prm800k
secondary_sources:
  - rest-mcts-2024
  - tree-search-survey-2025
  - inference-scaling-survey-2025
  - mcts-rag-2025
  - snell2024
related_topics:
  - reasoning/test-time-compute
  - reasoning/process-supervision
  - reasoning/reasoning-training
  - reasoning/chain-of-thought
---

# Inference-time search

**Inference-time search** is the family of test-time-compute strategies
that explicitly construct and explore a tree (or DAG) of partial
reasoning prefixes, using a verifier or value function to direct
expansion. It is the *search-based* leg of the TTC taxonomy in
`kb/notes/reasoning/test-time-compute.md`. This note specialises to the
search algorithms — beam, BFS, MCTS — and their use of process reward
models as value functions.

> **PDF-fetch caveat (2026-05-04):** primary PDFs for `rstar-math2025`,
> the tree-search survey, and the inference-scaling survey were not
> downloaded in this Phase 2 pass. Section anchors below are to the
> canonical arXiv-version sectioning. Verbatim excerpt files are
> skeleton-only.

## 1. Formal definition

Given a problem $x$ and a partial trace $z_{1:t} = (s_1, \ldots, s_t)$,
treat the next step $s_{t+1}$ as an action drawn from
$\pi_\theta(\cdot \mid x, z_{1:t})$. The **search tree** has nodes
$z_{1:t}$ and edges $s_{t+1}$. A **value function** $V_\phi(x,
z_{1:t}) \to \mathbb{R}$ scores partial states; a **terminal verifier**
$\mathcal{V}(x, y) \to \{0, 1\}$ scores complete answers.

Search algorithms differ in how they combine $\pi_\theta$ (the policy
prior on actions) and $V_\phi$ (the value-based pruning signal):

- **Beam search:** at each depth, retain the top-$B$ partials by score.
- **BFS / level-order:** expand all partials at each depth (with
  pruning).
- **MCTS:** select-expand-simulate-backup loop, balancing
  exploration / exploitation via a UCT-style rule.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $B$ | beam width / branching factor |
| $D$ | maximum search depth (max steps) |
| $V_\phi$ | learned value function (typically a PRM) |
| $\mathcal{V}$ | terminal verifier (rule-based for math/code) |
| $N_\text{rollout}$ | number of MCTS rollouts (proxy for compute) |
| $c_\text{puct}$ | UCT exploration constant |

## 2. Mechanism

### 2.1 Beam over reasoning steps

Algorithm: maintain a frontier of $B$ partial traces; at each depth,
expand each by sampling $k$ next-step continuations; rescore the
resulting $B \cdot k$ partials with $V_\phi$; retain top $B$.

Cost: $O(B \cdot k \cdot D \cdot L_\text{step})$ tokens. Simple to
implement; effective when the policy's per-step entropy is moderate
and the verifier has good calibration on partial states
`[snell2024 §4 method.beam]`.

### 2.2 MCTS over reasoning steps (rStar-Math)

The canonical reasoning-MCTS formulation `[rstar-math2025 §3]`:

1. **Selection.** From the root, descend the tree by repeatedly
   choosing the child maximising
   $$U(s) = Q(s) + c_\text{puct} \, \pi_\theta(s \mid \text{parent}) \frac{\sqrt{N_\text{parent}}}{1 + N_s}$$
   where $Q(s)$ is the running mean of rollout outcomes through $s$
   and $N_s$ is the visit count.
2. **Expansion.** When reaching a leaf, sample $k$ candidate next-
   steps from $\pi_\theta$.
3. **Simulation.** From an expanded child, roll out a complete trace
   under $\pi_\theta$; score the final answer with $\mathcal{V}$ (and
   optionally evaluate the partial $V_\phi$).
4. **Backup.** Propagate the rollout's reward up the path, updating
   $Q$ and $N$ at each visited node.

After $N_\text{rollout}$ iterations, the answer is selected as the
most-visited path from root or the rollout that achieved highest
reward.

Cost: $O(N_\text{rollout} \cdot D \cdot L_\text{step})$ tokens. The
critical-path constant is the cost of a full rollout (one complete
trace per simulation step), which dominates expansion cost.

### 2.3 ReST-MCTS* (training-data generation)

`[rest-mcts-2024]` uses MCTS not just for inference but for **iterative
self-training**:

1. Run MCTS with current policy + PRM.
2. Collect MCTS-discovered correct trajectories (as judged by terminal
   verifier $\mathcal{V}$).
3. SFT the policy on these trajectories.
4. Update the PRM using the new policy's distribution.
5. Repeat.

This is structurally identical to AlphaZero's self-play loop, with
"games" replaced by "math problem solutions" and "MCTS rollouts under
a learned value function" replacing self-play. The training-side
counterpart is documented in
`kb/notes/reasoning/reasoning-training.md §3.3`.

## 3. The role of the PRM

Search-based TTC depends on a value function that is informative on
**partial** traces. PRMs (see
`kb/notes/reasoning/process-supervision.md`) are the standard choice
because their training signal is directly per-step.

### 3.1 PRM as MCTS value head

In rStar-Math, the PRM serves both as the simulation reward (terminal
states score under $\mathcal{V}$, intermediate ones under PRM) and as
a guidance signal during selection. The policy and PRM are co-trained
across rounds, so the PRM stays calibrated to the current policy's
output distribution `[rstar-math2025 §3.2]`.

### 3.2 Generator-discriminator mutual verification (rStar / rStar-Math)

A distinguishing feature of the rStar lineage `[arXiv (rStar)
forum-signal]`: two SLM roles that mutually verify.

- **Policy SLM** generates candidate next-steps.
- **Reward / discriminator SLM** scores them.

The policy-discriminator separation is similar to GAN-era setups but
with the discriminator as a step-evaluator, not a real / fake
classifier. The mutual verification allows small models (7B) to reach
o1-mini-class performance on competition math `[rstar-math2025 §4]`.

## 4. Variants and lineage

| Method | Year | Algorithm | Value function | Notable result |
|---|---|---|---|---|
| Tree-of-Thoughts | 2023 | BFS / DFS / beam over thought nodes | Heuristic LLM-rater | Established the "thought tree" framing `[arXiv 2305.10601]` |
| Graph-of-Thoughts | 2023 | Generalised graph allowing merges | LLM-rater | Niche `[arXiv 2308.09687]` |
| Lookahead beam (Snell) | 2024 | Beam over partial steps with PRM rollout-scoring | PRM | Compute-optimal mix of beam vs sampling `[snell2024 §4]` |
| ReST-MCTS* | 2024 | MCTS + iterative self-training | PRM | NeurIPS 2024; bootstrap data via search `[rest-mcts-2024]` |
| rStar / rStar-Math | 2024–25 | MCTS with policy-SLM + reward-SLM | PRM (trained alongside) | 90% MATH at 7B `[rstar-math2025 §4]` |
| AGoT (Adaptive GoT) | 2025 | Single framework adaptively chooses chain / tree / graph | LLM-rater | Open whether competitive with trained long-CoT `[arXiv 2502.05078]` |
| MCTS-RAG | 2025 | MCTS over retrieval + reasoning decisions | Hybrid | Bridges search and RAG `[mcts-rag-2025, arXiv 2503.20757]` |

[CONTRADICTION] The 2025 question is whether explicit search **adds
value over** a long-CoT trained model that internally simulates many
branches via its trained trace. Some 2026 measurements suggest at
matched compute budget, R1-distill + budget forcing matches MCTS +
PRM on competition math `[arXiv 2510.10787]`; others find search
still wins on the hardest tail. The taxonomy `tree-search-survey-2025`
catalogues the disagreement without resolving it.

## 5. The search-vs-trained-CoT debate

The 2024 hypothesis was: "long-CoT models internalise tree search; you
don't need explicit search at inference." The 2025 evidence is mixed:

- **For internalisation:** R1's training-time trace-length growth from
  200 to 10K tokens looks behaviourally like a model learning to
  consider and reject branches in-line `[deepseek-r1 §2.2.2]`. R1-
  distilled small models reach near-MCTS performance without explicit
  tree construction `[deepseek-r1 §3.2]`.
- **Against internalisation:** rStar-Math reaches 90% MATH at 7B with
  explicit MCTS, comparable to R1-distill-32B at much smaller policy
  scale. At small scales, explicit search wins
  `[rstar-math2025 §4]`.

A clean reading: explicit search is most valuable where the policy is
weak (small model, narrow domain, hard tail of a distribution). At
strong policy + standard distributions, internalised search via long
CoT is competitive. This matches Snell 2024's compute-optimal-mix
finding: search dominates on hard problems
`[snell2024 §5]`.

## 6. Frontier and open questions (as of 2026-05)

- **Generalisation beyond math.** All published rStar / rStar-Math /
  reasoning-MCTS work focuses on math (and some code). Step-level
  reward signals are easy there. Whether MCTS over reasoning steps
  helps in factual QA, agent planning, or open-ended reasoning is
  largely open `[tree-search-survey-2025 §5 future]`.
- **Search budget vs trace budget.** [CONTRADICTION] Reports vary on
  whether MCTS-with-rollouts uses fewer total tokens than budget-
  forced long CoT for the same accuracy. Likely policy- and benchmark-
  dependent.
- **Compute amortisation across problems.** MCTS-discovered
  trajectories can be cached and used for related problems (transfer
  via SFT or in-context retrieval). This is exploited by ReST-MCTS*
  for training but not yet systematically exploited at inference time.
- **Search for tool-using agents.** MCTS-RAG `[mcts-rag-2025]` extends
  search to retrieval decisions. Generalising to arbitrary tool calls
  (compute, web, code execution) is an active direction; the value
  function must score tool-use plans, not just text.
- **PRM-free search.** When no step-level verifier is available, can
  one bootstrap value estimates from rollout-with-outcome-verifier
  alone? Standard MCTS works this way (sparse rewards), but for LLM
  reasoning the rollout cost dominates. Open whether short-rollout
  approximations match PRM guidance.

## 7. Intuitions and analogies

[ANALOGY] Reasoning-MCTS is **AlphaZero with text steps**. Selection,
expansion, simulation, backup are the same; the action space is the
LLM's next-step distribution; the value head is a PRM. The analogy
returns to canonical form via the UCT formula in §2.2 — same equation,
different domain. This analogy is load-bearing: it explains why
co-training the PRM (value head) with the policy-via-search-SFT loop
is effective, mirroring AlphaZero's self-play
`[rest-mcts-2024 §3]`.

[INTUITION] Beam search and MCTS exchange **anytime** for **target
quality**: MCTS gets better with more rollouts, beam gets better with
wider beams, but both are bounded by the policy's per-step quality. A
weak policy makes branch quality low, no amount of search compensates;
a strong policy makes search redundant. The sweet spot is moderate-
quality policies on hard problems, which is exactly where rStar-Math's
7B + MCTS configuration sits.

[INTUITION] Why long-CoT can substitute for explicit tree search.
Inside a long trace the model can write things like "let me try x …
that doesn't work because y, let's try z instead". Each of these is a
search-tree-like operation realised in token space rather than as
explicit branching. The policy is doing implicit search inside the
attention computation. The computational equivalence is informal but
the empirical data are consistent.
[CONTRADICTION] Whether implicit and explicit search are
*computationally equivalent* in any precise sense is open.

## 8. See also

- `kb/notes/reasoning/test-time-compute.md` — search-based TTC sits
  here in the broader taxonomy.
- `kb/notes/reasoning/process-supervision.md` — PRMs as the value
  functions search needs.
- `kb/notes/reasoning/reasoning-training.md` — search as a training-
  data generator (rStar-Math, ReST-MCTS*).
- `kb/notes/reasoning/chain-of-thought.md` — the substrate over which
  search expands.
