---
topic: post-training/rlvr-and-grpo
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - shao2024  # GRPO origin
  - deepseek-r1
  - dapo2025
  - ouyang2022-instructgpt  # for the PPO-RLHF baseline
  - rlvr-limits-2025
secondary_sources:
  - tulu3
  - qwen3
related_topics:
  - post-training/rlhf
  - reasoning/reasoning-training
  - reasoning/process-supervision
  - reasoning/test-time-compute
  - scaling/inference-time-compute-scaling
---

# RLVR and GRPO

**RLVR** (Reinforcement Learning with Verifiable Rewards) is the
post-training paradigm where the reward signal is a **programmatic
verifier** — exact-match for math, unit tests for code, format checker
for structured output — rather than a learned reward model trained on
human preferences `[deepseek-r1 §2.2; kb/excerpts/deepseek-r1#sec-2-2-reward]`.
**GRPO** (Group Relative Policy Optimization) is the dominant
RL algorithm of the RLVR era: a critic-free PPO variant that uses
group-mean reward as the value baseline `[shao2024 §4.1;
kb/excerpts/shao2024#sec-4-1]`, restated with verified anchors in
DeepSeek-R1 `[deepseek-r1 §2.1, Eq.1, Eq.3; kb/excerpts/deepseek-r1#sec-2-1-grpo]`. RLVR + GRPO is the post-training
recipe behind DeepSeek-R1, the Qwen 2.5/3 reasoning lines, Tülu 3,
and the open RL-for-reasoning ecosystem of 2024–2026.

This note covers (a) the **foundations** — verifiable rewards, GRPO
objective, KL anchoring — and (b) the **variant lineage** — DAPO,
VAPO, REINFORCE++, VinePPO. Phase 1 flagged splitting this topic into
two leaf nodes; we keep it as one note with two clearly-labeled
sub-sections, per the topics.md "held for Phase 2 review" note.

## 1. Formal definition

### 1.1 The verifiable-reward setting

In standard RLHF the reward $r_\phi(x,y)$ is a neural network trained
on human preference comparisons (cf.
`kb/notes/post-training/rlhf.md` and `kb/excerpts/ouyang2022-instructgpt#sec-3`).
In RLVR the reward is **a deterministic function**:

$$
R_{\mathrm{verifier}}(x, y) = \mathbb{1}\big[\, \mathrm{verify}(x, y) = \mathsf{true}\,\big] \in \{0,1\}.
$$

For math, $\mathrm{verify}$ is exact match against the canonical answer
(after normalization). For code, $\mathrm{verify}$ is "compiles and
passes the test cases." For format compliance, $\mathrm{verify}$ is
a regex / parser. The reward is **sparse** (one bit per trajectory)
and **terminal** (only the final answer matters)
`[deepseek-r1 §2.2; kb/excerpts/deepseek-r1#sec-2-2-reward]`.

This sidesteps two failure modes of learned reward models:

- **Reward hacking** — the policy exploits an artifact of the RM that
  doesn't correspond to true quality (verbosity, sycophancy). With a
  programmatic verifier, the only way to score 1 is to actually solve
  the problem.
- **Reward overoptimization** — the gap between proxy reward and gold
  reward grows as KL($\pi_\theta\|\pi_{\mathrm{ref}}$) grows
  (Gao et al. 2023). Verifiers don't have this gap because they *are*
  the gold reward (modulo dataset-side issues like wrong gold answers).

### 1.2 The GRPO objective

The policy-gradient objective `[shao2024 §4.1;
kb/excerpts/shao2024#sec-4-1]`:

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta) = \mathbb{E}_{q\sim P(Q),\, \{o_i\}_{i=1}^G \sim \pi_{\theta_\mathrm{old}}(\cdot|q)} \left[ \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\!\left( r_{i,t}\hat{A}_{i,t},\, \mathrm{clip}(r_{i,t},1-\epsilon,1+\epsilon)\hat{A}_{i,t} \right) - \beta\, \mathbb{D}_{\mathrm{KL}}\!\left[\pi_\theta\|\pi_{\mathrm{ref}}\right] \right] \tag{1}
$$

with importance ratio $r_{i,t}(\theta) =
\pi_\theta(o_{i,t}|q,o_{i,<t}) /
\pi_{\theta_{\mathrm{old}}}(o_{i,t}|q,o_{i,<t})$ and **group-normalized
advantage**

$$
\hat{A}_{i,t} = \tilde{A}_i = \frac{R_i - \mathrm{mean}(\{R_1,\ldots,R_G\})}{\mathrm{std}(\{R_1,\ldots,R_G\})}, \tag{2}
$$

constant in $t$ across the trajectory `[shao2024 §4.1;
kb/excerpts/shao2024#sec-4-1]`.

Symbols:

| Symbol | Meaning |
|---|---|
| $q$ | prompt sampled from training distribution |
| $G$ | group size; number of rollouts per prompt (typical: 8–64) |
| $o_i$ | the $i$-th sampled trajectory (chain-of-thought + answer) |
| $R_i$ | terminal reward for trajectory $o_i$ (typically 0/1 from verifier) |
| $\hat{A}_{i,t}$ | advantage estimate at token $t$ of trajectory $i$ |
| $\pi_{\mathrm{ref}}$ | frozen reference policy (typically the SFT cold-start) |
| $\beta$ | KL coefficient (typical: 0.001–0.04) |
| $\epsilon$ | PPO clip range (typical: 0.2) |

### 1.3 The KL anchor

The KL term $\mathbb{D}_{\mathrm{KL}}[\pi_\theta\|\pi_{\mathrm{ref}}]$
keeps the trained policy from drifting from the reference. Without it,
the policy optimizes the verifier without constraint and can produce
reward-hacking behaviors (e.g., outputting only the answer with no
reasoning, or producing degenerate text patterns that the verifier
accepts). With it, the optimization is trust-region constrained: the
policy can only move in directions the reference still has support for.

Empirically $\beta$ is often very small or scheduled to decay
`[deepseek-r1 §2.2]`. DAPO removes the KL term entirely in some
configurations and replaces its role with sampling discipline (see
§3.1).

## 2. Mechanism — the protocol

### 2.1 Per-step procedure

For each training step on a batch of $B$ prompts:

1. **Sample group.** For each prompt $q^{(b)}$, sample $G$ trajectories
   $\{o_i^{(b)}\}_{i=1}^G$ from $\pi_{\theta_{\mathrm{old}}}$ at a
   temperature $T \in [0.7, 1.0]$. Trajectories are typically capped at
   a max-length $L_{\max}$ (8K–32K tokens) and may include a structured
   thinking section (`<think>...</think>`).

2. **Score.** Run the verifier on each trajectory to get $R_i^{(b)}$.

3. **Compute advantages.** For each prompt's group, compute the
   group-normalized $\tilde{A}_i^{(b)}$ via Eq. (2). Broadcast the
   trajectory-level $\tilde{A}$ to every token.

4. **PPO inner-loop.** Take $K \in \{1,4\}$ gradient steps on $\theta$
   using the loss in Eq. (1) over the buffered $(B \cdot G)$ trajectories
   `[shao2024 §4.1; kb/excerpts/shao2024#sec-4-1]`. Each step updates
   $\theta$ a small distance from $\theta_{\mathrm{old}}$.

5. **Refresh.** Set $\theta_{\mathrm{old}} \leftarrow \theta$ and continue.

### 2.2 Memory profile (why GRPO matters)

PPO with GAE requires a **value model** $V_\phi(s_t)$ co-sized with the
policy `[ouyang2022-instructgpt §3.5;
kb/excerpts/ouyang2022-instructgpt#sec-3]`. At LLM scale this **doubles
the active parameter count and activation memory** of training. GRPO's
group-mean baseline replaces $V_\phi$ entirely, removing the second
network. The cost is increased rollout count: $G$ trajectories per prompt
instead of 1
`[shao2024 §4.1; kb/excerpts/shao2024#sec-4-1-motivation]`.

This trade is favorable when (a) policy size is large and value-model
co-size is the binding constraint, (b) reward is sparse (so a value
function doesn't help much anyway), and (c) the training infrastructure
can amortize rollout-time. For most modern reasoning training runs all
three hold.

### 2.3 The verifiable-reward set in practice

DeepSeek-R1's reward `[deepseek-r1 §2.2; kb/excerpts/deepseek-r1#sec-2-2-reward]`
is a sum of:

- **Accuracy reward** (binary, from verifier): math equivalence (sympy
  + custom normalization), code unit-tests, multi-choice exact-match.
- **Format reward** (small constant, e.g. 0.5): the trajectory contains
  a well-formed `<think>...</think>` block before the answer.

DeepSeek explicitly notes they "abstain from applying neural reward
models" because of reward-hacking susceptibility under large-scale RL
`[deepseek-r1 §2.2; kb/excerpts/deepseek-r1#sec-2-2-reward]`.

R1 does not use a process reward model (PRM) for the headline RL run.
Process-supervised RLVR is treated separately in
`kb/notes/reasoning/process-supervision.md`.

## 3. Variants and lineage

The GRPO design is one of several critic-free variants that have
emerged in the RLVR era. They differ along three axes: (a) what to do
with the value baseline, (b) what to do with the clip range, (c) what
to do with the KL.

### 3.1 DAPO — four targeted improvements over GRPO

DAPO introduces four modifications to GRPO `[dapo2025 §2;
kb/excerpts/dapo2025#sec-2]`:

| Technique | Modification | Why |
|---|---|---|
| **Clip-Higher** | $\epsilon_{\mathrm{low}} < \epsilon_{\mathrm{high}}$ | Allow larger upward updates on under-probable correct tokens |
| **Dynamic Sampling** | Reject groups where all $R_i$ are equal | Eliminate zero-gradient batches; concentrate compute on informative samples |
| **Token-Level Loss** | $\sum_t L_t / \sum_i \|o_i\|$ instead of $\sum_i \sum_t L_t/(G\|o_i\|)$ | Equal-weight every token regardless of trajectory length; avoid underweighting long correct CoTs |
| **Overlong Reward Shaping** | Soft penalty for length-truncated trajectories | Length-cut failures are uninformative; soft-shape reduces variance |

Empirical headline `[dapo2025 §3-headline; kb/excerpts/dapo2025#sec-3-headline]`:
50 points on AIME 2024 with Qwen2.5-32B base, ~50% fewer training steps
than the GRPO baseline. Open-sourced via the verl framework.

### 3.2 VAPO — putting the value model back

VAPO (Value-Augmented PPO, ByteDance, April 2025, arXiv:2504.05118)
re-introduces a learned value function but trains it more carefully than
GAE-PPO: shared backbone with the policy, value-loss-clipped, advantage
normalized per-batch. Phase 1 sweep reports VAPO outperforms DAPO on
long-CoT tasks in fewer steps, suggesting the GRPO-vs-PPO advantage is
not "value models are useless" but "naïve value models are
expensive-and-useless; carefully-trained ones are useful."
**[CONTRADICTION]** Whether VAPO's gain over DAPO replicates outside
the original recipe is an open question as of 2026-05; only the original
paper has reported the comparison.

### 3.3 VinePPO — Monte Carlo credit assignment

VinePPO (Kazemnejad et al., ICML 2025, arXiv:2410.01679) keeps the PPO
framework but replaces the value network's per-token bootstrap with
**Monte Carlo re-simulation**: at each intermediate state $s_t$, sample
$M$ continuations, compute their average reward, and use that as
$V(s_t)$. Reported 3× wall-clock speedup over PPO on MATH/GSM8K because
the rollouts can be batched and re-used across mini-batches.
[INTUITION] VinePPO is what happens if you take "GRPO's group-mean is a
single-state MC value estimate" and apply it at every prefix instead of
just the prompt.

### 3.4 REINFORCE-class methods

REINFORCE++ (and the family of RLOO, GRPO-no-clip, etc.) drop the
PPO-style clip entirely. The argument: the clip protects against
off-policy drift, but at small per-step learning rates and with KL
anchoring already present, the clip is a redundant safeguard. Whether
clip-or-no-clip matters in practice is contested in 2025–2026 papers;
implementations like `verl` and `trl` support both and the empirical
gap is small.

### 3.5 Comparison table

| Method | Critic | Clip | KL | Advantage estimator |
|---|---|---|---|---|
| **PPO-RLHF** (Ouyang 2022) | $V_\phi$, GAE | symmetric $\epsilon$ | per-token in reward | $\hat{A}^{\mathrm{GAE}}_t$ |
| **GRPO** (Shao 2024) | none (group baseline) | symmetric $\epsilon$ | term in objective | $\tilde{A}_i$ (constant in $t$) |
| **DAPO** (ByteDance 2025) | none | asymmetric $[\epsilon_l, \epsilon_h]$ | optional / removed | $\tilde{A}_i$, dyn-sampled |
| **VAPO** (ByteDance 2025) | $V_\phi$, value-loss-clip | symmetric $\epsilon$ | per-token in reward | shared-backbone value |
| **VinePPO** (Kazemnejad 2025) | MC re-sim, no $V_\phi$ | symmetric $\epsilon$ | per-token in reward | per-prefix MC mean |
| **REINFORCE++** | none (group baseline) | none | KL anchor only | $\tilde{A}_i$ (constant in $t$) |

`[deepseek-r1 §2.2; dapo2025 §2; shao2024 §4.1]`

## 4. Intuitions and analogies

[ANALOGY] **GRPO is the median-of-rollouts as policy gradient.** If you
sample $G$ trajectories from the current policy, score each, and ask
"which were better than typical?", the answer is exactly the
sign-of-$\tilde{A}_i$. Up-weight the better-than-mean rollouts'
log-probs; down-weight the worse-than-mean ones. The canonical math
returns to Eq. (2): $\tilde{A}_i$ is just the standardization of $R_i$
within its group.

[ANALOGY] **The verifier is a pass/fail oracle, not a teacher.** RLVR
doesn't tell the policy *how* to be better, only *whether* it is. The
optimization works because the policy can already produce the correct
answer some fraction $p_0$ of the time at the start of training; RLVR
amplifies the trajectories that lead to correct answers and suppresses
the ones that don't, without any process-level guidance. This is also
the basis of the `[CONTRADICTION]` in §5 below: if the base model
already had to be capable of $R_i = 1$ for some $i$ to bootstrap, then
RLVR is necessarily "amplification of pre-existing capability" rather
than "creation of new capability."

[INTUITION] **Why group-relative beats absolute baselines.** The point
of a value baseline in REINFORCE is variance reduction:
$\mathrm{Var}(\hat{A}) < \mathrm{Var}(R)$ for any baseline that doesn't
correlate with the policy's specific action. A learned $V_\phi$ would in
principle reduce variance more than the group mean, because it
conditions on the prompt. But $V_\phi$ has its **own** variance — it has
to be trained, and at initialization is wrong. The group mean is
zero-variance (it's an exact statistic) and unbiased on-policy; in the
sparse-reward regime where $V_\phi$ struggles to learn anyway, the
group mean wins.

[INTUITION] **Why the format reward matters.** Without it, the policy
under verifier reward learns to output only the answer. The format
reward (small constant for `<think>...</think>` block presence) keeps
the chain-of-thought pathway alive during early training, which is
necessary for the policy to ever discover *new* solution paths beyond
its already-known shortcut answers
`[deepseek-r1 §2.2; kb/excerpts/deepseek-r1#sec-2-2-reward]`. R1 also
reports an "aha moment" during training where the model's use of
"wait" tokens during reflection spikes
`[deepseek-r1 §2.3; kb/excerpts/deepseek-r1#sec-2-3-aha]`, suggesting
the chain-of-thought structure is being actively reorganized rather
than just used.

## 5. Frontier and open questions (as of 2026-05)

### 5.1 [CONTRADICTION] Does RLVR create new reasoning capability?

The headline `[CONTRADICTION]` of the field
`[rlvr-limits-2025; arXiv:2504.13837]`:

- **DeepSeek-R1, Tülu 3, Qwen 3 framing:** RLVR "incentivizes" or
  "elicits" reasoning. The implication is that the policy becomes
  capable of solving problems it could not solve before.
- **Yue et al. NeurIPS 2025 oral (arXiv:2504.13837):** Carefully
  comparing pass@1 vs pass@K of base vs RLVR'd models, they show RLVR
  **improves pass@1** (sampling efficiency) on problems the base model
  can already solve at higher K, but **does not expand the set of
  problems solvable at any K**. The capability ceiling is set by
  pre-training. RLVR is sampling-efficient sharpening, not capability
  expansion.

This is a strong claim and the empirical setup (matching K, controlling
for the sampling temperature, choice of benchmark) is contested in
follow-up papers. As of 2026-05 the consensus in the open-source
research community (per Phase 1 sweep §6.2) is leaning toward "Yue et
al. is mostly right on current math/code benchmarks, but the framing
has caveats on temperature and sample budget."

### 5.2 RLVR beyond verifiable domains

RLVR works for math, code, formal proofs, and structured generation —
domains with a programmatic check. **Open question:** how to extend it
to domains without verifiers (creative writing, open-ended QA, agentic
tasks). 2025–2026 directions:

- **LLM-as-judge as a soft verifier** (with the inherent reward-hacking
  risk of any learned reward model).
- **Process Reward Models** (PRMs) — see
  `kb/notes/reasoning/process-supervision.md`.
- **Programmatic checks plus LLM judgment** — hybrid signals.
- **Self-play with verifier symmetry** — agentic settings.

### 5.3 Group size, prompt diversity, and the dataset distribution

GRPO's variance-reduction depends on having a non-degenerate group: at
least one trajectory at $R = 1$ and at least one at $R = 0$. DAPO's
"dynamic sampling" enforces this. **Open question:** what's the optimal
$G$ as a function of base-model accuracy on the dataset? Phase 1 sweep
notes some practitioners use $G = 8$, others $G = 64$; no systematic
study yet.

### 5.4 The role of cold-start SFT

DeepSeek-R1-Zero (no SFT cold-start) reaches a high level of reasoning
but produces less readable text than DeepSeek-R1 (with cold-start). The
cold-start serves as a (a) regularizer of language quality and (b) seed
of the `<think>...</think>` format. **Open question:** is the cold-start
necessary for capability or only for stylistic/legibility reasons?
Anecdotal evidence from 2025–2026 suggests removing cold-start
sometimes destabilizes RL, sometimes works fine — the determining factor
is unclear.

### 5.5 Process reward models and reasoning trees

Recent work (ThinkPRM `[thinkprm2025]`, R-PRM, FOVER) trains generative
process reward models that themselves reason about correctness of
intermediate steps. These can substitute for or augment terminal
verifiers in RLVR. The **frontier question** is whether process-level
RL fundamentally beats terminal RL on reasoning tasks; current evidence
is mixed.

## 6. See also

- `kb/notes/post-training/rlhf.md` — the PPO-RLHF baseline that GRPO is a
  variant of.
- `kb/notes/post-training/dpo-and-offline.md` — the offline alternative
  to RL altogether; orthogonal to RLVR.
- `kb/notes/reasoning/reasoning-training.md` — the broader story of
  training models to reason; RLVR is one branch.
- `kb/notes/reasoning/process-supervision.md` — PRMs, ThinkPRM, the
  process-level alternative to terminal RLVR.
- `kb/notes/reasoning/test-time-compute.md` — pass@K and sampling-budget
  scaling, which interacts directly with the §5.1 contradiction.
- `kb/notes/scaling/inference-time-compute-scaling.md` — the scaling-law
  framing of RLVR + test-time compute.
