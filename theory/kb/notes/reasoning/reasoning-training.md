---
topic: reasoning/reasoning-training
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - deepseek-r1
  - dapo2025
  - rstar-math2025
  - rlvr-limits-2025
secondary_sources:
  - lightman2023-prm800k
  - thinkprm2025
  - s1-2025
  - qwen3
related_topics:
  - reasoning/test-time-compute
  - reasoning/chain-of-thought
  - reasoning/process-supervision
  - reasoning/inference-time-search
  - post-training/rlvr-and-grpo
  - architecture/reasoning-architectures
---

# Reasoning training

This note covers the training-side methods that produce models whose
inference-time behaviour includes long, useful chain-of-thought traces
— the so-called "thinking" or "reasoning" models (o1-class, DeepSeek-R1,
QwQ, Gemini-2.5-thinking). The key technical idea is **RLVR** —
reinforcement learning with verifiable rewards — typically using GRPO
or a close variant. This note is the application-side companion to
`kb/notes/post-training/rlvr-and-grpo.md`, which covers GRPO as a
general optimisation method.

> **PDF-fetch caveat (2026-05-04):** primary PDFs for `deepseek-r1`,
> `dapo2025`, `rstar-math2025`, and `rlvr-limits-2025` were not
> downloaded in this Phase 2 pass. Section anchors below refer to the
> canonical arXiv-version structure. Verbatim excerpt files are
> skeleton-only and must be populated before any LaTeX propagation.

## 1. Formal definition

Given a base LLM $\pi_\theta$ and a verifier $r: (x, y) \to \mathbb{R}$
(typically rule-based: $r = 1$ if extracted answer matches ground truth
else $0$), reasoning training optimises:

$$\theta^* = \arg\max_\theta \mathbb{E}_{x \sim \mathcal{D}, \, y \sim \pi_\theta(\cdot \mid x)} \bigl[r(x, y) - \beta \, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})\bigr] \tag{1}$$

with $\pi_{\text{ref}}$ a frozen reference policy (typically the
SFT-initialised model) and $\beta$ a KL-regularisation coefficient
`[deepseek-r1 §2.2.1]`. The defining feature versus generic RLHF is
that $r$ is **verifiable**: it is computed mechanically from the
emitted answer (regex match for math, unit-test execution for code,
proof-checker for formal proofs), not from a learned reward model
trained on human preferences `[deepseek-r1 §2.2.1]`.

### 1.1 GRPO objective

GRPO (Group Relative Policy Optimisation) avoids training a separate
critic by estimating advantages from the empirical reward distribution
within a group of $G$ samples per prompt
`[deepseek-r1 §2.2.1, eq.GRPO]`:

$$A_{i,t} = \frac{r_i - \mathrm{mean}(\{r_1, \ldots, r_G\})}{\mathrm{std}(\{r_1, \ldots, r_G\})} \tag{2}$$

The PPO-style clipped policy update uses these group-normalised
advantages:

$$\mathcal{L}_\text{GRPO}(\theta) = \mathbb{E}\!\left[\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\!\bigl(\rho_{i,t} A_{i,t},\, \mathrm{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon) A_{i,t}\bigr)\right] - \beta \, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) \tag{3}$$

where $\rho_{i,t} = \pi_\theta(y_{i,t} \mid x, y_{i, < t}) /
\pi_{\theta_\text{old}}(y_{i,t} \mid x, y_{i, < t})$ is the importance
ratio.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $G$ | group size (number of samples per prompt; R1 uses $G \approx 16$) |
| $r_i$ | binary or sparse reward for the $i$-th sample |
| $A_{i,t}$ | advantage assigned to position $t$ of sample $i$ |
| $\rho_{i,t}$ | per-token importance ratio (PPO style) |
| $\epsilon$ | clip range (typically 0.2; DAPO splits into asymmetric clip-low/clip-high) |
| $\beta$ | KL coefficient |
| $\pi_{\text{ref}}$ | frozen SFT-initialised reference policy |

## 2. Mechanism — the R1 training pipeline

R1 establishes a four-stage protocol that subsequent open-replication
work (Tülu 3, OpenR1, Qwen3 thinking) largely follows
`[deepseek-r1 §2.3]`:

### Stage 1: Cold-start SFT

A small set ($\sim 10^4$) of high-quality long-CoT exemplars is used
for SFT on the base model, producing a "readable" CoT initialisation.
Skipping this stage gives R1-Zero, which has the same reasoning
capability but produces unreadable interleaved-language traces
`[deepseek-r1 §2.2.4]`.

### Stage 2: Reasoning-oriented RL (RLVR / GRPO)

GRPO with rule-based verifier rewards on math, coding, and STEM
problems with verifiable answers. During this stage, average response
length grows from $\sim 200$ to $\sim 10{,}000$ tokens, accompanied by
benchmark-score growth (AIME 2024: $15.6\% \to 71\%$)
`[deepseek-r1 §2.2.2 fig.3]`.

### Stage 3: Rejection-sampling SFT

The RLVR-trained model generates a large set of solutions; correct ones
(filtered by the verifier) plus a writing-quality filter form a 600K-
example reasoning SFT set. Combined with 200K non-reasoning examples
(general chat, safety, etc.), this produces a 800K mixed set used for a
second SFT pass `[deepseek-r1 §2.3.3]`. This stage is the bridge
between "specialist reasoning model" (Stage 2) and "general assistant
that also reasons" (Stage 4).

### Stage 4: General-domain RL

A second RL phase using both verifiable rewards (for reasoning) and
preference rewards (for helpfulness, harmlessness, format) on a
broader prompt distribution `[deepseek-r1 §2.3.4]`. This stage produces
the released DeepSeek-R1 model.

### Stage 5 (separate track): distillation

The Stage-2 / Stage-3 outputs serve as an SFT-only training set for
smaller dense models (Qwen2.5 / Llama 3 in the 1.5B–32B range). The
distilled models retain o1-mini-class reasoning capability without
requiring RL on the small model, demonstrating that **RL is needed
once at the teacher scale; the resulting traces serve as supervised
data for student scales** `[deepseek-r1 §3.2]`.

## 3. Variants and lineage

### 3.1 GRPO-family variants

| Method | Year | Change vs GRPO | Reported gain |
|---|---|---|---|
| GRPO (R1) | 2025 | Baseline: group-mean advantage, symmetric clip | AIME 71% Qwen2.5-Math-7B-base after stage 2 `[deepseek-r1 §2.2.2]` |
| DAPO | 2025 | Asymmetric clip (`clip-higher`); dynamic sampling (drop groups where all $r_i$ equal); token-level loss | AIME 50pts on Qwen2.5-32B in 50% of R1-Zero steps `[dapo2025 §3]` |
| Dr.GRPO | 2025 | Removes length normalisation; argues it biases toward short reasoning | Reported gains on MATH `[forum-signal: Lambert/interconnects]` |
| DRA-GRPO | 2025 | Diversity-regularised: penalty for collapsed reasoning paths | Addresses mode collapse `[arXiv 2505.09655]` |
| RLOO / REINFORCE++ | 2024-25 | Leave-one-out variants; do not rely on group-mean baseline | Comparable to GRPO at scale `[various, see arXiv 2505.00551]` |

[CONTRADICTION] Length normalisation (per-token loss vs per-sequence
loss) is a contested choice. R1's group-mean GRPO normalises by
sequence length implicitly through the inner $1/|y_i|$; DAPO uses
token-level loss instead, arguing that long-reasoning samples are
under-weighted by sequence-level loss `[dapo2025 §3.3]`. Dr.GRPO argues
the opposite — that the inner $1/|y_i|$ biases toward short reasoning.
Empirical results vary across model scales; no consensus as of 2026.

### 3.2 The capability-ceiling debate

A landmark 2025 result challenges the dominant "RL teaches reasoning"
narrative: `[rlvr-limits-2025 §3]` shows that on well-controlled
benchmarks, RLVR improves $\mathrm{pass}@1$ on problems the base
model can already solve at $\mathrm{pass}@k$ for some moderate $k$,
but does **not expand the set of problems solvable at any $k$**. The
capability ceiling is set at pre-training; RL is reweighting, not
adding `[rlvr-limits-2025 §4]`.

[CONTRADICTION] DeepSeek-R1's headline AIME jump $15.6\% \to 71\%$ is
hard to square with a strict "no new capabilities" reading. Two
reconciliations from the 2026 literature:

1. AIME gains are dominated by improvements at $\mathrm{pass}@1$ on
   problems where the base model had moderate $\mathrm{pass}@k$. Both
   papers can be right because the base $\mathrm{pass}@1$ floor is
   very low and the ceiling at $\mathrm{pass}@k$ for moderate $k$ is
   already close to 71% `[rlvr-limits-2025 §5 discussion]`.
2. The reasoning-trace-length growth from 200 to 10K tokens is itself
   a behavioural change, even if the latent solvability set is
   unchanged. RL trains the model to *use* TTC, even if the underlying
   problem-solving primitives were already there.

This is the load-bearing open question of 2026 reasoning training: is
RLVR teaching new computation, or just learning to invoke pretrained
computation under the verifier signal?

### 3.3 Search-based training (rStar-Math)

An alternative track avoids RL entirely. rStar-Math
`[rstar-math2025 §3]` runs a four-cycle self-evolution:

1. MCTS with a process reward model (PRM) generates correct solution
   trees for a curated set of math problems.
2. The policy model (Qwen2.5-Math-7B) is SFT-trained on the MCTS-
   discovered correct trajectories.
3. The PRM is updated using the new policy's distribution.
4. Repeat for four rounds.

Result: Qwen2.5-Math-7B reaches 90% on MATH and 53.3% on AIME 2024,
rivalling o1-mini at 7B scale `[rstar-math2025 §4]`. The contrast with
R1 is sharp:

- **R1**: outcome-reward RL, no PRM, scale-of-rollouts driven.
- **rStar-Math**: process-reward search + SFT, no policy gradient,
  search-quality driven.

Both produce similar final capability at 7B; whether one is more
sample-efficient depends on benchmark `[rstar-math2025 §5 ablation]`.

### 3.4 Distillation as a training method

A surprising R1 finding is that **distillation outperforms RL on small
models**: SFT on R1-generated traces of a $\le 32$B base produces
better reasoning than running RL on that same base directly
`[deepseek-r1 §3.2.2]`. The implication: at small scales, the
bottleneck is data (long-CoT exemplars), not algorithm; once a teacher
exists, distillation is the cheap route. RL is needed once at teacher
scale to bootstrap the trace-generating distribution.

[INTUITION] This pattern matches the broader picture of `weak-to-
strong` (`burns2023-w2s`): once high-quality supervision exists, SFT is
remarkably efficient. RL's role is producing that supervision when no
prior exemplar set is available.

### 3.5 Hybrid think / non-think models

Qwen3 and DeepSeek-V3.1+ ship as **single models** that toggle between
thinking and non-thinking modes via system-prompt instruction
`[qwen3 §4]`. The training implication: post-training mixes long-CoT
data and short-answer data in known proportions, with mode-tagged
labels in SFT. This is the productisation step of reasoning training:
the user no longer chooses between "the chat model" and "the reasoning
model".

## 4. Frontier and open questions (as of 2026-05)

- **What does RLVR actually teach?** [CONTRADICTION] The
  capability-ceiling result `[rlvr-limits-2025]` and the R1 trace-
  growth result `[deepseek-r1 §2.2.2]` are not yet reconciled. The
  cleanest reading is that RLVR teaches *trace-length use* (when to
  spend more tokens) without teaching new primitives, but the evidence
  is not yet decisive `[rlvr-limits-2025 §5]`.
- **PRM vs ORM in the training loop.** R1 uses outcome rewards; rStar-
  Math uses process rewards via search. Whether mixing PRM signal into
  GRPO improves over pure outcome rewards is contested
  (`thinkprm2025 §4` shows PRMs can serve as test-time verifiers; their
  role *inside* GRPO is less established).
- **Reward hacking under RLVR.** With only outcome rewards, models can
  hack the verifier (find shortcut answers, exploit regex). R1 reports
  several mitigation steps but treats this as ongoing
  `[deepseek-r1 §2.2.4]`. The 2026 literature catalogues hacking modes
  but offers no unified defence.
- **Generalisation beyond verifiable domains.** RLVR is, by definition,
  bounded to domains with cheap mechanical verifiers (math, code,
  formal logic, some scientific QA). Extending to creative writing,
  scientific synthesis, or open-ended planning currently requires
  learned reward models — bringing back the distribution-shift
  problems RLVR was designed to avoid.
- **Distillation vs on-policy training in the student.** Self-
  distilled reasoner `[arXiv 2601.18734]` proposes that on-policy
  student-sampled trajectories (with teacher supervision) outperform
  vanilla teacher-trace SFT. As of 2026 this is a fresh thread,
  potentially the next step in the distillation lineage.
- **Diversity collapse.** Long-trace RL has a known mode-collapse
  failure (model converges on a single reasoning style)
  `[arXiv 2505.09655]`. Diversity-regularised variants address this
  but at compute cost.

## 5. Intuitions and analogies

[ANALOGY] GRPO is **REINFORCE with a group baseline**. The policy
gradient theorem allows any baseline that is independent of the
sampled action; the group's mean reward is such a baseline. The
analogy returns to canonical form via Eq. (2): $A_{i,t}$ is an
unbiased advantage estimator under that baseline, with variance
reduced by removing the group's average.

[INTUITION] Why GRPO works without a critic: PPO needs a value function
because the trajectory is long and reward is delayed. For reasoning
problems with sparse outcome reward, a value function is hard to
train; the group-mean baseline is a cheap, low-variance substitute
that does not need its own neural network. The cost: GRPO needs $G$
samples per prompt where PPO needs 1, so total compute per gradient
step is higher by a factor of $G$.

[INTUITION] R1's trace-length growth is a **policy-improvement
fixed point**: the rule-based verifier rewards correctness, and on
hard problems longer traces achieve higher correctness, so policy
gradient pushes toward longer traces. The 200 → 10 000 trajectory is
not magic; it is the policy finding the local maximum in length-vs-
correctness under its current capabilities `[deepseek-r1 §2.2.2]`.

[ANALOGY] The R1-Zero / R1 split — RL from base vs RL from SFT
initialisation — mirrors the **AlphaGo Zero / AlphaGo split**: the
all-RL version reaches comparable capability but at the cost of
readability / speed. The analogy returns to canonical form as the
question of whether a small-but-curated supervised initialisation
substantially shortens the RL horizon. R1 answers yes for readability,
without changing the asymptote `[deepseek-r1 §2.2.4 ablation]`.

## 6. See also

- `kb/notes/reasoning/test-time-compute.md` — the inference-side use
  of trained long traces.
- `kb/notes/reasoning/chain-of-thought.md` — the prompting-era
  precursor; faithfulness considerations.
- `kb/notes/reasoning/process-supervision.md` — PRMs, both as test-time
  verifiers and as candidate training signals.
- `kb/notes/reasoning/inference-time-search.md` — rStar-Math's MCTS-
  based alternative training track.
- `kb/notes/post-training/rlvr-and-grpo.md` — GRPO as a general method;
  this note specialises to the reasoning application.
- `kb/notes/architecture/reasoning-architectures.md` — architectural
  features (long-context attention, KV cache) that make long-trace
  decoding affordable.
