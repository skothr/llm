---
topic: post-training/dpo-and-offline
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - rafailov2023-dpo
  - meng2024-simpo
  - ouyang2022-instructgpt  # for the RLHF equation DPO derives from
secondary_sources:
  - azar2023-ipo  # planned add
  - ethayarajh2024-kto
  - hong2024-orpo
  - meta-llama3  # DPO at scale
  - tulu3
related_topics:
  - post-training/rlhf
  - post-training/sft
  - post-training/rlaif-and-constitutional
  - post-training/rlvr-and-grpo
---

# DPO and offline preference optimization

**DPO** (Direct Preference Optimization) is the closed-form, classification-loss
reformulation of the RLHF objective: the optimal policy is extracted directly
from preference comparisons, with **no reward model and no on-policy
sampling**. This single technical move turned post-training from a
PPO-class engineering problem into a fine-tuning-class engineering
problem `[rafailov2023-dpo §4; kb/excerpts/rafailov2023-dpo#sec-4]`.

This note covers the DPO derivation and the variant family that has
grown around it (IPO, KTO, ORPO, SimPO, anchored variants). They share
a common pattern: a closed-form loss on policy log-ratios, no separate
RM, no PPO loop.

## 1. Formal definition

### 1.1 The DPO loss (canonical form)

The DPO loss for a preference dataset $\mathcal{D} = \{(x, y_w, y_l)\}$
where $y_w$ is preferred over $y_l$:

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}}) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right] \tag{1}
$$

`[rafailov2023-dpo §4 Eq.7; kb/excerpts/rafailov2023-dpo#sec-4-loss]`.

Symbols:

| Symbol | Meaning |
|---|---|
| $\pi_\theta$ | the policy being trained |
| $\pi_{\mathrm{ref}}$ | the frozen reference (typically the SFT-stage model) |
| $\beta$ | inverse-temperature; how aggressively to enforce preferences (typical: 0.1–0.5) |
| $y_w, y_l$ | preferred and rejected responses |
| $\sigma$ | sigmoid function |

The implicit reward is

$$
\hat{r}_\theta(x,y) = \beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}. \tag{2}
$$

Eq. (1) is the Bradley–Terry log-likelihood evaluated on $\hat{r}_\theta$.
There is **no separate reward model** — the policy log-ratio is the
reward.

### 1.2 The reparameterization derivation

DPO's central insight is to **invert the closed-form RL optimum**
`[rafailov2023-dpo §4 Eq.4-5; kb/excerpts/rafailov2023-dpo#sec-4]`. The
KL-constrained reward maximization (Eq. 1 of `kb/notes/post-training/rlhf.md`)
has the optimal policy

$$
\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\mathrm{ref}}(y|x)\exp\!\left(\frac{1}{\beta}r(x,y)\right). \tag{3}
$$

Solving for $r$:

$$
r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta\log Z(x). \tag{4}
$$

The Bradley–Terry preference $p(y_w \succ y_l|x) = \sigma(r(x,y_w) -
r(x,y_l))$ has $\beta\log Z(x)$ cancel because both terms share it,
yielding a preference probability that depends only on policy
log-ratios. Maximum-likelihood on observed pairs gives Eq. (1).

**The crucial structural fact:** if $\pi_\theta$ approaches the
maximum-likelihood solution of Eq. (1), it approaches the closed-form
RL optimum in Eq. (3). DPO and PPO-RLHF have the **same fixed point**
when run to convergence on the same preference data — but DPO reaches
it without sampling from $\pi_\theta$ during training, without a reward
model, and without an RL loop.

### 1.3 The DPO gradient

`[rafailov2023-dpo §4-gradient; kb/excerpts/rafailov2023-dpo#sec-4-gradient]`:

$$
\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta\, \mathbb{E}\!\left[ \sigma(\hat{r}_\theta(x,y_l) - \hat{r}_\theta(x,y_w))\, \big(\nabla_\theta\log\pi_\theta(y_w|x) - \nabla_\theta\log\pi_\theta(y_l|x)\big) \right].
$$

The sigmoid factor is large when the preference is currently violated
(model assigns lower implicit reward to $y_w$); the gradient pushes
$y_w$'s log-prob up and $y_l$'s log-prob down. This is the
"preference-margin" gradient that defines all DPO-class methods.

## 2. Mechanism — the protocol

The training pipeline:

1. **Start from $\pi^{\mathrm{SFT}}$** (output of the SFT stage of the
   pipeline; cf. `kb/notes/post-training/sft.md`). Set
   $\pi_{\mathrm{ref}} \leftarrow \pi^{\mathrm{SFT}}$.
2. **Initialize $\pi_\theta \leftarrow \pi^{\mathrm{SFT}}$.**
3. **Loop over preference batches** $(x, y_w, y_l)$:
   1. Compute $\log\pi_\theta(y_w|x), \log\pi_\theta(y_l|x)$ via a
      forward pass on $\pi_\theta$.
   2. Compute $\log\pi_{\mathrm{ref}}(y_w|x),
      \log\pi_{\mathrm{ref}}(y_l|x)$ via a forward pass on
      $\pi_{\mathrm{ref}}$ (cached or computed each step).
   3. Compute the DPO loss in Eq. (1) and back-propagate to $\theta$.
4. **No sampling, no RL loop, no reward model.** Just supervised
   classification on log-ratios.

The total compute per step is **~2 forward passes + 1 backward pass
on $\pi_\theta$** vs. PPO's ~4 forward + 1 backward + reward-model
forward + sampling. DPO is dramatically cheaper per step and
dramatically simpler to debug.

### 2.1 Practical hyperparameters

The Tülu 3 recipe `[tulu3]` (canonical open reference):

- $\beta = 0.1$ (Llama-3 also uses $\beta = 0.1$ `[meta-llama3]`)
- LR: $5\times 10^{-7}$ (notice: ~10× smaller than SFT LR)
- 1–3 epochs over preference data
- Batch size: 32–256 (effective)

The low LR is critical: DPO is more stable than PPO but can still
collapse if $\pi_\theta$ drifts too far from $\pi_{\mathrm{ref}}$.

### 2.2 On-policy vs off-policy preferences

DPO is **off-policy** by default — it can be run on preferences
collected from *any* model, not just $\pi^{\mathrm{SFT}}$. But the
modern best practice (Llama-3, Tülu 3) is to **iterate**:

1. Run DPO on initial preferences.
2. Sample new responses from the resulting model.
3. Have a labeler (human or LLM-judge) compare them.
4. Run DPO again with $\pi_{\mathrm{ref}}$ now set to the new
   $\pi_\theta$ (or kept at $\pi^{\mathrm{SFT}}$ — both schemes used).

This iteration is **on-policy preference collection with off-policy
optimization** — recovers some of the on-policy benefit of PPO without
the per-step sampling cost.

## 3. Variants and lineage

The DPO variant space is large; this section covers the most-cited.

### 3.1 Comparison table

| Method | Reference | Reward form | KL anchor | Length-normalize | Source |
|---|---|---|---|---|---|
| **PPO-RLHF** (online) | $\pi^{\mathrm{SFT}}$ | learned $r_\phi$ | per-token in reward | no | `[ouyang2022-instructgpt]` |
| **DPO** | $\pi_{\mathrm{ref}}$ frozen | implicit, $\beta\log\pi_\theta/\pi_{\mathrm{ref}}$ | implicit (KL emerges) | no | `[rafailov2023-dpo]` |
| **IPO** | $\pi_{\mathrm{ref}}$ frozen | implicit, ψ-loss instead of log-sigmoid | implicit | no | `[azar2023-ipo]` |
| **KTO** | $\pi_{\mathrm{ref}}$ frozen | prospect-theory utility on log-ratios | implicit | no | `[ethayarajh2024-kto]` |
| **ORPO** | none | log-odds ratio of SFT loss | none (combined SFT+pref) | no | `[hong2024-orpo]` |
| **SimPO** | none | $\beta/|y| \cdot \log\pi_\theta(y|x)$ + margin $\gamma$ | none | yes | `[meng2024-simpo]` |
| **CPO** | none | hard preference loss + SFT regularizer | none | no | Xu et al. 2024 |

`[rafailov2023-dpo §4; meng2024-simpo §3; ouyang2022-instructgpt §3]`

### 3.2 IPO — fixing DPO's overconfidence trap

DPO's log-sigmoid loss has a known pathology: when one preference is
clearly satisfied, the implicit reward gap can grow without bound while
the loss becomes vanishingly small. This drives the policy to overfit
to a few samples with extreme log-ratios and undertrains on the bulk.

IPO `[azar2023-ipo]` replaces the log-sigmoid with a squared loss on
the gap:

$$
\mathcal{L}_{\mathrm{IPO}}(\theta) = \mathbb{E}\!\left[\big(\hat{r}_\theta(x,y_w) - \hat{r}_\theta(x,y_l) - \frac{1}{2\tau}\big)^2\right],
$$

where $\tau$ controls the target margin. The squared loss is bounded
above and saturates symmetrically, preventing runaway log-ratio growth.

### 3.3 KTO — single-rating data, prospect-theoretic loss

KTO `[ethayarajh2024-kto]` (Kahneman-Tversky Optimization) operates on
**single-rating data** $(x, y, \mathrm{label})$ with $\mathrm{label}
\in \{\text{good}, \text{bad}\}$, instead of pairwise preferences. The
loss is asymmetric (good and bad outcomes weighted differently),
inspired by prospect theory's loss-aversion. Useful when only
yes/no labels are available, not pairwise comparisons.

### 3.4 ORPO — combining SFT and preferences in one loss

ORPO `[hong2024-orpo]` combines SFT and preference losses into a single
objective that trains from base directly (no SFT stage):

$$
\mathcal{L}_{\mathrm{ORPO}} = \mathcal{L}_{\mathrm{SFT}}(y_w) + \lambda\, \mathcal{L}_{\mathrm{OR}}(y_w, y_l),
$$

where $\mathcal{L}_{\mathrm{OR}}$ is a log-odds ratio on the SFT loss
between chosen and rejected responses. **No reference model needed** —
the SFT loss replaces the KL anchor's role. Reduces compute (no second
forward pass) and removes a stage from the pipeline.

### 3.5 SimPO — reference-free, length-normalized

SimPO `[meng2024-simpo §3; kb/excerpts/meng2024-simpo#sec-3]`:

$$
\mathcal{L}_{\mathrm{SimPO}}(\theta) = -\mathbb{E}\!\left[\log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right]. \tag{5}
$$

Two changes vs. DPO:

1. **No reference model** — the implicit reward is just the
   length-normalized average log-prob.
2. **Target margin $\gamma$** — the policy must beat the rejected
   response by at least $\gamma$ in implicit reward.

The length-normalization is the key practical contribution: it directly
addresses DPO's well-known **length bias** (DPO tends to make outputs
longer because long responses are easier to satisfy with higher
log-ratios). On length-controlled benchmarks SimPO outperforms DPO
substantially `[meng2024-simpo §4-headline; kb/excerpts/meng2024-simpo#sec-4-headline]`.

### 3.6 GAPO and explicit-PO — anchoring variants

Recent (2026) variants `[gapo2026 arXiv:2602.04909; explicit-po-2025
arXiv:2506.07492]` introduce additional **anchoring** terms to stabilize
the optimization in the high-margin regime. GAPO uses a pessimistic
local surrogate; Explicit PO targets specific limitations of the
implicit-reward DPO formulation.

### 3.7 Anchored vs. reference-free

The variant landscape splits along one axis: **does the loss depend on
$\pi_{\mathrm{ref}}$?**

- **Reference-anchored:** DPO, IPO, KTO, GAPO. Loss depends on
  $\pi_{\mathrm{ref}}(y|x)$. Behavior is "stay close to ref, but prefer
  $y_w$."
- **Reference-free:** ORPO, SimPO, CPO. Loss depends only on
  $\pi_\theta$. Behavior is "rank $y_w > y_l$ in absolute log-prob."

Phase 1 sweep recommended splitting this topic along this axis. For
this note we keep them together, table-organized.

## 4. Intuitions and analogies

[ANALOGY] **DPO is "the language model is the reward model."** In RLHF,
the reward model is a separate neural net trained on preferences and
then queried during PPO. DPO observes that the reparameterization Eq.
(4) makes the policy log-ratio *itself* a valid reward function;
training the policy on preferences therefore implicitly trains the
"reward model" (the log-ratio) at the same time. Returns to canonical
form: the "implicit reward" is exactly $\hat{r}_\theta = \beta\log
\pi_\theta/\pi_{\mathrm{ref}}$ in Eq. (2).

[INTUITION] **Why DPO is sometimes worse than RLHF and sometimes
better.** DPO fits a fixed dataset; PPO continually generates fresh
samples and gets RM feedback on them. **For small or low-coverage
preference data, PPO's continual sampling explores beyond the dataset,
so it can outperform DPO.** **For large iterated-preference data, DPO's
direct optimization avoids RL's variance and instability and matches
or beats PPO** — this is the regime Llama-3 and Tülu 3 operate in.

[ANALOGY] **SimPO's length-normalization is to DPO what perplexity is
to log-likelihood.** Raw log-likelihood favors short sequences (each
token contributes a negative term); perplexity normalizes by length to
compare sequences of different lengths. Similarly, DPO's $\log\pi(y|x)$
favors sequences whose token-level log-probs are easy to inflate;
SimPO's $\log\pi(y|x)/|y|$ normalizes for length so the implicit reward
is comparable across sequence lengths.

[INTUITION] **The reference model is a regularizer disguised as a
distribution.** $\pi_{\mathrm{ref}}$'s role in the DPO loss is to
prevent the policy from collapsing to argmax-preference (which would
overfit to a few extreme pairs). If the policy starts equal to
$\pi_{\mathrm{ref}}$, the loss is zero on average pairs, and gradient
flows only where the data deviates from the reference's view. The
reference is a soft inductive bias toward the SFT-stage solution.

## 5. Frontier and open questions (as of 2026-05)

### 5.1 [CONTRADICTION] DPO at frontier scale: as good as PPO, or just cheaper?

Two competing narratives:

- **Llama-3 / Tülu 3 view** `[meta-llama3; tulu3]`: DPO is **better
  than PPO** at large scale, in addition to being cheaper. Open-source
  evidence supports this.
- **DeepMind / Gemma 3 view** `[gemma3]`: PPO with WARM/WARP-augmented
  RM is still in production for general alignment. Not deprecated.

The discrepancy is partly recipe-specific (different RMs, different
preference datasets) and partly task-specific (DPO more competitive on
instruction-following; PPO retains advantage on creative / open-ended
generation per anecdotal reports). No controlled head-to-head at
frontier scale has been published.

### 5.2 Length bias and verbosity

DPO has a documented **length bias** (Singhal et al. 2024; many
follow-ups): trained models become longer than the reference. SimPO's
length normalization addresses this directly. Open question: is length
bias inherent to log-likelihood-class losses (and so SimPO's fix is
the answer), or is it a property of the preference data (annotators
favor longer / more detailed responses, transferring through the
loss)?

### 5.3 The DPO survey landscape

Two surveys (arXiv:2503.11701; arXiv:2410.15595) consolidate KTO, IPO,
CPO, ORPO, SimPO, DAPO into a unified framework based on:
- the choice of reward parameterization
- presence/absence of reference model
- form of the preference loss (sigmoid vs. squared vs. ratio)
- length normalization
- KL anchoring strategy

The Phase 1 sweep notes that **no canonical winner has emerged**;
SimPO and ORPO are the most-adopted reference-free variants, DPO
remains the default reference-anchored method.

### 5.4 Reward overoptimization with DPO

`[FORUM-SIGNAL]` arXiv:2406.02900 (2024) — scaling laws for direct
alignment — shows DPO suffers from reward overoptimization analogous
to PPO-RLHF: the gap between proxy implicit reward and gold preference
grows with KL($\pi_\theta\|\pi_{\mathrm{ref}}$). This contradicts the
early DPO narrative that "no explicit reward model means no reward
hacking." The mechanism is the same as PPO-RLHF — DPO's implicit
reward $\hat{r}_\theta$ is a learned proxy and can be hacked.

### 5.5 Sycophancy and calibration collapse

Same concerns as RLHF (cf. `kb/notes/post-training/rlhf.md` §5.1):
DPO inherits annotator bias through the preference data and amplifies
sycophancy. arXiv:2604.10585 (April 2026) reports calibration collapse
under DPO-class fine-tuning on agreeableness data.

### 5.6 The relation to RLVR

DPO and RLVR are **orthogonal axes** of post-training:

- DPO replaces the **algorithm** (PPO → classification).
- RLVR replaces the **reward source** (learned RM → verifier).

In principle one could combine: classification on verifier-graded
preferences (where $y_w$ is the trajectory that the verifier accepted
and $y_l$ is one it rejected). In practice the field has gone with
GRPO for verifiable rewards and DPO for general preferences, treating
them as complementary stages of multi-step pipelines (Tülu 3, Qwen 3).

## 6. See also

- `kb/notes/post-training/rlhf.md` — the PPO-RLHF baseline DPO derives
  from, with the closed-form optimum (Eq. 4) DPO inverts.
- `kb/notes/post-training/sft.md` — DPO assumes an SFT-stage starting
  point.
- `kb/notes/post-training/rlvr-and-grpo.md` — the verifiable-reward
  alternative to learned-RM-based methods; orthogonal to DPO.
- `kb/notes/post-training/rlaif-and-constitutional.md` — replacing
  human preference labels with AI labels under a constitution; can
  feed DPO directly.
