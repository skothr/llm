---
topic: post-training/rlhf
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - ouyang2022-instructgpt  # canonical 3-stage pipeline
  - christiano2017-pref-rl  # preference RL foundation
  - stiennon2020-summarize  # the LM-RLHF predecessor
  - rafailov2023-dpo  # included for the offline-vs-online comparison
secondary_sources:
  - schulman2017-ppo  # the underlying RL algorithm
  - bai2022-hh  # Anthropic HH-RLHF (helpful + harmless)
  - meta-llama3  # DPO replaces PPO at Llama-3 scale
related_topics:
  - post-training/sft
  - post-training/dpo-and-offline
  - post-training/rlaif-and-constitutional
  - post-training/rlvr-and-grpo
  - alignment/safety-evaluation
---

# RLHF — Reinforcement Learning from Human Feedback

**RLHF** is the canonical three-stage post-training pipeline that
converts a base LM into an instruction-following assistant: SFT →
reward-model training → PPO against the reward model with a KL anchor
to the SFT initialization
`[ouyang2022-instructgpt §3; kb/excerpts/ouyang2022-instructgpt#sec-3]`.
This note covers the canonical pipeline, the underlying objective, and
how the field has evolved (DPO replacing PPO at Llama-3 scale; GRPO
replacing PPO for verifiable-reward training; constitutional / RLAIF
variants for reward labeling).

The lineage: Christiano et al. 2017 (preference RL on Atari /
locomotion) `[christiano2017-pref-rl]` → Stiennon et al. 2020
(summarization with human preferences) `[stiennon2020-summarize]` →
**InstructGPT (Ouyang et al. 2022)** `[ouyang2022-instructgpt]` — the
last is the canonical reference.

## 1. Formal definition

### 1.1 The KL-constrained reward maximization objective

Let $\mathcal{D}$ be a prompt distribution, $\pi^{\mathrm{SFT}}$ the
supervised fine-tuned policy from stage 1, $r_\phi(x,y)$ the learned
reward model from stage 2. Stage 3 maximizes:

$$
\mathcal{J}_{\mathrm{RLHF}}(\theta) = \mathbb{E}_{x\sim\mathcal{D},\, y\sim\pi_\theta(\cdot|x)} \big[ r_\phi(x,y) \big] - \beta\, \mathbb{E}_{x,y}\!\left[\log\frac{\pi_\theta(y|x)}{\pi^{\mathrm{SFT}}(y|x)}\right] \tag{1}
$$

`[ouyang2022-instructgpt §3.5; kb/excerpts/ouyang2022-instructgpt#sec-3]`.

Symbols:

| Symbol | Meaning |
|---|---|
| $\pi_\theta$ | the policy being trained (LM with parameters $\theta$) |
| $\pi^{\mathrm{SFT}}$ | the SFT-stage initialization, frozen during RL |
| $r_\phi(x,y)$ | learned scalar reward model, frozen during RL |
| $\beta$ | KL coefficient (typical: 0.01–0.05 in InstructGPT-era runs) |
| $x, y$ | prompt, response |

InstructGPT also adds a pre-training-mixing term ("PPO-ptx"):

$$
\mathcal{J}_{\mathrm{RLHF}}^{\mathrm{ptx}}(\theta) = \mathcal{J}_{\mathrm{RLHF}}(\theta) + \gamma\, \mathbb{E}_{x \sim \mathcal{D}_{\mathrm{pretrain}}}\!\left[\log\pi_\theta(x)\right] \tag{2}
$$

The third term reduces the **alignment tax** (regression on standard
NLP benchmarks after RLHF)
`[ouyang2022-instructgpt §3.5; kb/excerpts/ouyang2022-instructgpt#sec-3]`.

### 1.2 The reward model objective (Bradley–Terry)

The reward model is trained on **pairwise preference comparisons**
$(x, y_w, y_l)$ where $y_w$ is preferred over $y_l$ by a human labeler.
Under the Bradley–Terry assumption $p(y_w \succ y_l|x) = \sigma(r(x,y_w)
- r(x,y_l))$, maximum likelihood gives:

$$
\mathcal{L}_{\mathrm{RM}}(\phi) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\mathrm{RM}}}\big[\log\sigma(r_\phi(x,y_w) - r_\phi(x,y_l))\big] \tag{3}
$$

`[ouyang2022-instructgpt §3.5; kb/excerpts/ouyang2022-instructgpt#sec-3]`.
$r_\phi$ is typically initialized from $\pi^{\mathrm{SFT}}$ with the LM
head replaced by a scalar head (one linear layer over the final hidden
state of the last token).

InstructGPT samples $K \in \{4, 9\}$ responses per prompt and trains on
all $\binom{K}{2}$ pairs with a per-prompt correlation correction. This
is more sample-efficient than independent pairwise comparisons because
the same $K$ responses are scored together.

### 1.3 The closed-form RL optimum (foundation for DPO)

The maximum of Eq. (1) over $\pi_\theta$ has a known closed form
`[rafailov2023-dpo §4; kb/excerpts/rafailov2023-dpo#sec-4]`:

$$
\pi^*(y|x) = \frac{1}{Z(x)}\, \pi^{\mathrm{SFT}}(y|x)\, \exp\!\left(\frac{1}{\beta} r_\phi(x,y)\right). \tag{4}
$$

This is a re-weighting of the SFT distribution by an exponential of
the reward. **Policy gradient methods (PPO, REINFORCE) approximate this
optimum iteratively; DPO inverts Eq. (4) to derive a closed-form loss
that bypasses RL altogether** (see
`kb/notes/post-training/dpo-and-offline.md`).

## 2. Mechanism — the three-stage protocol

The InstructGPT pipeline `[ouyang2022-instructgpt §3.5;
kb/excerpts/ouyang2022-instructgpt#sec-3]`:

### Stage 1 — SFT

1. Collect human-written demonstrations $\{(x, y)\}$ on prompts drawn
   from a target distribution (OpenAI API in InstructGPT's case).
2. Fine-tune base GPT-3 with cross-entropy maximum-likelihood:

$$
\mathcal{L}_{\mathrm{SFT}}(\pi_\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{SFT}}}[\log\pi_\theta(y|x)].
$$

Result: $\pi^{\mathrm{SFT}}$. Typical scale in InstructGPT: ~13K
labeler-written prompt+demonstration pairs.

### Stage 2 — Reward Model

1. Sample $K$ responses from $\pi^{\mathrm{SFT}}$ per prompt.
2. Have human labelers rank them.
3. Train $r_\phi$ on Eq. (3) using all $\binom{K}{2}$ pairs.

Typical scale: ~33K prompts, $K \in \{4,9\}$, ~50K to ~200K pairs.

### Stage 3 — PPO

The implementation detail: the KL penalty is **applied per-token in the
reward** rather than as an explicit constraint:

$$
r_{\mathrm{shaped}}(x, y_t) = r_\phi(x,y) \cdot \mathbb{1}[t = T] - \beta \log\frac{\pi_\theta(y_t|x,y_{<t})}{\pi^{\mathrm{SFT}}(y_t|x,y_{<t})}. \tag{5}
$$

This makes the RL problem a token-level MDP with reward sparse at the
final token plus a per-token KL anchor. PPO's GAE-based advantage
estimation then handles credit assignment along the trajectory
`[schulman2017-ppo §5]`.

Typical scale in InstructGPT-1.3B: ~31K prompts, ~256K rollouts, KL
coefficient $\beta \in [0.01, 0.05]$, learning rate ~$10^{-6}$.

### Stage 4 (modern addendum) — rejection sampling and iteration

Modern RLHF runs iterate over multiple rounds: after stage 3, sample
new responses from $\pi_\theta$, have humans label them, retrain
$r_\phi$, run PPO again. Llama-3 `[meta-llama3]` makes this iterative,
adding **rejection sampling** (sample $N$ from current model, keep
top-$k$ by reward, SFT on those) as a quality gate before each new
preference round.

## 3. Variants and lineage

### 3.1 PPO variants and the infrastructure problem

**TRL, DeepSpeed-Chat, OpenRLHF, verl** — these are the open-source
implementations. The PPO-RLHF training run requires:

- **4 model copies** in memory: policy ($\pi_\theta$), reference
  ($\pi^{\mathrm{SFT}}$), reward ($r_\phi$), and value ($V_\phi$).
- **Sampling**, **scoring**, and **policy optimization** as three
  separate phases per step, often on different worker pools.
- **KL spikes and reward-hacking** as the policy drifts from
  $\pi^{\mathrm{SFT}}$ — requires careful $\beta$ scheduling.

This complexity is the primary reason DPO and GRPO have largely
displaced PPO at the open-source frontier (see §3.4).

### 3.2 Reward model variants

| Variant | Idea | Source |
|---|---|---|
| **Single RM** | One reward model trained on Bradley–Terry | InstructGPT, default |
| **Ensemble RMs** | Multiple RMs with different seeds; reduce variance, mitigate reward hacking | Coste et al. 2023 |
| **WARM (Weight-Averaged Reward Models)** | Average parameters of $K$ independently-trained RMs | Ramé et al. 2024; Gemma 3 |
| **WARP** | WARM + alternation of training and weight-averaging during RL | Ramé et al. 2024; Gemma 3 |
| **RLAIF (Constitutional)** | Replace human labels with AI labels conditioned on a constitution | `bai2022-cai` |
| **RLVR** | Replace learned RM with deterministic verifier | `deepseek-r1` |

The trend is **away from a single learned RM**: WARM/WARP introduce
ensembling for robustness; RLAIF replaces humans with AI; RLVR replaces
the learned RM entirely with a programmatic check.

### 3.3 Stiennon 2020 → InstructGPT lineage

Stiennon et al. 2020 (summarization) is the direct technical predecessor
to InstructGPT. They establish:

- The Bradley–Terry reward-model loss form (Eq. 3).
- The PPO objective with per-token KL anchor (Eq. 1, Eq. 5).
- The empirical observation that RLHF outperforms supervised
  fine-tuning **for tasks where human preference is more reliable than
  human production** — humans can rank summaries better than they can
  write summaries.

InstructGPT `[ouyang2022-instructgpt]` extends this to the open-domain
instruction-following setting, with the headline result that **a 1.3B
RLHF'd model beats 175B GPT-3** on user-prompt preference rate
`[ouyang2022-instructgpt §4; kb/excerpts/ouyang2022-instructgpt#sec-4]`.

### 3.4 Why DPO replaced PPO at Llama-3 scale

Llama-3's post-training `[meta-llama3]`:

- Uses SFT → rejection sampling → DPO. **No PPO at scale.**
- Reports DPO required less compute than PPO and performed as well or
  better at large scale.
- Iterates the SFT → DPO cycle multiple rounds with on-policy
  preference data.

The empirical narrative shift around 2024: **for general assistant
alignment, DPO is the new default. PPO retains a niche for
verifiable-reward reasoning training (where it has been replaced by
GRPO).** See `kb/notes/post-training/dpo-and-offline.md` for the offline
side and `kb/notes/post-training/rlvr-and-grpo.md` for the verifiable-
reward side.

## 4. Intuitions and analogies

[ANALOGY] **The reward model is a learned distillation of human
judgment.** Humans rank pairs faster and more reliably than they can
score in absolute terms; the Bradley–Terry model is the simplest
calibration of pairwise rankings into a real-valued utility. The
canonical math returns to Eq. (3): the reward is *defined* as the
parameter that fits the observed pair preferences best under the BT
assumption.

[INTUITION] **The KL term is a trust region, not a regularizer.** Eq.
(1)'s KL penalty looks like an L2-style regularizer on the reward
maximization, but its effect is more like trust-region constraint:
the policy can only move in directions $\pi^{\mathrm{SFT}}$ has support
for. Without it, $\pi_\theta$ collapses to argmax-reward and produces
degenerate text (mode collapse, reward hacking). With it, the
optimization is constrained to a shell around $\pi^{\mathrm{SFT}}$
proportional to $\beta$.

[INTUITION] **Why preferences and not absolute scores.** Asking humans
"on a scale of 1–10, how good is this response?" is famously unreliable
(scale drift, anchoring effects, individual differences). Asking
"which of these two is better?" is much more reliable but only gives
ordinal information. RLHF's design choice: train on the reliable signal
(preferences) and let the Bradley–Terry calibration recover the latent
real-valued reward.

[ANALOGY] **InstructGPT vs Constitutional AI.** Both have the same
RL machinery (PPO with KL anchor, learned reward model). The difference
is the source of preference labels: humans (InstructGPT) vs AI under a
constitution (CAI). RLAIF is "InstructGPT with a different labeler."
See `kb/notes/post-training/rlaif-and-constitutional.md`.

## 5. Frontier and open questions (as of 2026-05)

### 5.1 Reward hacking and overoptimization

`[CONTRADICTION]` The community currently disagrees about whether
RLHF's reward-hacking problem is fundamentally tractable or
fundamentally a property of optimizing learned proxy rewards:

- **Tractable view** (Coste et al. 2023; Ramé et al. 2024 / WARM/WARP;
  most production teams): better RM training, ensembling, KL scheduling,
  and on-policy preference iteration mitigate it. Demonstrated in Gemma
  3's deployment.
- **Fundamental view** (Phase 1 sweep §rlhf, citing arXiv:2604.13602
  April 2026; arXiv:2406.02900 scaling laws for direct alignment):
  overoptimization persists across RLHF, DPO, and inference-time
  methods; verbosity, sycophancy, and math-hallucination are localized
  shortcuts that emerge at all scales.

Sycophancy specifically is well-documented as an RLHF failure mode
(Sharma et al. 2023; Phase 1 sweep landscape report on rlhf): RM
inherits annotator confirmation bias, policy amplifies it. Mitigation
(Reward Shaping / PAR, arXiv:2502.18770) is partial.

### 5.2 The end of "pure" RLHF

The InstructGPT pipeline as originally specified (single RM + PPO + KL)
is **rarely used in production in 2025-2026**. The replacements are:

- DPO and SimPO at large scale for general alignment.
- GRPO/DAPO/VAPO for verifiable-reward reasoning training.
- WARM/WARP-augmented RM ensembles for the cases where PPO is still
  used (Gemma 3).
- Multi-stage pipelines that combine SFT, DPO, and RLVR (Tülu 3, Qwen
  3, OLMo 3).

The term "RLHF" has accordingly drifted: in 2022 it meant the
InstructGPT pipeline specifically, in 2026 it often means "any
preference-based post-training," including DPO. The Phase 1 sweep
recommended renaming this topic to `reward-modeling-and-rlhf` to
disambiguate.

### 5.3 RewardBench 2 and reward-model evaluation

RewardBench (Lambert et al. 2024) and RewardBench 2 (2025,
arXiv:2506.01937) are the current standards for reward-model
evaluation. **Open question:** does RewardBench score correlate with
downstream RLHF outcome? Phase 1 sweep §rlhf flags this as not yet
systematically established — RM benchmarks measure RM accuracy, but
the policy's behavior depends on RM *gradient signal*, which is a
different quantity.

### 5.4 Calibration collapse under sycophancy fine-tuning

`[FORUM-SIGNAL]` arXiv:2604.10585 (April 2026) reports that sycophancy
fine-tuning breaks calibration / uncertainty quantification — models
fine-tuned to be more agreeable lose ability to express uncertainty.
This suggests the costs of RLHF-induced sycophancy include a hidden
calibration cost, not just user-facing agreeableness.

### 5.5 RLHF at frontier-lab scale

Frontier labs (OpenAI, Anthropic, DeepMind) do not publicly disclose
their full post-training recipes. Inferred from model behavior, model
cards, and adjacent publications:

- **Anthropic** uses Constitutional AI variants (cf. Bai 2022) plus
  human feedback; details proprietary.
- **OpenAI** (post-o1) appears to use RLVR-class methods for reasoning
  models, RLHF-class for general models.
- **DeepMind / Gemma 3** uses BOND/WARM/WARP with multi-reward (math,
  code, safety, multilingual). `[gemma3]`.
- **Llama 4** post-training details mostly undisclosed.

## 6. See also

- `kb/notes/post-training/sft.md` — stage 1 of the RLHF pipeline.
- `kb/notes/post-training/dpo-and-offline.md` — the offline replacement
  for PPO; closed-form derivative of Eq. (4).
- `kb/notes/post-training/rlaif-and-constitutional.md` — replacing
  human preference labels with AI labels under a constitution.
- `kb/notes/post-training/rlvr-and-grpo.md` — replacing the learned RM
  with a deterministic verifier; the frontier of post-training in 2025-26.
- `kb/notes/alignment/sycophancy.md` — RLHF as a cause of sycophancy.
- `kb/notes/alignment/safety-evaluation.md` — RewardBench 2, HarmBench,
  and how RLHF interacts with safety evals.
