---
topic: post-training/sft
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - ouyang2022-instructgpt
  - tulu3
  - meta-llama3
  - magpie-2024
  - deepseek-r1
secondary_sources:
  - qwen3
  - phi4
  - wei2021-flan
  - chung2022-flan-t5
related_topics:
  - post-training/rlhf
  - post-training/dpo-and-offline
  - post-training/rlvr-and-grpo
  - training/synthetic-data-and-distillation
---

# Supervised Fine-Tuning (SFT)

**SFT** is stage 1 of essentially every modern post-training pipeline:
maximum-likelihood fine-tuning of a base LM on curated prompt+response
pairs, producing the instruction-following starting point
$\pi^{\mathrm{SFT}}$ that all subsequent stages (DPO, PPO, RLVR) take as
their reference and KL anchor
`[ouyang2022-instructgpt §3; kb/excerpts/ouyang2022-instructgpt#sec-3]`.

The interesting research questions are not in the loss — it is plain
cross-entropy — but in the **data**: how to curate, synthesize, mask,
deduplicate, balance, and quality-gate the SFT set. This note covers
the canonical objective, the protocol (loss masking, multi-turn
conventions, rejection sampling), modern data-construction recipes
(Tülu 3, Llama-3 iterative SFT, MAGPIE self-synthesis, R1-distilled
reasoning SFT), and frontier questions (how much data is enough, why
does SFT before RL matter at all, when does pure-SFT distillation match
RL-trained reasoning).

## 1. Formal definition

### 1.1 The SFT loss

Given a dataset $\mathcal{D}_{\mathrm{SFT}} = \{(x^{(i)}, y^{(i)})\}_{i=1}^N$
of prompt-response pairs, SFT trains $\pi_\theta$ to minimize the
expected negative log-likelihood of the response conditional on the
prompt:

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}_{\mathrm{SFT}}}\!\left[\sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})\right] \tag{1}
$$

`[ouyang2022-instructgpt §3.5; kb/excerpts/ouyang2022-instructgpt#sec-3]`.

| Symbol | Meaning |
|---|---|
| $x$ | prompt tokens (instruction, system message, prior context) |
| $y = (y_1, \ldots, y_{|y|})$ | response tokens (assistant turn) |
| $\pi_\theta$ | the policy being trained, initialized from a base LM |
| $\mathcal{D}_{\mathrm{SFT}}$ | the SFT dataset |
| $|y|$ | response length in tokens |

The crucial difference from pre-training is **prompt masking**: the
sum in Eq. (1) is over response tokens $y_t$ only; loss on prompt
tokens $x$ is excluded by setting their weight to zero. Implementations
realize this with a per-token loss mask
$m_t \in \{0,1\}$: $\mathcal{L} = -\sum_t m_t \log\pi_\theta(z_t|z_{<t})$
where $z = (x, y)$ is the concatenated sequence and $m_t = 1$ iff
$z_t$ is a response token.

### 1.2 Relationship to pre-training

Pre-training optimizes Eq. (1)'s analogue with $m_t = 1$ for *every*
token in a long unstructured corpus. SFT differs only in:

1. **Data**: curated $(x, y)$ pairs rather than raw text.
2. **Mask**: only response tokens contribute to the loss.
3. **Scale**: $|\mathcal{D}_{\mathrm{SFT}}|$ is $10^4$–$10^6$ examples
   (Tülu 3: ~939K; InstructGPT: ~13K demonstrations
   `[ouyang2022-instructgpt §3.4; kb/excerpts/ouyang2022-instructgpt#sec-3]`)
   — orders of magnitude smaller than pre-training's $10^{12}$ tokens.
4. **Epochs**: SFT typically runs 1–4 epochs over the curated set;
   pre-training is a single pass.
5. **LR**: SFT uses a small constant or cosine-decay LR (~$10^{-5}$ to
   $5\times 10^{-6}$), an order of magnitude below pre-training peak.

Architecturally there is no change. SFT is a continued-training stage,
not a new model.

### 1.3 Multi-turn loss-mask conventions

For multi-turn conversations $(x, y^{(1)}, x^{(2)}, y^{(2)}, \ldots)$
two conventions are in use:

- **Train on all assistant turns** (Tülu 3 default `[tulu3]`):
  $m_t = 1$ for every assistant token, $m_t = 0$ for every user/system
  token. Each assistant turn contributes loss; the model learns to
  generate every assistant response in context.
- **Train on the last assistant turn only** (some legacy datasets):
  $m_t = 1$ only for the final assistant response. The model sees prior
  turns as context but doesn't directly imitate them.

The former is now standard because it gives more loss signal per
example with no obvious cost.

## 2. Mechanism — the protocol

### 2.1 Pre-SFT: dataset construction

The dominant cost is data, not compute. Modern pipelines combine four
streams:

1. **Human-written demonstrations.** InstructGPT used ~13K labeler-
   written prompt+response pairs `[ouyang2022-instructgpt §3.4;
   kb/excerpts/ouyang2022-instructgpt#sec-3]`. Expensive; bottlenecked
   by labeler throughput.
2. **Filtered organic data.** Data from existing sources (Reddit Q&A,
   StackExchange, math forums) reformatted as instruction-response.
   Tülu 3's "FLAN-style" subset is in this class `[tulu3]`.
3. **Synthetic data from teacher models.** Sample $(x, y)$ pairs from
   a strong teacher (GPT-4, Claude, an internal frontier model). Phi-4
   uses ~400B tokens of synthetic data from "50+ pipelines"
   `[phi4 §1, §2.1; kb/excerpts/phi4#sec-1-pillars,
   kb/excerpts/phi4#sec-2-1-purpose]`. The teacher provides quality;
   diversity is engineered through prompt-design pipelines.
4. **Self-synthesis from the aligned model itself.** **MAGPIE**
   `[magpie-2024]` exploits the fact that a chat-tuned LM, when given
   only its instruction-template prefix (system prompt + `<|user|>`
   tag) with no actual question, will fabricate an instruction first;
   sampling then yields the answer. This produces prompt+response pairs
   at zero human cost from any open-weights aligned model.

These streams are typically deduplicated, filtered for safety, balanced
across domains (math, code, reasoning, multi-turn dialogue, refusal,
multilingual), and quality-gated by a reward model or LLM-judge before
training.

### 2.2 SFT training step

For each minibatch of tokenized $(x, y)$ pairs:

1. Concatenate $z = x \,||\, y$ with the appropriate chat template
   (e.g. `<|user|>x<|assistant|>y<|end|>`); construct loss mask $m_t$.
2. Forward pass: compute logits $\ell_t = \pi_\theta(\cdot | z_{<t})$
   for $t = 1, \ldots, |z|$. Shape $(B, |z|, V)$ where $V$ is vocab.
3. Compute per-token loss
   $L_t = -m_t \log \mathrm{softmax}(\ell_t)_{z_t}$.
4. Reduce: $\mathcal{L} = \sum_t L_t / \sum_t m_t$ (mean over masked
   tokens) or $\sum_t L_t / \sum_t 1$ (mean over all tokens, including
   masked-out positions where the mask zeros the contribution). The
   choice affects effective LR but not the gradient direction; Tülu 3
   uses the response-token-only mean.
5. Backward; optimizer step (typically AdamW with $\beta_1 = 0.9,
   \beta_2 = 0.95$, weight decay 0.1; cosine-decay LR from
   $5\times 10^{-6}$ to 0; cf. `kb/notes/training/optimization.md §2.1`).

Total compute per epoch: $\sim 6 N_{\mathrm{tok}} P$ FLOPs (1 forward +
2 backward for activations and gradients, in standard mixed-precision
training), where $N_{\mathrm{tok}} = \sum_i |x^{(i)}| + |y^{(i)}|$.

### 2.3 Post-SFT: rejection sampling and iteration

Modern recipes do not stop at one SFT pass. **Rejection sampling**
(Llama-3 `[meta-llama3]`) is the canonical quality gate before each new
preference round:

1. From the current $\pi^{\mathrm{SFT}}$, sample $N$ responses per
   prompt ($N = 10$–$30$).
2. Score each with a reward model.
3. Keep the top-$k$ ($k = 1$–$3$) per prompt.
4. SFT-fine-tune on the kept responses, producing
   $\pi^{\mathrm{SFT}'}$.

The Llama-3 post-training loop alternates rejection-sampling SFT with
DPO across multiple rounds `[meta-llama3]`. This is "on-policy SFT
filtered by an off-policy reward model" — the model learns from its own
distribution but only where the RM judges it correct.

### 2.4 Reasoning-distillation SFT (DeepSeek-R1)

`[deepseek-r1; kb/excerpts/deepseek-r1#abstract]`: after training R1 by
RL with verifiable rewards, the team curated **800K SFT samples** (600K
reasoning + 200K non-reasoning) by sampling from R1 and filtering for
correctness. Pure-SFT fine-tuning of base models (1.5B–32B Qwen and
Llama) on this set yielded "DeepSeek-R1-Distill" checkpoints that match
or approach R1's reasoning performance — without any RL.

This is the strongest evidence as of 2026 that **reasoning capability
is largely transferable through SFT**: the RL stage produces good
rollouts, but distilling those rollouts into a smaller model via SFT
preserves most of the gain. The follow-on work `s1-2025`
`[s1-2025; kb/excerpts/s1-2025]` reproduces this with only ~1K
hand-curated reasoning traces and budget-forcing at inference time,
showing the SFT data needed is far smaller than originally supposed.

## 3. Variants and lineage

### 3.1 Comparison of canonical SFT recipes

| Recipe | Year | Data scale | Source | Iterations | Stage in pipeline |
|---|---|---|---|---|---|
| **InstructGPT** `[ouyang2022-instructgpt]` | 2022 | ~13K demos | human labelers | 1 | $\to$ RM $\to$ PPO |
| **FLAN / FLAN-T5** `[wei2021-flan; chung2022-flan-t5]` | 2021–22 | ~1.8K tasks $\times$ multiple templates | task-format conversion | 1 | terminal (no RL) |
| **Llama-3** `[meta-llama3]` | 2024 | ~10M+ samples (rejection-sampled) | iterative on-policy + RM filter | $\geq 6$ rounds | $\to$ DPO |
| **Tülu 3** `[tulu3]` | 2024 | ~939K samples | mixed: organic + synthesized + persona | 1 | $\to$ DPO $\to$ RLVR |
| **Qwen 3** `[qwen3]` | 2025 | mixed thinking + non-thinking | mixed | 4-stage post-training | $\to$ RLVR |
| **MAGPIE** `[magpie-2024]` | 2024 | self-generated | aligned-LM self-prompting | 1 | drop-in for SFT data |
| **DeepSeek-R1-Distill** `[deepseek-r1]` | 2025 | 800K (600K reasoning) | sampled from R1 + filtered | 1 | terminal (no RL on small model) |
| **Phi-4** `[phi4]` | 2024 | ~400B tokens synthetic | 50+ teacher-driven pipelines | 1 (mixed with pre-train) | $\to$ DPO |

### 3.2 The FLAN lineage — instruction-tuning before RLHF

Before the RLHF era, **instruction-tuning** was the post-training
default. **FLAN** `[wei2021-flan]` reformatted ~1.8K NLP tasks
(classification, QA, summarization, NLI) as natural-language
instructions and fine-tuned on them with cross-entropy. **T0**
(Sanh et al. 2021) and **FLAN-T5** `[chung2022-flan-t5]` extended this
multi-task instruction-tuning, with FLAN-T5 (1.8K tasks, multiple
prompting techniques, CoT inclusion) becoming the canonical reference.
**The FLAN Collection** (Longpre et al. 2023, arXiv:2301.13688)
consolidated the recipes.

The 2022 InstructGPT result `[ouyang2022-instructgpt §4;
kb/excerpts/ouyang2022-instructgpt#sec-4]` — that a 1.3B RLHF'd model
beats 175B GPT-3 — was the inflection point: SFT alone (the FLAN
recipe) gave good zero-shot generalization, but SFT+RLHF gave better
helpfulness *as judged by users on free-form prompts*. After 2022, the
open-source frontier converged on SFT-as-first-stage in a multi-stage
pipeline rather than SFT-as-terminal.

### 3.3 Synthetic-data SFT — MAGPIE, persona-hub, self-instruct

**MAGPIE** `[magpie-2024]` is the most striking variant. The procedure
is structurally trivial:

1. Take any aligned LM $\pi^{\mathrm{Aligned}}$ (LLaMA-3-Instruct, Qwen-
   Chat, etc.).
2. Prompt it with **only its instruction-template prefix** —
   `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n` —
   with no question.
3. Sample. The model fabricates the user question.
4. Append the instruction-template response delimiter; sample again to
   get the answer.

This produces $(x, y)$ pairs at the cost of two forward passes per
example, with no human input and no teacher distillation curriculum.
SFT on MAGPIE-generated data on Llama-3-8B-base recovers most of the
performance of Llama-3-8B-Instruct on AlpacaEval and MT-Bench
`[magpie-2024]`.

[INTUITION] What MAGPIE exposes is that an aligned model's
**conditional distribution** $p(\text{question} | \text{template-prefix
only})$ is itself a useful instruction distribution. The base model's
unconditional distribution doesn't have this property — it would
generate web text. Alignment training has produced a "user-question
prior" baked into the conditional distribution after the system prompt;
MAGPIE samples from that prior. Returns to canonical form: SFT on
MAGPIE data is just Eq. (1) with $\mathcal{D}_{\mathrm{SFT}}$
constructed by sampling from $\pi^{\mathrm{Aligned}}(\cdot | \text{prefix})$.

### 3.4 Reasoning SFT — DeepSeek-R1-Distill, s1, rStar-Math

The 2025 result that **reasoning capability transfers through SFT**
(no RL needed for distillation) is not yet in the canonical textbook
treatment but is well-replicated:

- **DeepSeek-R1-Distill** `[deepseek-r1]`: 800K samples from R1; SFT on
  Qwen-2.5 / Llama-3 base; matches GPT-4o on AIME, beats it on MATH-500.
- **s1** `[s1-2025]`: 1K hand-curated reasoning traces from Gemini 2.0
  Flash Thinking; SFT + budget-forcing. Reaches near-o1-mini on AIME.
- **rStar-Math** `[rstar-math2025]`: MCTS-generated traces for math;
  SFT-only training of a small math expert.

The shared finding: **a few thousand high-quality CoT traces are enough
to elicit reasoning behavior in a strong base model**, provided the
traces are clean (correct, fully-worked, in the model's preferred
style). Per-example data quality dominates per-example data quantity at
this scale.

### 3.5 SFT in MoE / mixture models

For mixture-of-experts models (Mixtral, DeepSeek-V3, Qwen3-MoE), SFT
proceeds identically with one wrinkle: **expert utilization** can drift
during SFT if the SFT data is concentrated in a few domains. Tülu 3
notes balancing expert load by mixture composition; DeepSeek-V3
`[deepseek-v3]` reports expert routing remained stable across post-
training. No dedicated MoE-SFT theory exists as of 2026.

## 4. Intuitions and analogies

[ANALOGY] **SFT as "behavior cloning."** SFT is supervised imitation
of $(x, y)$ pairs. In RL terms, $\mathcal{L}_{\mathrm{SFT}}$ is the
log-likelihood of the demonstrator policy
$\pi^{\mathrm{demo}}$ — the human or teacher that produced $y$. The
canonical math returns: minimizing Eq. (1) is equivalent to minimizing
the forward KL $\mathrm{KL}(\pi^{\mathrm{demo}} \| \pi_\theta)$ in the
infinite-data limit (with $\pi^{\mathrm{demo}}$ implicit in
$\mathcal{D}_{\mathrm{SFT}}$). RLHF/DPO use the *reverse* KL with
$\pi^{\mathrm{SFT}}$ as anchor; SFT and RLHF are dual KL directions
around the demonstrator and the SFT-stage model respectively.

[INTUITION] **Why prompt-masking matters.** Without prompt masking the
SFT loss includes $\sum_t \log \pi_\theta(x_t | x_{<t})$ — the
likelihood of the prompt itself. For a base model, prompts are already
near-distribution; the gradient through the prompt teaches the model
to *generate prompts*, which is the wrong objective and dilutes the
response signal. Mathematically, prompt-masking is equivalent to
treating $x$ as a fixed conditioning input rather than a sample to
model.

[ANALOGY] **The "1/3 synthetic, 2/3 organic" rule of thumb.** Several
2024 OSS post-training recipes report a ratio around 1:2 synthetic to
organic in the SFT mix. [SPECULATION] The intuition is that synthetic
data is dense in skill but narrow in distribution; organic data is
distribution-broad but skill-sparse. Mixing balances both. No paper
has rigorously derived this ratio; Phi-4 uses synthetic-heavy mixes,
Tülu 3 uses organic-heavy mixes, and both work at their respective
scales. Treat as a [FORUM-SIGNAL] heuristic until ablated.

[INTUITION] **Why SFT before RL at all.** RLHF/DPO/RLVR all need a
"reasonable starting policy" to roll out from. A base LM produces
incoherent or web-style text and is not a useful RL initialization —
the rollouts are mostly unscorable, the reward gradient is uselessly
sparse, and KL($\pi_\theta \| \pi^{\mathrm{base}}$) is meaningless as a
trust region for "acting like an assistant." SFT provides the
distribution shift from "language model" to "instruction-follower"
cheaply, after which RL can refine. Removing the SFT stage is feasible
in the verifiable-reward setting (DeepSeek-R1-Zero in
`kb/notes/post-training/rlvr-and-grpo.md`) but the common practice
remains SFT-then-RL.

## 5. Frontier and open questions (as of 2026-05)

### 5.1 [CONTRADICTION] How much SFT data is enough?

Two empirical traditions disagree:

- **"More is better, with quality gates"** (Tülu 3, Llama-3, Phi-4):
  scale to 1M+ samples, mostly synthetic, with rejection-sampling
  and reward-model filtering. Tülu 3 reports clean monotonic gains
  through ~1M samples `[tulu3]`.
- **"Few high-quality samples suffice"** (LIMA — Zhou et al. 2023; s1
  `[s1-2025]`; LessIsMore — Liu et al. 2024): 1K–10K carefully curated
  examples are enough to elicit instruction-following from a strong
  base.

Likely both true in different regimes: small-curated-set works for
*instruction style and tone*; large-set is needed for *capability
breadth* (math, code, multilingual, refusal, safety). The curated-set
results are typically benchmarked on AlpacaEval / MT-Bench, which
measure style; the large-set wins on reasoning + multilingual + safety
benchmarks. Phase 1 sweep §sft notes no canonical reconciliation has
been published.

### 5.2 The synthetic-fraction question

`[SPECULATION]` What fraction of an SFT mix should be synthetic vs.
organic? Phi-4 `[phi4]` makes the strongest synthetic-heavy case (~80%
synthetic by tokens). Llama-3 `[meta-llama3]` and Tülu 3 use organic-
heavier mixes. No paper has run a controlled ablation at frontier scale.
The closest is the data-mixing-laws line `[data-mixing-laws-2024]`,
which addresses pre-training mixtures, not SFT. Open.

### 5.3 [CONTRADICTION] Reasoning-distillation: is RL necessary for novel reasoning?

The DeepSeek-R1 result that reasoning distills cleanly through SFT
suggests pure-SFT can match RL-trained reasoners on the *same
distribution of problems*. But:

- **Anti-thesis (DeepSeek-R1 paper itself, §4 discussion):** SFT-
  distilled small models do not exceed their teacher; only RL can
  generate genuinely new reasoning trajectories.
- **Pro-thesis (s1, follow-on work):** s1's budget-forcing + minimal
  SFT data already extracts behavior that wasn't fully exploited by
  the teacher — suggesting SFT extracts more than one might think.

Open empirical question: does pure-SFT-from-strong-teacher saturate at
the teacher's reasoning frontier, or can it surpass through better
inference-time strategies? See `kb/notes/reasoning/test-time-compute.md`.

### 5.4 Loss-mask conventions for system + tool messages

Modern LLMs include system messages, tool-call results, and
multi-modal inputs. Conventions vary on which are masked:

- Tool-call *results* (the model didn't generate them): mask out.
  Universal.
- Tool-call *invocations* (the model generated them as part of its
  reasoning): include in loss. Universal.
- *System messages*: contested. Some recipes train on them (so the
  model learns to summarize a system message); others mask them
  (system is conditioning only).

No canonical reference exists for these conventions; they propagate
through OSS recipes as folklore.

### 5.5 Multilingual and refusal balance

The Tülu 3 recipe explicitly balances safety/refusal vs.
helpfulness vs. multilingual capability in the SFT mix `[tulu3]`. The
ratio tradeoff is treated as an empirical hyperparameter, with
overrefusal as the primary failure mode. No theoretical framework for
predicting the right balance; current practice is iterative human
evaluation.

### 5.6 SFT and the calibration question

`[FORUM-SIGNAL]` arXiv:2604.10585 (April 2026) reports that
sycophancy- and agreeableness-tuning (much of which lives in SFT data)
collapses the model's calibration. Open question: is this a property of
the SFT *data* (annotators rewarded confident-sounding answers) or of
the SFT *loss* (cross-entropy on a single response token suppresses
multi-modal predictive distributions)? Both plausible; not separated
empirically.

## 6. See also

- `kb/notes/post-training/rlhf.md` — PPO-RLHF starts from
  $\pi^{\mathrm{SFT}}$ and uses it as the KL anchor.
- `kb/notes/post-training/dpo-and-offline.md` — DPO starts from
  $\pi^{\mathrm{SFT}}$ and uses it as $\pi_{\mathrm{ref}}$ in Eq. (1)
  of that note.
- `kb/notes/post-training/rlvr-and-grpo.md` — DeepSeek-R1's "cold-start"
  SFT precedes the RL stage; R1-Zero attempts to skip it.
- `kb/notes/post-training/rlaif-and-constitutional.md` — SL-CAI is a
  CAI-specific SFT variant where revisions are critique-driven.
- `kb/notes/training/synthetic-data-and-distillation.md` — the data
  source for SFT in modern recipes; deeply overlapping.
- `kb/notes/reasoning/reasoning-training.md` — DeepSeek-R1-Distill,
  s1, rStar-Math reasoning-via-SFT recipes.
