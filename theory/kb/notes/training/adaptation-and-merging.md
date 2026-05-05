---
topic: training/adaptation-and-merging
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - hu2021-lora
  - dettmers2023-qlora
  - wortsman2022-model-soups
  - ilharco2022-task-vectors
  - yadav2023-ties-merging
secondary_sources:
  - mergekit2024
  - phi4
  - deepseek-v3
related_topics:
  - training/pre-training-data
  - training/synthetic-data-and-distillation
  - post-training/sft
  - inference/quantization
---

# Adaptation and merging

This note covers techniques that **modify a trained model without full
re-training**. The unit of action is parameter-level — adding,
averaging, mixing, or low-rank-updating weights — rather than gradient-
via-loss on data, which is the territory of SFT/RLHF/DPO. Five
sub-areas:

1. **Continual / late-stage pre-training** — same architecture and
   pipeline, mixture shifts toward higher-quality sources at the end
   of training. Phi-4's "midtrain" stage `[phi4 §1, §2.1;
   kb/excerpts/phi4#sec-1-pillars]`, OLMo 2's Dolmino mix, DeepSeek-V3's
   119K-H800-hr context-extension stage `[deepseek-v3 §1;
   kb/excerpts/deepseek-v3-training#sec-1-cost]`. Discussed in
   `kb/notes/training/pre-training-data.md`.
2. **Parameter-efficient fine-tuning (PEFT)** — adapt a frozen base via
   low-rank or sparse updates. **LoRA** (Hu et al. 2021)
   `[hu2021-lora]` is canonical; **QLoRA** (Dettmers et al. 2023)
   `[dettmers2023-qlora]` quantizes the base for memory efficiency.
3. **Model merging** — average / interpolate / sign-resolve weights
   from two or more trained models that share a common ancestor.
   **Model Soups** `[wortsman2022-model-soups]`, **Task Arithmetic**
   `[ilharco2022-task-vectors]`, **TIES-Merging**
   `[yadav2023-ties-merging]`, plus DARE, evolutionary merging, and
   the **MergeKit** toolkit `[mergekit2024]`.
4. **Pruning / weight surgery** — SparseGPT, Wanda, layer-drop. Closer
   to inference-time compression in practice; see
   `kb/notes/inference/quantization.md`.
5. **Frankenmerges** — layer-wise interleaving of two models. Widely
   used in OSS practice; limited primary literature. `[FORUM-SIGNAL]`.

This note treats (2) and (3) as the load-bearing scientific topics;
(1) cross-references pre-training-data; (4) is in inference; (5) is
flagged as forum-signal.

## 1. Formal definition

### 1.1 PEFT — LoRA

LoRA `[hu2021-lora §3, §4]` parameterizes the *update* to a frozen
linear-layer weight $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
as a low-rank product:

$$
W = W_0 + \Delta W = W_0 + \alpha \cdot B A, \quad B \in \mathbb{R}^{d_{\text{out}} \times r},\ A \in \mathbb{R}^{r \times d_{\text{in}}} \tag{1}
$$

with $r \ll \min(d_{\text{in}}, d_{\text{out}})$, typically $r \in
\{8, 16, 32, 64\}$. Only $A$ and $B$ are trained; $W_0$ is frozen.
$\alpha$ is a scalar scaling factor (typical: $\alpha = 2r$, sometimes
called the "LoRA $\alpha$"; the effective scale is $\alpha / r$).

| Symbol | Meaning |
|---|---|
| $W_0$ | frozen pretrained weight |
| $B, A$ | trainable low-rank factors; $A$ initialized $\mathcal{N}(0, \sigma^2)$, $B$ initialized 0 |
| $r$ | rank of the update |
| $\alpha$ | scaling factor (effective LR multiplier on $\Delta W$) |
| $d_{\text{in}}, d_{\text{out}}$ | input / output widths of the linear layer |

Because $B$ is initialized to zero, $\Delta W = 0$ at the start of
training and the model behaves identically to the base. Trainable
parameter count drops from $d_{\text{out}} \cdot d_{\text{in}}$ to
$r(d_{\text{out}} + d_{\text{in}})$, typically $10^4$× fewer
`[hu2021-lora §1]`.

### 1.2 QLoRA — quantize the base

QLoRA `[dettmers2023-qlora §3]` extends LoRA by **quantizing $W_0$ to
4-bit NF4** (NormalFloat-4, a quantile-based 4-bit format) while keeping
the LoRA factors $A, B$ in BF16. The forward pass is:

$$
y = (\mathrm{dequant}(W_0^{\text{NF4}})\ x) + \alpha\,(B (A x)) \tag{2}
$$

with $W_0^{\text{NF4}}$ stored in 4-bit and dequantized to BF16
on-the-fly per linear layer. Memory savings come from the $W_0$ side:
a 65B-parameter base model fits in 48GB GPU RAM (vs. 130GB BF16),
allowing full fine-tuning of large models on commodity hardware.
QLoRA also introduces **double quantization** (quantize the
quantization scales themselves) and **paged optimizers** for
memory-constrained environments.

### 1.3 Model merging — task vectors and weight averaging

**Task vector** `[ilharco2022-task-vectors §2]`: given a pretrained
base $\theta_0$ and a fine-tuned $\theta_f$ on task $T$, define the
**task vector**:

$$
\tau_T = \theta_f - \theta_0 \in \mathbb{R}^P \tag{3}
$$

Task vectors live in weight-difference space. Arithmetic on them
produces new models: $\theta_0 + \tau_T$ recovers the fine-tuned
model; $\theta_0 - \tau_T$ produces "task negation" (unlearning the
task); $\theta_0 + \sum_i \alpha_i \tau_{T_i}$ produces a multi-task
combination `[ilharco2022-task-vectors §2, §3]`.

**Model Soup** `[wortsman2022-model-soups §3]`: given $K$ fine-tunes
$\{\theta_k\}_{k=1}^K$ from the same base $\theta_0$, the **uniform
soup** is:

$$
\theta_{\mathrm{soup}} = \frac{1}{K}\sum_{k=1}^K \theta_k \tag{4}
$$

The **greedy soup** adds models one at a time only if including them
improves a held-out metric `[wortsman2022-model-soups §3]`. The
empirical claim: averaging fine-tunes from the same base often
outperforms the best single fine-tune, with no inference-time cost
(the average is a single model).

### 1.4 TIES-Merging — sign-conflict resolution

TIES-Merging `[yadav2023-ties-merging §3]` resolves the **interference
problem** in naive task-vector addition: when two fine-tunes update
the same coordinate in opposite directions, plain summation cancels
both. TIES applies three steps to each task vector $\tau_i$:

1. **Trim** — keep the top-$k\%$ of $|\tau_i|$ entries by magnitude;
   zero out the rest.
2. **Elect sign** — for each coordinate $j$, take the **sign that has
   the largest total magnitude** across tasks:
   $s_j = \mathrm{sign}\left(\sum_i \tau_{i,j}\right)$ where the sum is
   weighted by $|\tau_{i,j}|$.
3. **Disjoint merge** — for each coordinate, average only the task
   vectors whose sign matches the elected $s_j$:
   $\bar{\tau}_j = \frac{1}{|\mathcal{A}_j|}\sum_{i \in \mathcal{A}_j} \tau_{i,j}$,
   where $\mathcal{A}_j = \{i : \mathrm{sign}(\tau_{i,j}) = s_j\}$.

The merged model is $\theta_0 + \lambda \bar{\tau}$ with scaling
$\lambda$. TIES outperforms naive task-vector sums and Model Soup
across vision and NLP benchmarks `[yadav2023-ties-merging §4]`.

## 2. Mechanism — PEFT and merging protocols

### 2.1 LoRA training protocol

For each linear layer chosen for adaptation (typically all
attention $Q, K, V, O$ projections; sometimes FFN as well):

1. Replace $W_0$ with the parameterization $W_0 + \alpha B A$;
   freeze $W_0$, register $A, B$ as trainable.
2. Forward pass: compute both
   $W_0 x$ and $\alpha B (A x)$, sum.
3. Backward pass: gradients flow through $A, B$ only. The base does
   not need to receive a gradient.
4. Optimizer step on $A, B$. Memory cost: optimizer state (Adam $m, v$)
   for $A, B$ only — orders of magnitude smaller than full fine-tune.
5. **Merge for inference**: at deployment time, compute $W = W_0 +
   \alpha B A$ once and use $W$ directly. No latency overhead vs.
   the base.

The merge step is what distinguishes LoRA from adapter layers
(Houlsby et al. 2019): adapters add inference-time computation; LoRA's
merge is free.

### 2.2 QLoRA wrinkles

QLoRA's **NF4 format** `[dettmers2023-qlora §3.1]` is a 4-bit
quantization with quantile-spaced bins matched to a standard normal
distribution — exploiting that pretrained weights are approximately
$\mathcal{N}(0, \sigma^2)$ blockwise. Per-block (block size 64)
quantization scales are stored alongside the 4-bit data.

**Double quantization**: the per-block scales themselves (FP32) are
quantized to 8-bit. Memory savings: ~0.37 bits per parameter average
overhead vs ~0.5 without. Critical at 65B+ scale.

**Paged optimizers**: NVIDIA's unified memory pages optimizer state
between GPU and CPU when GPU is full. Without this, optimizer-state
memory can OOM the 48GB-GPU 65B fine-tune.

QLoRA-trained models recover within ~1% of full BF16 fine-tunes on
benchmark evaluations `[dettmers2023-qlora §4]`.

### 2.3 Model merging protocols

The general merge pipeline:

1. **Common ancestor.** Verify $\theta_0$ is shared across all
   $\theta_k$ being merged. Merging unrelated bases (different
   pretraining) is not supported by any of LoRA, Soups, or TIES — the
   loss landscapes don't share a basin.
2. **Compute task vectors.** $\tau_k = \theta_k - \theta_0$.
3. **Merge step.** Choose method:
   - Uniform soup: $\theta_{\text{merge}} = \theta_0 + \frac{1}{K}\sum_k \tau_k$.
   - Greedy soup: search subset $S \subseteq \{1,\ldots,K\}$ maximizing
     held-out score, using $\theta_0 + \frac{1}{|S|}\sum_{k\in S} \tau_k$.
   - Task arithmetic: $\theta_0 + \sum_k \alpha_k \tau_k$ with
     hand-picked $\alpha_k$.
   - TIES: trim → elect sign → disjoint merge → scale.
   - DARE (Yu et al. 2024, arXiv:2311.03099): Bernoulli-drop $1 - p$
     of $\tau_k$ entries, rescale survivors by $1/p$, then apply
     TIES or Soup.
   - SLERP (spherical linear interpolation): $\theta_0 + \mathrm{slerp}(\tau_1, \tau_2, t)$
     for two-model merges.
4. **(Optional) recovery fine-tune.** Brief SFT (10–100M tokens) to
   smooth out merge artifacts. Most production merges skip this; a
   minority add a recovery pass.

### 2.4 MergeKit — the open-source consolidation

MergeKit `[mergekit2024]` is the canonical open-source toolkit
implementing all of the above, plus:

- **Layer-stacking / "passthrough" merges** — concatenate layers from
  different models (Frankenstein-style), e.g., copy layers 0–15 from
  model A and layers 16–31 from model B.
- **Linear / SLERP / TIES / DARE / TIES-DARE / DARE-SLERP** as
  configurable merge methods.
- **Per-tensor configuration** — different merge methods for
  embeddings vs. attention vs. FFN.

MergeKit YAML configs are shared on Hugging Face hub and reproduce
specific merged models. The Goddard et al. 2024 paper documents the
toolkit's design.

## 3. Variants and lineage

### 3.1 PEFT comparison

| Method | Year | Trainable params | Inference cost | Notes |
|---|---|---|---|---|
| **Full fine-tune** | — | $P$ | none | maximum capacity, maximum cost |
| **Adapter** (Houlsby 2019) | 2019 | tiny | adds layers | inference latency overhead |
| **Prefix-tuning** (Li & Liang 2021) | 2021 | $L \cdot d$ per task | adds prefix | tuning prompt embeddings |
| **LoRA** `[hu2021-lora]` | 2021 | $r(d_{\text{out}} + d_{\text{in}})$ per layer | none after merge | most-deployed PEFT |
| **DoRA** (Liu et al. 2024) | 2024 | LoRA + magnitude vector | none after merge | decomposes direction + magnitude |
| **IA³** (Liu et al. 2022) | 2022 | $d$ scalars per layer | tiny | rescaling-only PEFT |
| **QLoRA** `[dettmers2023-qlora]` | 2023 | LoRA over 4-bit base | dequant per layer | 65B fine-tune on 48GB GPU |

LoRA is the production default; QLoRA is the memory-constrained
default; DoRA is the most-cited 2024 variant. IA³ and prefix-tuning
are niche.

### 3.2 Merge method comparison

| Method | Year | Sign handling | Sparsity | Multi-task quality |
|---|---|---|---|---|
| **Linear / Uniform Soup** `[wortsman2022-model-soups]` | 2022 | average (cancellation) | dense | OK if fine-tunes share basin |
| **Greedy Soup** `[wortsman2022-model-soups]` | 2022 | conditional include | dense | best of soup family |
| **Task Arithmetic** `[ilharco2022-task-vectors]` | 2022 | weighted sum | dense | controllable; manual $\alpha_i$ |
| **TIES-Merging** `[yadav2023-ties-merging]` | 2023 | majority sign vote | trim top-$k\%$ | strong on multi-task |
| **DARE** (Yu 2024) | 2024 | drop+rescale + TIES/Soup | random Bernoulli sparsify | small gains over TIES on some tasks |
| **Evolutionary Merging** (Akiba et al. 2024, Sakana) | 2024 | search merge weights | varies | best when search budget is generous |
| **SLERP** | (folklore → community 2024) | spherical interp two models | dense | preserves vector norms |

The merge ecosystem is largely empirical; theoretical understanding
lags practice. The Phase 1 sweep notes no canonical winner across
benchmark sets.

### 3.3 LoRA variants and MoE-LoRA

The LoRA family has spawned dozens of variants. Notable:

- **DoRA** (Liu et al. 2024): decompose $W$ into magnitude and
  direction, LoRA the direction only. Outperforms LoRA on many tasks
  with similar parameter count.
- **AdaLoRA** (Zhang et al. 2023): allocate rank adaptively across
  layers based on importance. More parameter-efficient but more complex.
- **LoRA-MoE** (various 2024): multiple LoRAs per layer with a router.
  Lets one fine-tune cover multiple skills.
- **S-LoRA / Punica** (2024): serving-time multi-tenant LoRA — load
  thousands of LoRAs and route requests to the right adapter on the
  fly. Touches on `kb/notes/inference/serving-systems.md`.

### 3.4 Continual / late-stage pre-training

**Phi-4** `[phi4 §1, §2.1; kb/excerpts/phi4#sec-1-pillars,
kb/excerpts/phi4#sec-2-1-purpose]` introduces a "midtrain" stage:
between full pre-training and post-training, a late-stage mixture-
shift toward synthetic reasoning-heavy data. **DeepSeek-V3**'s
context-extension stage `[deepseek-v3 §1;
kb/excerpts/deepseek-v3-training#sec-1-cost]` uses 119K H800-hours
($0.238M, 4.3% of total compute) to extend context from 4K to 128K
via the YaRN scaling protocol — a form of late-stage adaptation.

These stages share a pattern with WSD-cooldown LR schedules
(`kb/notes/training/optimization.md §2.5`): when LR is low, the model
is locked in to whatever data it sees, so late-training data with
small $\eta$ has outsized impact.

## 4. Intuitions and analogies

[ANALOGY] **LoRA as a "plug-in patch" to a frozen base.** A LoRA
adapter is like a software patch: small, can be loaded/unloaded
without touching the base, and merges into the base for deployment.
The canonical math returns to Eq. (1): the patch is the rank-$r$
update $BA$, and the merge is the addition $W_0 + \alpha BA$. Merging
multiple LoRAs trained on different tasks corresponds to adding
multiple patches; this is exactly task arithmetic in the LoRA
parameter space.

[INTUITION] **Why low-rank works for fine-tuning but not for pre-
training.** Pre-training learns broad representations from scratch;
the parameter changes from initialization are full-rank. Fine-tuning
adjusts a much smaller "task-specific direction" in weight space; the
update is empirically near-low-rank `[hu2021-lora §4]`. This
distinction is critical: trying to pre-train a model with rank-$r$
constraints (Aghajanyan et al. 2020 'low intrinsic dimension')
underperforms; fine-tuning at rank $r = 8$ recovers ~99% of full-fine-
tune quality `[hu2021-lora §6]`.

[ANALOGY] **Model merging as ensembling-without-the-inference-cost.**
A traditional ensemble runs $K$ models in parallel and averages
predictions. Model Soup runs $K$ models in parallel **only at training
time**, averaging weights at the end and running 1 model at inference.
Returns to canonical form: when fine-tunes share a basin (i.e., are
linearly mode-connected), the average of weights approximates the
center of the basin, which behaves like the ensemble of models within
that basin. The condition (basin-sharing) is what makes the analogy
tight `[wortsman2022-model-soups §3]`.

[INTUITION] **Why TIES beats naive averaging.** The interference
problem: if two task vectors update coordinate $j$ in opposite
directions, naive averaging cancels them. TIES detects this with
sign-election and zeros out coordinates where signs disagree, so the
final merge keeps the dominant-sign updates and ignores the
contradictions. The math returns to step 3 of §1.4: $\bar{\tau}_j$ is
the average over the *agreeing* subset $\mathcal{A}_j$, not the full
set. The "interference" is exactly the gap between these two averages.

[CONTRADICTION] **Whether merged models exceed any single ingredient.**
Many merge papers report merged > best ingredient; many real-world
merges report merged < best ingredient. The discrepancy is partly
measurement: merge wins are most consistent on multi-task evaluations
where each ingredient was specialized to a subset of tasks; on
single-task evaluations the best specialist wins. Treat "merged
beats best ingredient" as a *task-set-conditional* claim.

## 5. Frontier and open questions (as of 2026-05)

### 5.1 Why merging works at all — linear mode connectivity

The dominant theory `[FORUM-SIGNAL: Frankle, Dziugaite et al. 2020,
linear mode connectivity]` is that fine-tunes of a shared base
$\theta_0$ land in a **connected loss basin** — the path
$\theta_0 + t (\theta_f - \theta_0)$ for $t \in [0, 1]$ stays in low-
loss territory. If true, then averaging fine-tunes lands at the
basin's center, which empirically behaves like the basin's ensemble.

Empirically this holds for SFT-from-the-same-base and partly for
RLHF-from-the-same-base; it breaks for from-scratch models with
different initializations. Open questions: at what training step does
linear mode connectivity emerge? Does it survive RL training? No
canonical paper, only scattered evidence. [SPECULATION] This is the
single most important theoretical question in adaptation right now,
because it determines which models *can* be merged.

### 5.2 Merging vs. distillation

Merging is parameter-level and pays no inference cost; distillation is
data-level and pays a teacher pass per sample (see
`kb/notes/training/synthetic-data-and-distillation.md`). Comparative
empirical results exist (distillation often wins on capability transfer;
merging wins on specialist combination) but no unified theory predicts
which wins for which task. As of 2026 the OSS community uses merging
for combining specialists from a shared base and distillation for
transferring capability across model families.

### 5.3 [CONTRADICTION] LoRA quality at scale

Two competing claims:

- **LoRA-only views (academic)**: rank $r = 8$ recovers full-FT
  quality `[hu2021-lora §6]` on standard benchmarks.
- **Production views**: LoRA underperforms full-fine-tune on
  reasoning-heavy or long-context tasks at frontier scale; full-FT
  is preferred when memory permits.

The discrepancy is task-distribution-dependent. LoRA is excellent for
style adaptation, instruction-following, and persona; it is shakier
for reasoning capability transfer (where full-FT or a higher rank
$r = 64$–$128$ is needed). [FORUM-SIGNAL] vendor reports diverge.

### 5.4 Frankenmerge / passthrough merges

Layer-wise interleaving of two models — "frankenmerges" — is widely
used in the Hugging Face community (Mistral derivatives, Llama-2
chat variants). The mechanism is unstudied: in principle layer
$\ell$ from model A composed with layer $\ell+1$ from model B
should produce activations that are out-of-distribution for layer
$\ell+1$. In practice many such merges pass benchmarks without
recovery fine-tuning. No primary literature explains this; treat as
[FORUM-SIGNAL].

### 5.5 Multi-LoRA serving and dynamic routing

S-LoRA (2024), Punica (2024), and follow-ons let a single serving
instance host thousands of LoRAs. Open questions:
- Optimal dispatch when LoRAs share a tenant.
- KV-cache implications when adapter changes per request.
- Routing models that select the right adapter per request.

This is moving rapidly with the production push to per-customer fine-
tunes. See `kb/notes/inference/serving-systems.md §8`.

### 5.6 Evolutionary / black-box merging

Akiba et al. 2024 (Sakana) showed merge-coefficient search via CMA-ES
can produce merged models that exceed naive merges by significant
margins. This treats merging as a black-box optimization problem.
Open question: is the search landscape smooth enough that simpler
methods (gradient-based merge-coefficient learning) would work? No
published comparison.

### 5.7 Pruning + recovery as adaptation

Post-training structured pruning (SparseGPT, Wanda) followed by a
brief recovery SFT is increasingly used to compress models. The
boundary between this and "weight surgery" is fuzzy. Treated in
`kb/notes/inference/quantization.md` for the inference-side view.

## 6. See also

- `kb/notes/training/pre-training-data.md` — late-stage / midtrain
  data mixture is the data-side of "adaptation."
- `kb/notes/training/synthetic-data-and-distillation.md` — distillation
  as the data-level alternative to weight-merging.
- `kb/notes/training/optimization.md` — WSD-cooldown LR is the
  schedule-side mechanism that makes late-stage adaptation effective.
- `kb/notes/post-training/sft.md` — fine-tuning protocol; LoRA is
  often used as the SFT optimization target.
- `kb/notes/inference/quantization.md` — QLoRA's NF4 quantization
  shares its substrate with inference-time INT4 quantization (GPTQ,
  AWQ).
- `kb/notes/inference/serving-systems.md §8` — multi-tenant LoRA
  serving (S-LoRA, Punica).
