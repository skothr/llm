---
topic: training/optimization
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - muon-moonlight2025
  - hoffmann2022-chinchilla   # context: compute-optimal frame
  - kaplan2020                # context: power-law scaling
secondary_sources_planned:
  - sophia2023        # arXiv 2305.14342 — Sophia second-order
  - lion2023          # arXiv 2302.06675 — Lion sign-momentum
  - minicpm2024       # arXiv 2404.06395 — WSD origin
  - wsd-river-valley  # arXiv 2410.05192 — WSD theoretical basis
  - optimal-lr-2026   # arXiv 2602.06797 — optimal LR schedules
related_topics:
  - training/distributed-training
  - training/mixed-precision-and-stability
  - scaling/chinchilla
  - scaling/mu-transfer
---

# Optimization (LLM training)

The choice of optimizer + learning-rate schedule is one of the largest
remaining levers on training compute efficiency. As of 2026, **AdamW**
is the production default and **Muon** is the only optimizer with
published evidence of a ~2× compute-efficiency lift at LLM scale. The
schedule story has shifted from cosine-decay to **WSD** (Warmup–Stable–
Decay), with a theoretical basis arriving in 2024–2026.

## 1. Formal definition

The pre-training objective for a decoder-only LM with parameters
$\theta$ on a corpus of token sequences $\mathbf{x}$ is the
next-token cross-entropy:

$$\mathcal{L}(\theta) = -\mathbb{E}_{\mathbf{x} \sim \mathcal{D}}\!\left[\sum_{t=1}^{T} \log \pi_\theta(x_t \mid x_{<t})\right] \tag{1}$$

We minimize $\mathcal{L}(\theta)$ by stochastic first-order updates:

$$\theta_{t+1} = \theta_t - \eta_t \, U_t(g_t, \theta_t)$$

where $g_t = \nabla_\theta \mathcal{L}(\theta_t)$ on a mini-batch,
$\eta_t$ is the learning rate at step $t$ (governed by the schedule),
and $U_t$ is the update produced by the optimizer (which can hold
state — moments, momentum buffers, etc.).

| Symbol | Meaning |
|---|---|
| $\theta \in \mathbb{R}^P$ | full parameter vector ($P$ ~ 1B–1T) |
| $g_t$ | mini-batch gradient at step $t$ |
| $\eta_t$ | scalar learning rate (schedule-dependent) |
| $\mu, \beta_1, \beta_2$ | momentum and Adam moments |
| $\lambda$ | weight-decay coefficient |
| $\varepsilon$ | numerical-stability constant |
| $C$ | total training compute (FLOPs) |

## 2. Mechanism — the four canonical optimizers

### 2.1 AdamW — the production default

Adam keeps two exponential moving averages of the gradient and its
square, then divides one by the (square root of the) other:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

$$\hat m_t = m_t / (1 - \beta_1^t),\quad \hat v_t = v_t / (1 - \beta_2^t)$$

$$\theta_{t+1} = \theta_t - \eta_t \!\left[\hat m_t / (\sqrt{\hat v_t} + \varepsilon) + \lambda \theta_t\right] \tag{2}$$

The "W" in AdamW (Loshchilov & Hutter 2019) is the **decoupled weight
decay**: instead of folding $\lambda \theta$ into $g_t$ (which would
make it act through Adam's preconditioning), AdamW applies it
*after* the Adam step. At LLM scale, AdamW is the universal default —
LLaMA, Mistral, Qwen, Gemma, OLMo, DeepSeek-V3 all use it, with
typical $(\beta_1, \beta_2) = (0.9, 0.95)$, $\lambda = 0.1$,
$\varepsilon = 10^{-8}$. The empirical update RMS of AdamW is "usually
around 0.2 to 0.4" per matrix `[muon-moonlight2025 §2.2;
kb/excerpts/muon-moonlight2025#sec-2-2-rms]`.

### 2.2 Muon — orthogonalized momentum (matrix params only)

Muon (K. Jordan et al. 2024 / Liu et al. 2025 at scale) treats
matrix-shaped parameters $\mathbf{W} \in \mathbb{R}^{A \times B}$ as
matrices, not flattened vectors, and *orthogonalizes the momentum*
before the step:

$$\begin{aligned}
\mathbf{M}_t &= \mu \mathbf{M}_{t-1} + g_t \\
\mathbf{O}_t &= \mathrm{Newton\text{-}Schulz}(\mathbf{M}_t) \\
\mathbf{W}_t &= \mathbf{W}_{t-1} - \eta_t\!\left( 0.2\,\mathbf{O}_t \cdot \sqrt{\max(A,B)} + \lambda \mathbf{W}_{t-1}\right)
\end{aligned} \tag{3}$$

`[muon-moonlight2025 Eq.1, Eq.4; kb/excerpts/muon-moonlight2025#sec-2-1-update-rule, #sec-2-2-rms]`

The Newton-Schulz iteration is a matrix polynomial that approximates
$(\mathbf{M}_t \mathbf{M}_t^\top)^{-1/2} \mathbf{M}_t = \mathbf{U}
\mathbf{V}^\top$ (the "polar factor" — the orthogonal part of
$\mathbf{M}_t$'s SVD) without explicitly computing the SVD:

$$\mathbf{X}_k = a \mathbf{X}_{k-1} + b (\mathbf{X}_{k-1}\mathbf{X}_{k-1}^\top) \mathbf{X}_{k-1} + c (\mathbf{X}_{k-1}\mathbf{X}_{k-1}^\top)^2 \mathbf{X}_{k-1} \tag{4}$$

with $\mathbf{X}_0 = \mathbf{M}_t / \|\mathbf{M}_t\|_F$, coefficients
$(a,b,c) = (3.4445, -4.7750, 2.0315)$ tuned to converge in ~5
iterations `[muon-moonlight2025 §2.1, footnote;
kb/excerpts/muon-moonlight2025#sec-2-1-ns]`.

The two non-obvious ingredients that made Muon scale-stable:

1. **Weight decay is mandatory.** Without it, "the weight and the
   layer output's RMS keep growing to a large scale, exceeding the
   high-precision range of bf16" `[muon-moonlight2025 §2.2;
   kb/excerpts/muon-moonlight2025#sec-2-2-wd]`. With AdamW-style
   decoupled $\lambda \mathbf{W}$, Muon outperforms both vanilla
   Muon and AdamW in the over-train regime.
2. **Per-shape update-RMS scaling.** Muon's natural update RMS is
   $\sqrt{1/\max(A,B)}$ (Lemma 1 of the paper), so without correction,
   tall-thin matrices get under-updated and square small matrices
   get over-updated. The fix is to multiply each matrix's update by
   $\sqrt{\max(A,B)}$ and a factor $0.2$ that brings the RMS into
   AdamW's typical range — at which point AdamW's tuned $\eta$ and
   $\lambda$ can be **reused directly** `[muon-moonlight2025 §2.2;
   kb/excerpts/muon-moonlight2025#sec-2-2-rms]`.

Muon applies *only* to matrix parameters — embeddings, the LM head,
and 1-D parameters (RMSNorm gains, biases) are kept on AdamW. Hence
production runs use a **hybrid Muon + AdamW** optimizer, sharing the
same $(\eta, \lambda)$ across both.

The headline scaling-law claim is that Muon achieves the same loss as
AdamW with 52% of the FLOPs, fitted as $L_{\text{Muon}} = 2.506
C^{-0.052}$ vs $L_{\text{AdamW}} = 2.608 C^{-0.054}$
`[muon-moonlight2025 §3.2; kb/excerpts/muon-moonlight2025#sec-3-2-scaling]`.
The Moonlight model — 3B activated / 16B-MoE total, trained on 5.7T
tokens — is the largest published Muon training run as of 2026
`[muon-moonlight2025 §3.3;
kb/excerpts/muon-moonlight2025#sec-3-3-moonlight]`.

### 2.3 Lion — sign-momentum (briefly)

Lion (Chen et al. 2023) replaces the per-coordinate magnitude with
the *sign* of an exponentially smoothed gradient:

$$c_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad \theta_{t+1} = \theta_t - \eta_t (\mathrm{sign}(c_t) + \lambda \theta_t) \tag{5}$$

and $m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t$. Lion uses one moment
buffer (vs Adam's two) so the optimizer-state memory is halved. Phase
1 sweep finds Lion gives wall-clock savings on vision transformers
but **limited gains on LLMs** in head-to-head comparisons (e.g. the
2025 three-optimizer comparison arXiv:2507.08472).
[STUB — citation pending PDF download; treat as Phase-1-tier signal.]

### 2.4 Sophia — diagonal Hessian preconditioning

Sophia (Liu, Li et al. 2023, arXiv:2305.14342, ICLR 2024) uses a
stochastic estimate of the diagonal Hessian as a preconditioner. The
2× speedup-over-AdamW claim *on training loss* held up empirically,
but follow-on studies (Phase 1 sweep: arXiv:2507.08472) find that
**AdamW still wins on downstream task generalization** — Sophia
overfits the training distribution faster.
[STUB — Sophia equation pending PDF download.]

### 2.5 The schedule axis: cosine → WSD

Two schedules dominate frontier LLM training:

**Cosine decay** (Loshchilov & Hutter 2017):

$$\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\bigl(1 + \cos(\pi t / T)\bigr)$$

after a linear warmup. Used by GPT-3, LLaMA-1/2/3, most pre-2024
runs. Pin: $T$ = total training steps must be known in advance.

**WSD (Warmup–Stable–Decay)** (originated in MiniCPM, Hu et al. 2024,
arXiv:2404.06395; theoretical justification in arXiv:2410.05192,
2602.06797):

- **Warmup**: linear $\eta$ from 0 to $\eta_{\max}$ over a fixed
  prefix.
- **Stable**: hold $\eta = \eta_{\max}$ for the bulk of training.
- **Decay** (a.k.a. "cooldown"): rapid decay (linear or cosine) to
  $\eta_{\min}$ over the final 10–20% of training.

The decisive practical advantage: WSD lets you continue the stable
phase indefinitely without committing to $T$ in advance — useful for
*continuous training* runs and mid-training data-mixture changes (the
"Dolmino mix" pattern in OLMo 2). The Moonlight Muon run uses an
explicit WSD schedule:

> "0 to 33B tokens: … learning rate linearly increases to 4.2e-4 in
> 2k steps. … 33B to 5.2T tokens: learning rate decays from 4.2e-4
> to 4.2e-5 in a cosine style. … 5.2T to 5.7T tokens: In this stage
> (also referred as the cooldown stage), the learning rate increases
> to 1e-4 in 100 steps, and then linearly decays to 0 in 500B tokens
> … In this stage, we use the highest quality data, focusing on math,
> code, and reasoning." `[muon-moonlight2025 §3.3;
> kb/excerpts/muon-moonlight2025#sec-3-3-moonlight]`

The cooldown phase is *also* where curated high-quality data is
injected (the Dolmino-mix / OLMo 2 pattern), exploiting that
late-training data has outsized impact on final loss when $\eta$ is
small.

## 3. Variants and lineage — comparison table

| Optimizer | Year | State / matrix | Update RMS | LLM-scale evidence | Notes |
|---|---|---|---|---|---|
| **SGD + momentum** | 1964/1986 | $\mu$ | $\propto \|g\|$ | obsolete for LLMs | poor in deep + large-batch regime |
| **Adam** (Kingma & Ba 2015) | 2015 | $m, v$ | ≈1 | small models | unstable LR with weight decay folded in |
| **AdamW** (Loshchilov & Hutter 2019) | 2019 | $m, v$ | 0.2–0.4 `[muon-moonlight2025#sec-2-2-rms]` | universal | the production default |
| **Adafactor** (Shazeer & Stern 2018) | 2018 | $r, c$ (factored) | similar to Adam | T5, PaLM | rank-1 second-moment factorization saves memory |
| **Lion** (Chen et al. 2023) | 2023 | $m$ only | $\eta$ (sign step) | limited LLM gains | half the optimizer memory of Adam |
| **Sophia** (Liu, Li et al. 2023) | 2023 | $m, h$ (Hessian diag) | bounded by clip | lower train loss, downstream wins go to AdamW | second-order |
| **Muon** (Jordan 2024 / Liu et al. 2025) | 2024 | $\mathbf{M}$ (matrix) + Newton–Schulz | tuned to ≈ AdamW | ~2× AdamW efficiency at 16B-MoE / 5.7T tokens `[muon-moonlight2025#sec-3-2-scaling]` | matrix-only; AdamW for embeddings/heads/1-D params |
| **Schedule-Free** (Defazio et al. 2024) | 2024 | momentum + averaging | similar to Adam | ICML 2024 | removes LR schedule entirely |

The Muon lineage is the only 2024–2026 result that has displaced
AdamW from the *Pareto frontier* of pre-training compute (per the
landscape sweep, Muon now sits north-west of every published AdamW
checkpoint at matched arch + tokens). Whether this holds at 70B+
dense or 600B+ MoE scale is open
(`kb/excerpts/muon-moonlight2025#sec-1-challenges`).

## 4. Intuitions and analogies

[INTUITION] **Why Adam works for LLMs and SGD does not.** Per-
coordinate $1/\sqrt{\hat v}$ rescaling means coordinates with small
gradient variance get large effective LR and vice versa, which
matters because Transformer parameters of different shapes
(embeddings, attention QKV, FFN, LayerNorm gains) live on wildly
different scales. SGD's uniform step underweights low-variance
coordinates; Adam normalizes them to comparable scales. Returning to
math: under the assumption $\mathbb{E}[g_i^2]$ varies by orders of
magnitude across $i$, Adam's update is approximately invariant to
diagonal coordinate rescaling, SGD's is not.

[INTUITION] **Why Muon's orthogonalization wins.** Bernstein et al.
2024 frame both Adam and Muon as steepest descent under a *norm
constraint*. Adam's implicit norm is "Max-of-Max" (per-coordinate),
which is rotationally non-invariant — it bakes in a coordinate basis.
Muon's norm constraint is a Schatten-$p$ norm (a function of singular
values, which is rotation-invariant on each side) and "lies in a
static range" `[muon-moonlight2025 §2.1;
kb/excerpts/muon-moonlight2025#sec-2-1-norm]`. The orthogonalization
"can ensure that the update matrices are isomorphic, preventing the
weight from learning along a few dominant directions"
`[muon-moonlight2025 §2.1;
kb/excerpts/muon-moonlight2025#sec-2-1-update-rule]`. Returning to
canonical form: $\mathbf{O}_t = \mathbf{U}\mathbf{V}^\top$ has all
singular values equal to 1, so the update has the same energy in every
direction in the row/column subspaces of $\mathbf{M}_t$ — no direction
is implicitly preferred by gradient magnitude.

[ANALOGY] **WSD as a marathon-then-sprint.** Cosine decay is a
gradual taper from start to finish; WSD is a long run at constant
speed followed by a brief sprint at the end. The "sprint" (cooldown)
is when the model is locked in to high-quality data and a small
$\eta$ stabilizes the final weights. Returning to math: the
cooldown's small $\eta$ acts approximately like SGD-on-a-quadratic
near a minimum, where late training is well-modeled by averaging.
WSD-cooldown phase has its own active research literature
(arXiv:2508.01483, *Training Dynamics of the Cooldown Stage*).

[CONTRADICTION] **Lion vs AdamW for LLMs.** Lion was originally
reported to give ~1.5–2× speedup, but follow-on LLM-specific
comparisons (the Phase-1 cited 2025 three-optimizer paper) show
AdamW catches up or wins on downstream evaluation. The discrepancy
is consistent with Sophia: lower training loss does not always mean
better downstream generalization. Treat reported optimizer speedups
as *training-loss-conditional* until downstream is verified.

## 5. Frontier and open questions (as of 2026-05)

- **Muon at 70B+ dense and 600B+ MoE.** Moonlight (16B-MoE on 5.7T
  tokens) is the largest published. Whether the ~2× efficiency holds
  at frontier scale or the orthogonalization assumption breaks down
  is the open question per the paper's own Section 1
  `[muon-moonlight2025 §1; kb/excerpts/muon-moonlight2025#sec-1-challenges]`.
- **Distributed Muon communication.** The naive ZeRO-1 partitioning
  doesn't apply because Newton–Schulz needs the full gradient matrix.
  The Distributed Muon (Algorithm 1) gathers gradients per matrix,
  runs N-S, then discards the unused rows. Latency overhead is "1%
  to 3%" of forward-backward `[muon-moonlight2025 §2.3;
  kb/excerpts/muon-moonlight2025#sec-2-3-distributed]`. At 32k+ GPU
  scale, scaling this efficiently is open.
- **Optimal LR schedule from theory.** The Feb 2026 paper "Optimal
  LR Schedules under Functional Scaling Laws" (arXiv:2602.06797)
  derives WSD as a special case. Whether the prescribed schedules
  match empirically at frontier scale is open. [STUB — pending PDF.]
- **Schedule-Free (Defazio et al. 2024) at LLM scale.** Removes the
  LR schedule entirely via online averaging. Limited large-scale LLM
  validation. Open.
- **Sophia generalization gap.** Why does a method that achieves
  lower training loss generalize worse downstream? An open
  optimization-theory question with potentially deep implications for
  what we consider "good" pre-training optima.

## 6. See also

- `kb/notes/training/distributed-training.md` — Distributed Muon
  needs the full gradient matrix; this changes the ZeRO/FSDP
  algorithm.
- `kb/notes/training/mixed-precision-and-stability.md` — weight RMS
  growth in vanilla Muon "exceeds the high-precision range of bf16";
  precision and stability are deeply linked to optimizer choice.
- `kb/notes/scaling/chinchilla.md` — compute-optimal $D \approx 20 N$
  is the regime where optimizer comparisons are usually run.
- `kb/notes/scaling/mu-transfer.md` — μP and Muon both interact with
  per-shape update scaling; an integrated story has yet to be written.
