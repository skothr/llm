---
topic: scaling/chinchilla
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - hoffmann2022-chinchilla
  - kaplan2020
secondary_sources:
  - meta-llama3      # Llama 3 ~1875 tokens/param: industry post-Chinchilla
  - scaling-laws-precision-2024
related_topics:
  - scaling/kaplan-laws
  - scaling/scaling-frontier
  - scaling/inference-time-compute-scaling
  - training/optimization
  - training/pre-training-data
---

# Chinchilla — compute-optimal training

The Chinchilla paper (Hoffmann et al., 2022) established that, at a
fixed pre-training compute budget, model parameters and training tokens
should be scaled in **equal proportions** — overturning the Kaplan
(2020) recommendation that pre-training compute should be spent
primarily on bigger models. The empirical kernel: 70B parameters
trained on 1.4T tokens beats 280B params trained on 300B tokens at the
*same* total compute, on essentially every downstream task they
measured `[hoffmann2022-chinchilla §4.2; kb/excerpts/hoffmann2022-chinchilla#sec-4-2]`.

## 1. Formal definition

### 1.1 The setup

Let $C$ be a fixed training-compute budget in FLOPs, $N$ the number of
model parameters, $D$ the number of training tokens. The standard
Transformer-training compute approximation
`[kaplan2020 §2.1; kb/excerpts/kaplan2020#sec-2-1-definitions]` is

$$C \approx 6 N D \tag{1}$$

(forward pass $\approx 2N$ FLOPs/token; backward pass $\approx 4N$;
sum $\approx 6N$ per token, times $D$ tokens). The *compute-optimal*
allocation problem is

$$\min_{N, D} L(N, D) \quad \text{subject to} \quad 6 N D = C, \tag{2}$$

where $L(N, D)$ is the expected validation loss of the model that
results from the chosen $N, D$ (training to the natural endpoint of an
appropriate LR schedule).

### 1.2 The Chinchilla parametric loss model

Hoffmann et al. propose the functional form
`[hoffmann2022-chinchilla §3.3 Eq.(2); kb/excerpts/hoffmann2022-chinchilla#sec-3-3]`:

$$\hat{L}(N, D) \;=\; E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}, \tag{3}$$

with five fitted constants $(A, B, E, \alpha, \beta)$, fit by minimizing
the Huber loss between predicted and observed log-loss across all 400+
training runs:

$$\min_{A, B, E, \alpha, \beta} \;\sum_{\text{Runs } i} \mathrm{Huber}_\delta\!\big(\log \hat{L}(N_i, D_i) - \log L_i\big), \quad \delta = 10^{-3}. \tag{4}$$

`[hoffmann2022-chinchilla §3.3 Eq.(3); kb/excerpts/hoffmann2022-chinchilla#sec-3-3]`.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $N$ | Model parameter count (Hoffmann reports total parameters; Kaplan reports non-embedding) |
| $D$ | Training tokens |
| $C$ | Training FLOPs |
| $E$ | Irreducible loss (entropy of natural text on the eval distribution) |
| $A/N^\alpha$ | Capacity-limited residual: a perfectly-trained $N$-parameter model still falls short of $E$ |
| $B/D^\beta$ | Optimization-limited residual: finite training steps over a finite sample of the data |
| $\alpha, \beta$ | Power-law exponents fit from the data |

### 1.3 The efficient frontier — closed form

Minimizing (3) under the constraint (1) yields the compute-optimal
$N_{opt}, D_{opt}$ in closed form
`[hoffmann2022-chinchilla §3.3 Eq.(4); kb/excerpts/hoffmann2022-chinchilla#sec-3-3]`:

$$N_{opt}(C) = G\!\left(\frac{C}{6}\right)^a, \quad D_{opt}(C) = G^{-1}\!\left(\frac{C}{6}\right)^b, \tag{5}$$

with
$$G = \left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}, \quad a = \frac{\beta}{\alpha+\beta}, \quad b = \frac{\alpha}{\alpha+\beta}.$$

The exponents $a, b$ on $C$ satisfy $a + b = 1$ identically; the
empirical question is the *split* between them.

## 2. Mechanism — three independent estimation approaches

Hoffmann et al. estimate $a, b$ three different ways and find them all
agree closely
`[hoffmann2022-chinchilla §3 Table 2; kb/excerpts/hoffmann2022-chinchilla#sec-3-table-2]`:

| Approach | $a$ ($N_{opt} \propto C^a$) | $b$ ($D_{opt} \propto C^b$) |
|---|---|---|
| 1. Minimum over training curves | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
| 2. IsoFLOP profiles | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
| 3. Parametric $\hat{L}$ fit | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
| **Kaplan et al. 2020** | **0.73** | **0.27** |

The bracketed numbers are 10th/90th-percentile bootstrap intervals.

### 2.1 Approach 1 — fix $N$, sweep $D$

For each parameter count $N \in \{75M, 250M, 500M, 1B, 2.5B, 5B, 10B\}$,
train 4 models with cosine LR decay over 4 different token horizons
(spanning $16\times$). At 1500 log-spaced FLOP values, find the
$(N, D)$ pair achieving lowest loss. Fit $N_{opt} \propto C^{a}$,
$D_{opt} \propto C^{b}$
`[hoffmann2022-chinchilla §3.1; kb/excerpts/hoffmann2022-chinchilla#sec-3-1]`.

**Result: $a = 0.50$, $b = 0.50$.**

### 2.2 Approach 2 — IsoFLOP profiles

For each of 9 fixed FLOP budgets $C \in [6\!\times\!10^{18}, 3\!\times\!10^{21}]$,
sweep model size and read off the parabola minimum to get
$N_{opt}(C)$. Fit power laws across the 9 minima
`[hoffmann2022-chinchilla §3.2; kb/excerpts/hoffmann2022-chinchilla#sec-3-2]`.

**Result: $a = 0.49$, $b = 0.51$.**

### 2.3 Approach 3 — fit the parametric loss $\hat{L}$

Fit Eq. (3) to all 400 runs simultaneously using L-BFGS on the Huber
loss; derive $a, b$ from the closed form (5)
`[hoffmann2022-chinchilla §3.3; kb/excerpts/hoffmann2022-chinchilla#sec-3-3]`.

**Result: $a = 0.46$, $b = 0.54$.**

The three approaches disagree slightly (Approach 3 favors a touch more
data) but all reject the Kaplan exponents at high confidence.

## 3. The headline result — Chinchilla 70B vs Gopher 280B

Both Chinchilla and Gopher were trained for the **same** total FLOPs
($\approx 5.76 \times 10^{23}$). The difference is the $N : D$ split
`[hoffmann2022-chinchilla §4.1; kb/excerpts/hoffmann2022-chinchilla#sec-4-1]`:

| | Gopher | Chinchilla |
|---|---|---|
| Parameters | 280B | 70B (≈ 4× smaller) |
| Tokens | 300B | 1.4T (≈ 4.7× more) |
| Tokens / parameter | 1.07 | 20.0 |
| Layers | 80 | 80 |
| Heads | 128 | 64 |
| $d_{\text{model}}$ | 16,384 | 8,192 |
| Max LR | $4 \times 10^{-5}$ | $1 \times 10^{-4}$ |
| Batch size | 3M → 6M | 1.5M → 3M |

Same depth, half the width, doubled LR, halved batch. Chinchilla beats
Gopher uniformly: MMLU 67.5% vs 60.0%, +0.6 pp average across BIG-bench,
better on every Pile subset, etc.
`[hoffmann2022-chinchilla §4.2; kb/excerpts/hoffmann2022-chinchilla#sec-4-2]`.

## 4. The "tokens-per-parameter ≈ 20" rule

From Table 3 of Hoffmann et al.
`[hoffmann2022-chinchilla §3 Table 3; kb/excerpts/hoffmann2022-chinchilla#sec-3-table-3]`,
the Approach-1 projections give:

| $N$ | $D_{opt}$ | $D_{opt}/N$ |
|---|---|---|
| 400M | 8.0B | 20.0 |
| 1B | 20.2B | 20.2 |
| 10B | 205.1B | 20.5 |
| 67B (≈ Chinchilla) | 1.5T | 22.4 |
| 175B (≈ GPT-3) | 3.7T | 21.1 |
| 280B (≈ Gopher) | 5.9T | 21.1 |
| 520B (≈ MT-NLG) | 11.0T | 21.2 |
| 1T | 21.2T | 21.2 |
| 10T | 216.2T | 21.6 |

Hence the well-known rule of thumb: **at compute optimum, tokens ≈ 20 ×
parameters** (with a slow drift toward more data at very large scale).

[INTUITION] Why did Kaplan get 0.73/0.27? The fitting setup used a fixed
LR-decay horizon for all models (decay to a fixed token count, e.g.,
130B), regardless of how many tokens each individual run actually saw
`[hoffmann2022-chinchilla §2; kb/excerpts/hoffmann2022-chinchilla#sec-2-disagree-kaplan]`.
For runs with $D' \ll 130B$, the LR was still high at the end, so the
final loss was a worse-than-true estimate of "this model trained to its
optimal endpoint at $D' $ tokens." This systematically *over-penalized
short-data* training, biasing the fit toward "more model, less data."
Hoffmann et al. matched LR schedule length to actual token count and
the bias vanished.

## 5. Where Chinchilla applies — and where it doesn't

### 5.1 Inference-cost-aware scaling — the Llama era

Chinchilla is *pre-training-compute optimal*. If you pre-train once and
serve to billions of users, total cost is dominated by **inference**
FLOPs, not training FLOPs. A smaller model that's slightly under-Chinchilla
on its pre-training loss curve can deliver enormously lower per-token
inference cost.

Llama 3 8B was trained on ≈ 15T tokens
`[meta-llama3 (Meta tech report); kb/index/papers.json#meta-llama3]`,
giving a tokens-per-param ratio of ≈ 1875 — roughly **94× the
Chinchilla optimum**. This is not a refutation of Chinchilla; it's a
different optimization. The total cost minimized by Llama 3 is

$$\text{Total} = C_{\text{train}} + N_{\text{tokens-served}} \cdot C_{\text{inf-per-token}}.$$

When $N_{\text{tokens-served}}$ is large, the second term dominates and
the optimum shifts toward smaller $N$ (lower $C_{\text{inf-per-token}}$)
with proportionally more $D$ (which Chinchilla's loss function
predicts is still fine, just sub-optimal on the $C_{\text{train}}$
margin). [CONTRADICTION] Some 2024 community discussion frames this as
"Chinchilla is wrong"; the more accurate framing is "Chinchilla solves
a different objective than the one industry actually optimizes."

### 5.2 Precision-aware scaling adds another wrinkle

Kumar et al. 2024
`[scaling-laws-precision-2024; kb/index/papers.json#scaling-laws-precision-2024]`
show that as training-data $D$ grows, models become **harder to
quantize post hoc**, with quantization-degradation increasing
super-linearly past a threshold. Token-heavy over-training (Llama-3
style) trades pre-training-compute efficiency for both inference-cost
efficiency *and* increased post-quantization cost. The full
multi-objective frontier is still being mapped.

### 5.3 Data-constrained regime

Muennighoff et al. 2023 (NeurIPS) extend Chinchilla to the case where
the unique-token corpus is finite and you must repeat data. Roughly:
each token-epoch beyond ≈ 4 contributes diminishing-returns loss
improvement, and the compute-optimal point shifts back toward larger
models when tokens are scarce. Status: still in the canonical post-
Chinchilla portfolio.

## 6. Frontier and open questions (as of 2026-05)

- **The Chinchilla "trap."** A model trained exactly at Chinchilla
  optimum is both (a) the cheapest to *pre-train*, and (b) often too
  large to serve cheaply. "Trap" is a community term for the
  realization that compute-optimal pre-training is not deployment-
  optimal. See `kb/notes/scaling/scaling-frontier.md`.
- **Empirical re-fits at frontier scale.** As models cross the 100B–1T
  param range, mild curvature in the fit of (3) becomes important. The
  closed-form (5) implicitly assumes the power-law functional form
  extrapolates; Hoffmann et al. note this caveat in §3 ("future work
  may want to include potential curvature in this relationship for
  large model sizes"). Recent re-fits with more data points give
  similar but not identical exponents.
- **Architecture-conditional Chinchilla.** Chinchilla was fit on a
  specific decoder-only Transformer family. Whether MoE, hybrid SSM
  (Jamba), Mamba-2, or Multi-Token-Prediction architectures have the
  same exponents is open. DeepSeek V3's 14.8T tokens on 671B-A37B
  (active params: 37B) implies a tokens-per-active-param of ≈ 400 —
  far over-trained by dense Chinchilla standards. Whether MoE shifts
  the optimum is an active question.
- **The composition of $D$.** The original Chinchilla paper used
  MassiveText with a near-fixed mixture. Data Mixing Laws (Liu et al.,
  ICLR 2025; `data-mixing-laws-2024`) show that the *composition* of
  $D$ has a predictable functional effect on loss; this changes the
  optimal $N : D$ point as a function of mixture quality.
- **Compute-optimal vs. perplexity-optimal vs. downstream-optimal.**
  Chinchilla optimizes pre-training cross-entropy. Whether the same
  optimum holds for downstream task accuracy, MMLU, reasoning
  benchmarks, etc., is empirically not always — the relationship from
  loss to accuracy is task-dependent. See `kb/notes/evaluation/eval-methodology.md`.

## 7. See also

- `kb/notes/scaling/kaplan-laws.md` — the predecessor paper Chinchilla
  corrects.
- `kb/notes/scaling/scaling-frontier.md` — what frontier labs actually
  do (over-train past Chinchilla for inference cost).
- `kb/notes/scaling/inference-time-compute-scaling.md` — the second
  scaling axis that's largely *displaced* the Chinchilla frame for
  reasoning models.
- `kb/notes/training/pre-training-data.md` — how the *quality* of $D$
  interacts with the $N : D$ frontier.
- `kb/notes/training/optimization.md` — Chinchilla used AdamW, found
  this slightly improves over Adam; LR schedule must match token count.
