---
topic: scaling/kaplan-laws
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - kaplan2020
secondary_sources:
  - hoffmann2022-chinchilla   # supersedes Kaplan's compute-allocation rule
related_topics:
  - scaling/chinchilla
  - scaling/scaling-frontier
  - architecture/transformer-overview
---

# Kaplan scaling laws — original power-law functional forms

Kaplan et al. (2020) is the foundational empirical study establishing
that LLM cross-entropy loss decreases as a **power law** in three
independent inputs: model parameters $N$, dataset size $D$, and
training compute $C$. The functional forms in this paper are still
broadly used; the **specific compute-allocation rule** ($N \propto
C^{0.73}$, lots of model + little data) was overturned by Chinchilla
(Hoffmann et al., 2022). This note treats Kaplan as historical and
foundational, not as current guidance.

[CONTRADICTION] On compute-optimal allocation: Kaplan's rule
$N \propto C^{0.73}$, $D \propto C^{0.27}$ (10× compute → 5.5× model,
1.8× data) is **superseded** by Chinchilla's
$N \propto C^{\sim 0.50}$, $D \propto C^{\sim 0.50}$ (10× compute →
3.2× model, 3.2× data). The disagreement is traceable to a methodological
flaw in Kaplan: a fixed cosine-LR schedule for all model sizes biased
the loss estimates of short-data runs upward, falsely making "more
model, less data" look better. See §4 below and
`[hoffmann2022-chinchilla §2; kb/excerpts/hoffmann2022-chinchilla#sec-2-disagree-kaplan]`.

## 1. Formal definitions

### 1.1 The three single-axis power laws

For a Transformer language model trained on autoregressive
cross-entropy loss, when performance is bottlenecked by *only* one of
$N$, $D$, $C_{\min}$, the loss obeys a power law
`[kaplan2020 §1.2 Eq.(1.1)–(1.3); kb/excerpts/kaplan2020#sec-1-2-equations]`:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076, \;\; N_c \approx 8.8 \times 10^{13} \tag{1.1}$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095, \;\; D_c \approx 5.4 \times 10^{13} \tag{1.2}$$

$$L(C_{\min}) = \left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}}, \quad \alpha_C^{\min} \approx 0.050, \;\; C_c^{\min} \approx 3.1 \times 10^8\,\text{PF-days} \tag{1.3}$$

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $L$ | cross-entropy loss in nats per token (averaged over the context) |
| $N$ | **non-embedding** parameters — Kaplan excludes vocab and positional embeddings |
| $D$ | dataset size in tokens |
| $C_{\min}$ | "minimum compute" — what would be used at a batch size much smaller than the critical batch $B_{\text{crit}}$ |
| $C \approx 6 N B S$ | total non-embedding training compute, for $B$ = batch (tokens), $S$ = steps |
| $\alpha_N, \alpha_D, \alpha_C^{\min}$ | power-law exponents (dimensionless) |
| $N_c, D_c, C_c^{\min}$ | scale constants; depend on tokenizer/dataset, "do not have a fundamental meaning" `[kaplan2020 §1.2; kb/excerpts/kaplan2020#sec-1-2-equations]` |

The exponents are **small**: $\alpha_N \approx 0.076$ means doubling
parameters reduces loss by a factor $2^{-0.076} \approx 0.949$, i.e.
~5% relative loss improvement
`[kaplan2020 §1.2; kb/excerpts/kaplan2020#sec-1-2-equations]`.

### 1.2 The joint $L(N, D)$ form

Combining (1.1) and (1.2) into a single equation governing
simultaneous $N, D$ dependence (and the degree of overfitting):

$$L(N, D) = \left[ \left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D} \right]^{\alpha_D} \tag{1.5}$$

`[kaplan2020 §1.2 Eq.(1.5); kb/excerpts/kaplan2020#sec-1-2-joint]`.

The implication: to avoid the overfitting penalty, $D$ should grow as
$D \propto N^{\alpha_N/\alpha_D} \approx N^{0.74}$ — i.e., **sublinearly
in $N$**. Doubling $N$ requires only $\approx 1.67\times$ more data.
This sublinearity is the core technical reason Kaplan's compute-
optimal rule favors large $N$ over large $D$.

[CONTRADICTION] Hoffmann et al. (2022) re-fit with controlled LR
schedule and arrive at $D \propto N$ (linear, not sublinear) at the
compute-optimal frontier — see
`kb/notes/scaling/chinchilla.md` §1.3 and §2.

### 1.3 The compute approximation $C \approx 6 N D$

For non-embedding parameters $N$, the standard FLOP count per token is
`[kaplan2020 §2.1; kb/excerpts/kaplan2020#sec-2-1-definitions]`:

- Forward pass: $C_{\text{forward}} \approx 2 N + 2 n_{\text{layer}} n_{\text{ctx}} d_{\text{attn}}$
- For $d_{\text{model}} \gg n_{\text{ctx}}/12$: drop the
  context-quadratic term to $C_{\text{forward}} \approx 2 N$.
- Backward $\approx 2 \times$ forward, so total $\approx 6 N$ FLOPs per
  training token. Across $D$ tokens: $C \approx 6 N D$.

This is the formula every subsequent scaling paper inherits. It
ignores attention's $N^2$-in-context dependence (valid for context-bounded
training); modern long-context regimes need a correction.

## 2. Mechanism — what Kaplan actually measured

### 2.1 Experimental setup

- Train decoder-only Transformer LMs on **WebText2** (Reddit-link
  scrape, GPT-2 tokenizer, 50,257 BPE vocab) over a 1024-token context.
- Vary $N$ from 768 to $1.5 \times 10^9$ non-embedding parameters; vary
  $D$ from 22M to 23B tokens.
- Adam optimizer (Adafactor for $N > 1B$); cosine LR schedule with
  3000-step linear warmup, decaying to zero.
- Fix the LR schedule horizon at $S = 2.5 \times 10^5$ steps for all
  models, batch size 512 sequences × 1024 tokens
  `[kaplan2020 §2.2; kb/excerpts/kaplan2020#sec-2-1-definitions]`.

The fixed LR schedule is the methodological choice that biased the
compute-optimal allocation result — see §4 below.

### 2.2 Five "summary findings"

`[kaplan2020 §1.1; kb/excerpts/kaplan2020#sec-1-1-bullets]`:

1. **Performance depends strongly on scale, weakly on shape.** Within
   reasonable limits, depth-vs-width and number-of-heads have minimal
   effects compared to $N$, $D$, $C$.
2. **Smooth power laws.** Trends span 6+ orders of magnitude with no
   visible deviation at the upper end (in 2020 data; data was 6 orders
   of magnitude smaller than current frontier).
3. **Universality of overfitting.** $N$ and $D$ must scale in tandem
   (the $N^{0.74}/D$ ratio).
4. **Sample efficiency improves with size.** Larger models reach a
   given loss with fewer optimization steps.
5. **Convergence is inefficient.** With a fixed compute budget, the
   loss-minimizing strategy is to train a *very large* model and stop
   *significantly short* of convergence.

Findings 1–3 are still considered correct (with refinements). Finding
5 is the controversial one — Chinchilla showed that models were *not
too small but rather too data-starved*.

### 2.3 The compute-allocation prescription (the controversial part)

`[kaplan2020 §1.2 Eq.(1.7)–(1.8); kb/excerpts/kaplan2020#sec-1-2-compute-allocation]`:

$$N \propto C^{\alpha_C^{\min}/\alpha_N}, \quad B \propto C^{\alpha_C^{\min}/\alpha_B}, \quad S \propto C^{\alpha_C^{\min}/\alpha_S}, \quad D = B \cdot S \tag{1.7}$$

Plugging in measured values:

$$N \propto C_{\min}^{0.73}, \quad B \propto C_{\min}^{0.24}, \quad S \propto C_{\min}^{0.03}, \quad D = B \cdot S \propto C_{\min}^{0.27}$$

Reading: **a 10× increase in compute should buy a ~5.4× larger model,
~1.7× larger batch, only 1.07× more steps** — almost no extra training
time, dramatically more parameters. Equivalently, $D \propto C^{0.27}$
implies **data grows very slowly** with compute.

This is exactly the rule the GPT-3 generation followed: 175B
parameters, 300B tokens (tokens/param ≈ 1.7).

## 3. Other findings still considered correct

### 3.1 Architecture-shape weakness

Within wide ranges of $(d_{\text{model}}, n_{\text{layer}},
n_{\text{heads}}, d_{\text{ff}})$ that hold $N$ constant, performance
varies by < 1% on the $L(N)$ frontier
`[kaplan2020 §1.1; kb/excerpts/kaplan2020#sec-1-1-bullets]`. This
empirically justifies the convention of reporting "scaling laws as a
function of $N$" without micro-tuning architecture.

### 3.2 Critical batch size

The critical batch size $B_{\text{crit}}$ — the largest batch at which
training is still time-efficient (roughly halving steps without
proportional compute increase) — itself follows a power law in the
loss `[kaplan2020 §5.1]`:

$$B_{\text{crit}}(L) = \frac{B_*}{L^{1/\alpha_B}}, \quad B_* \approx 2 \times 10^8\text{ tokens}, \;\; \alpha_B \approx 0.21$$

Implication: as loss decreases (model improves), the optimal batch
grows. This is operationally important for distributed training — see
`kb/notes/training/distributed-training.md`.

### 3.3 Sample efficiency

For a fixed loss target, the number of optimization steps follows
roughly $S \propto N^{-\alpha_N/\alpha_S} \approx N^{-0.10}$, with
$\alpha_S \approx 0.76$. Larger models reach a given loss in fewer
steps. This is the empirical basis for the "pre-train very large,
stop early" intuition that Chinchilla nuanced.

## 4. Why Kaplan's allocation rule was wrong — the LR-horizon bug

The methodological core of Hoffmann et al. (2022)'s correction:

Kaplan trained every model with a cosine LR decay reaching zero at a
**fixed** total token count (or step count) for the entire study,
regardless of how many tokens any individual run actually consumed.
Chinchilla's authors observed:

> For a fixed learning rate cosine schedule to 130B tokens, the
> intermediate loss estimates (for $D' \ll 130B$) are therefore
> overestimates of the loss of a model trained with a schedule length
> matching $D'$.
> `[hoffmann2022-chinchilla §2; kb/excerpts/hoffmann2022-chinchilla#sec-2-disagree-kaplan]`

In words: a model trained to $D'$ tokens with an LR schedule that
*decays to zero* at $D'$ achieves a *lower* loss than the same model
trained to $D'$ tokens with an LR schedule that decays to zero at
$130B \gg D'$ (because at $D'$ the LR is still high, the model is
mid-training). Kaplan inferred his $L(D)$ scaling from the latter,
biased upward, runs.

When Hoffmann et al. *matched the LR horizon to the actual training
length* across their 400+ runs, the apparent advantage of "more model,
less data" disappeared. The exponents flipped from (0.73, 0.27) to
roughly (0.50, 0.50).

[INTUITION] Kaplan's setup was, essentially: take a partially-trained
model and ask "how much loss is left?" That penalized smaller-$D$ runs
because they weren't yet at their LR-decay endpoint. The lesson is
that **what you measure is the loss at training endpoint, where
endpoint = end of LR schedule**. Conflating "tokens consumed" with
"training endpoint" was the trap.

## 5. Notation note — embedding vs total parameters

Kaplan's $N$ excludes embedding and positional-encoding parameters
`[kaplan2020 §1.3; kb/excerpts/kaplan2020#sec-1-3-notation]`. Hoffmann
et al. (Chinchilla) and most subsequent papers report **total**
parameters. For modern models with vocab ≈ 32K–128K and
$d_{\text{model}}$ ≈ 4K–16K, the embedding contribution is on the
order of 0.5%–5% of total — small but non-negligible at high precision.
When comparing exponents and constants across papers, check the
convention.

## 6. Enduring contributions vs. superseded recommendations

**Still authoritative:**
- The **functional forms** (1.1)–(1.5).
- The **compute formula** $C \approx 6 N D$.
- The "performance depends weakly on shape" finding.
- The critical-batch-size power law.

**Superseded:**
- The compute-allocation exponents $(0.73, 0.27)$.
- The recommendation to "train very large models, stop short of
  convergence" — the pre-Chinchilla GPT-3 era took this literally and
  ended up with under-trained 175B models that Chinchilla (70B, 1.4T)
  beat at the same FLOP budget.

**Still unresolved at frontier scale:**
- Whether the power-law functional form (1.5) holds at 100B+ params
  and 10T+ tokens — there's mild curvature in modern data.
- How the exponents shift for MoE, hybrid, and multi-token-prediction
  architectures.
- How precision-aware loss (Kumar et al., 2024) interacts with the
  Kaplan/Chinchilla functional form.

## 7. See also

- `kb/notes/scaling/chinchilla.md` — the corrected compute-optimal rule.
- `kb/notes/scaling/scaling-frontier.md` — current industry practice
  (over-train past Chinchilla for inference economics).
- `kb/notes/training/optimization.md` — LR schedules, batch-size
  scaling, AdamW; the LR-horizon issue Kaplan tripped on.
- `kb/notes/training/distributed-training.md` — critical batch size as
  a guide to data-parallel scaling.
