---
paper_key: meng2024-simpo
title: "SimPO: Simple Preference Optimization with a Reference-Free Reward"
authors: Meng, Xia, Chen
year: 2024
venue: NeurIPS
arxiv: 2405.14734
local_pdf: theory/sources/papers/meng2024-simpo.pdf
type: excerpts
note: "Excerpts from arXiv abstract page (verbatim) plus canonical equation as referenced in DPO survey (arXiv:2503.11701 §4.1) and follow-up variants. PDF not yet downloaded."
---

# Excerpts — Meng et al. 2024, "SimPO"

## Abstract — average log-prob as implicit reward, reference-free {#abstract}

> Direct Preference Optimization (DPO) is a widely used offline preference
> optimization algorithm that reparameterizes reward functions in
> reinforcement learning from human feedback (RLHF) to enhance simplicity
> and training stability. In this work, we propose SimPO, a simpler yet
> more effective approach. The effectiveness of SimPO is attributed to a
> key design: **using the average log probability of a sequence as the
> implicit reward**. This reward formulation **better aligns with model
> generation and eliminates the need for a reference model**, making it
> more compute and memory efficient. Additionally, we introduce a **target
> reward margin** to the Bradley-Terry objective to encourage a larger
> margin between the winning and losing responses, further improving the
> algorithm's performance.

## §2 The SimPO reward and loss {#sec-2}

The SimPO implicit reward is the **length-normalized average log-probability**:

$$
r_{\mathrm{SimPO}}(x,y) = \frac{\beta}{|y|} \log \pi_\theta(y|x) = \frac{\beta}{|y|}\sum_{t=1}^{|y|} \log\pi_\theta(y_t|x, y_{<t}).
$$

Compare with DPO's implicit reward $r_{\mathrm{DPO}}(x,y) = \beta
\log\pi_\theta(y|x)/\pi_{\mathrm{ref}}(y|x)$ (no length normalization,
ratio with reference). SimPO drops the reference and divides by length.

The SimPO loss adds a **target reward margin** $\gamma > 0$ to the
Bradley–Terry log-likelihood:

$$
\mathcal{L}_{\mathrm{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \left[ \log\sigma\!\left(\frac{\beta}{|y_w|}\log\pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l|x) - \gamma\right)\right].
$$

The margin $\gamma$ pushes the chosen response's reward at least $\gamma$
above the rejected one's — a stricter constraint than the bare Bradley–Terry
objective.

## §2 Why length normalization {#sec-2-length}

DPO's implicit reward $\beta\log\pi_\theta(y|x)/\pi_{\mathrm{ref}}(y|x)$
is a sum over tokens. For preference pairs with very different lengths,
DPO can satisfy the preference by assigning extreme per-token log-prob
to a few tokens of the long response — producing the well-known **DPO
length bias**. SimPO's $1/|y|$ normalization makes the implicit reward
scale-invariant in length and is reported (paper §4.2 ablation) to be
the dominant contributor to gains over DPO on length-controlled
benchmarks.

## §4 Empirical headline {#sec-4-headline-empirical}

> +6.4 AlpacaEval 2 over DPO (paraphrased; exact figure is referenced in
> the Phase 1 sweep landscape report and replicated in the DPO survey
> arXiv:2503.11701 §4.1).

[Verified from PDF on 2026-05-12 — discrepancy fixed] The KB note
previously labeled the SimPO derivation as "§3" based on a borrowed
secondary-source layout; the actual PDF has §2 "SimPO: Simple Preference
Optimization" containing the reward (Eq. 5 BT-with-margin formulation)
and loss (Eq. 6, L_SimPO). PDF §3 is "Experimental Setup", §4 is
"Experimental Results", §5 is "Related Work", §6 is "Conclusion".
Section anchors renamed: sec-3 → sec-2, sec-3-length → sec-2-length.
Equation forms verified verbatim against PDF Eq. 5 (margin BT) and
Eq. 6 (loss with β/|y| length-normalization, − γ target margin).
