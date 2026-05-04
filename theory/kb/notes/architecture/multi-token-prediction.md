---
topic: architecture/multi-token-prediction
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - deepseek-v3
  - meta-llama4
---

# Multi-token prediction (MTP)

**Status:** stub. Drafted from Phase 1 landscape sweep; needs full Phase 2
treatment.

## What this topic covers

Multi-Token Prediction is a training-objective and architectural feature
introduced into production LLMs by **DeepSeek-V3** (Dec 2024). Instead
of training the model to predict only token $t+1$ from context up to
token $t$, MTP adds **auxiliary heads** that predict tokens $t+2,
t+3, \ldots, t+k$ at each position. At inference time, MTP heads can
be either (a) **discarded** (use only the next-token head — main-line
generation), or (b) **kept** for **speculative decoding** (the MTP
heads predict draft tokens that the main head verifies in parallel).

MTP is **architecturally distinct from speculative decoding** even
though it can be used as a draft model: speculative decoding
historically uses a smaller separate model as the drafter, while MTP
uses heads tied to the same backbone. As of mid-2026, MTP appears in
DeepSeek-V3, possibly LLaMA 4 (verify), and the technique is being
ported to other open MoE models
`[Phase 1 sweep §3 transformer-overview/v0→v1 changelog]`.

## Primary sources to read (in order)

1. `deepseek-v3` — *DeepSeek-V3 Technical Report* (2412.19437) §2.3 —
   MTP as a training objective; architectural details; the "main token
   prediction + sequential MTP modules" pattern.
2. `meta-llama4` — *Llama 4 Herd* (2601.11659) — verify whether
   LLaMA 4 also uses MTP and if so how.
3. *Better & Faster Large Language Models via Multi-token Prediction*
   (Gloeckle et al. 2024, arXiv 2404.19737) — Meta's earlier MTP
   research paper; preceded DeepSeek-V3's adoption. Worth reading as
   the foundational source for the technique.

## Key claims to ground (Phase 2 todo)

- MTP architecture: the main backbone produces hidden state $\boldsymbol{h}_t^L$
  at depth $L$. MTP adds $k$ "MTP modules": each takes $\boldsymbol{h}_t^L$
  as input and produces a prediction for token $t + i$ for
  $i \in \{2, \ldots, k\}$. The modules can be implemented as small
  Transformer blocks chained on top of the backbone.
- **DeepSeek-V3's specific design**: $k = 1$ MTP module (so it
  predicts $t+1$ via the main head and $t+2$ via the MTP head). The
  MTP module receives both $\boldsymbol{h}_t^L$ and the token
  embedding for $t+1$ (the previously-predicted token), so it
  effectively performs one extra forward step.
- Training objective: standard cross-entropy on the main head, plus
  auxiliary cross-entropy on each MTP head, weighted with a
  coefficient $\lambda$ (DeepSeek-V3 uses $\lambda$ schedule that
  decays). The MTP loss is auxiliary; it shapes the backbone to
  produce hidden states useful for multi-step prediction.
- Inference modes:
  - **Without MTP (default)**: just use the main next-token head.
  - **With MTP for self-speculative-decoding**: the MTP heads emit
    draft tokens; the main head verifies them in parallel. Reported
    1.8× speedup at acceptance rate ~85% in DeepSeek-V3's serving.
- MTP vs speculative decoding (`kb/notes/inference/speculative-decoding.md`):
  classical speculative decoding uses a separate small "drafter"
  model. MTP is an integrated drafter using the same backbone — no
  separate model to maintain, but ties the drafter quality to the
  backbone's training objective.
- Why MTP improves quality, not just speed: the auxiliary loss forces
  the backbone to encode information about $t+2$ in $\boldsymbol{h}_t^L$,
  which presumably tightens the representation. Reported in
  DeepSeek-V3 ablations as a small but consistent quality lift.
- Interaction with reasoning architectures: MTP's "predict multiple
  tokens ahead" is conceptually close to reasoning's "plan ahead" —
  but the mechanisms are different. MTP shapes hidden state; reasoning
  emits explicit tokens. Whether they synergize at training is an open
  question.

## Open questions

- Optimal $k$ (how many tokens ahead to predict) at various scales.
  DeepSeek-V3 uses $k = 2$ (1 main + 1 MTP); is $k = 4$ or $k = 8$
  better at frontier scale?
- Does MTP help with reasoning models specifically? The DeepSeek-V3 →
  R1 transition retains MTP; the R1 paper is light on detail about
  whether MTP heads are still used during reasoning.
- MTP × MoE interaction: do MTP heads benefit from MoE-style
  expert routing? Untested.

## Related notes

- `kb/notes/architecture/transformer-overview.md` — where MTP heads
  sit (after the final block, parallel to the main LM head).
- `kb/notes/architecture/reasoning-architectures.md` — DeepSeek-V3 →
  R1 retain MTP; possible synergy.
- `kb/notes/inference/speculative-decoding.md` — MTP is an integrated
  drafter; relationship to EAGLE / Medusa / external-drafter
  speculative decoding.
- `kb/notes/training/optimization.md` — auxiliary-loss-weight
  scheduling for MTP.
