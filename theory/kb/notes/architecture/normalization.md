---
topic: architecture/normalization
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - zhang2019  # RMSNorm
  - xiong2020  # Pre-LN paper
  - peri-ln2025
  - olmo2  # QK-Norm in production
---

# Normalization

**Status:** stub. Drafted from Phase 1 landscape sweep; needs full Phase 2
treatment. The zhang2019 (RMSNorm) and xiong2020 (Pre-LN) PDFs are
already in `theory/sources/papers/`; promoting to draft is a low-
effort follow-up.

## What this topic covers

Layer Normalization (LN) and its lighter-weight cousin **RMSNorm**
(Zhang & Sennrich 2019) stabilize gradients in deep Transformers.
Placement matters: **Pre-LN** (Xiong et al. 2020) was the standard for
2020–2024 (LN before each sublayer); 2025 brought **Peri-LN** (norm
both before and after, ICML 2025) — adopted by Gemma 3 and OLMo 2 —
and **Post-LN rehabilitation** (Jan 2026, arXiv 2601.19895). **QK-Norm**
(applied to Q and K before attention) is increasingly common in
production for training stability (OLMo 2, LLaMA 3.1+).
`[Phase 1 sweep §2 normalization]`

## Primary sources to read (in order)

1. `zhang2019` — *Root Mean Square Layer Normalization* — RMSNorm:
   drop the mean-subtraction from LN; only rescale by RMS. Already in
   `theory/sources/papers/zhang2019_rmsnorm.pdf`.
2. `xiong2020` — *On Layer Normalization in the Transformer
   Architecture* — Pre-LN vs Post-LN training dynamics. Already in
   `theory/sources/papers/xiong2020_layer_norm.pdf`.
3. `peri-ln2025` — *Peri-LN: Revisiting Normalization Layer in the
   Transformer Architecture* (2502.02732) — ICML 2025, peri-LN
   placement.
4. *Post-LayerNorm Is Back: Stable, Expressive, and Deep* (2601.19895)
   — Jan 2026 rehabilitation.
5. `olmo2` — production example of Peri-LN + QK-Norm.
6. *Gated Removal of Normalization in Transformers* (2602.10408) —
   norm-free with gating.
7. *Geometric Interpretation of LN vs RMSNorm* (2409.12951) —
   theoretical analysis.

## Key claims to ground (Phase 2 todo)

- LayerNorm formula: $\mathrm{LN}(\boldsymbol{x}) = \frac{\boldsymbol{x} - \mu}{\sigma} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$
  with mean and variance computed across the feature dimension.
- RMSNorm formula: $\mathrm{RMSNorm}(\boldsymbol{x}) = \frac{\boldsymbol{x}}{\mathrm{RMS}(\boldsymbol{x})} \odot \boldsymbol{\gamma}$,
  $\mathrm{RMS}(\boldsymbol{x}) = \sqrt{\frac{1}{d} \sum x_i^2}$. Drops
  the centering step (mean subtraction); empirically as good as LN
  with ~7–64% fewer FLOPs.
- Pre-LN vs Post-LN: Post-LN is the original Vaswani 2017 placement
  (LN after each sublayer + residual); Pre-LN moves LN before each
  sublayer. Pre-LN trains more stably without learning-rate warmup.
- Peri-LN: $\mathrm{LN}_\text{pre} \to \mathrm{Sublayer} \to \mathrm{LN}_\text{post} \to + \mathrm{residual}$.
  Empirically more stable than Pre-LN for very deep networks; adopted
  by Gemma 3, OLMo 2.
- QK-Norm: $\mathrm{LN}(Q), \mathrm{LN}(K)$ before the attention dot
  product. Stabilizes training in long-context regimes by preventing
  attention-logit explosion. Used by OLMo 2 and LLaMA 3.1+.
- Why centering can be omitted (RMSNorm's contribution): empirical
  result + geometric analysis suggesting LN's $\boldsymbol{\beta}$
  bias and mean-subtraction add negligible representational capacity
  in practice.
- Stale-prior correction (per Phase 1 sweep §3): "Pre-LN is standard;
  Post-LN is unstable" → 2025–2026 diversity: Peri-LN gaining,
  Post-LN rehabilitated, RMSNorm-only architectures explored.

## Related notes

- `kb/notes/architecture/transformer-overview.md` — placement of LN /
  RMSNorm in the block.
- `kb/notes/architecture/attention-mechanism.md` — QK-Norm sits inside
  the attention block.
- `kb/notes/training/mixed-precision-and-stability.md` — normalization
  is one of several training-stability levers; DeepNorm, ScaleNorm,
  scaling-aware init also belong there.
