---
topic: interpretability/lens-techniques
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - belrose2023-tuned-lens
  - nostalgebraist2020-logit-lens   # Tier B (LessWrong post; primary discovery doc)
  - chuang2023-dola
  - logitlens4llms-2025
---

# Lens techniques

**Status:** stub. Drafted from Phase 1 landscape sweep + a partial PDF
read of `belrose2023_tuned-lens.pdf`; needs full Phase 2 treatment.

## What this topic covers

"Lens techniques" are methods for reading out a transformer's
intermediate representations *as if they were final-layer outputs*. The
canonical method is the **logit lens** (nostalgebraist 2020, LessWrong):
take the residual-stream activation $\mathbf{h}^{(\ell)}$ at any layer,
apply the final LayerNorm + unembedding, and read the resulting
distribution over vocabulary tokens. Conceptually this asks "what
would the model's prediction be if we early-exited at layer $\ell$?"
— useful because for many prompts the prediction at intermediate
layers converges monotonically toward the final answer, revealing how
the model "decides" `[INTUITION]`.

The **tuned lens** (Belrose et al. 2023) addresses the logit lens's
failure on models like BLOOM and GPT-Neo: train an affine
transformation per layer (a "translator") that maps mid-layer
activations into the basis the unembedding expects. Lower perplexity
than the raw logit lens, more reliable across model families
`[belrose2023-tuned-lens §1; PDF p.1-2]`.

**DoLa** (Chuang et al. 2023) turns lens analysis into a decoding
strategy: contrast logits at the final layer against logits at a
"premature" earlier layer to amplify factual signal and suppress
hallucination. This is a practical *application* of lens techniques
to inference, not just analysis.

## Primary sources to read (in order)

1. `belrose2023-tuned-lens` — *Eliciting Latent Predictions from
   Transformers with the Tuned Lens* (arXiv 2303.08112) — replaces
   the raw logit lens with per-layer affine "translators", giving
   reliable intermediate readouts. PDF downloaded at
   `theory/sources/papers/belrose2023_tuned-lens.pdf`.
2. `nostalgebraist2020-logit-lens` — original LessWrong post (Tier B
   discovery-only; cite alongside an arxiv source). The technique
   that started the field.
3. `chuang2023-dola` — *DoLa: Decoding by Contrasting Layers Improves
   Factuality in Large Language Models* (arXiv 2309.03883) —
   widely-cited inference-time application.
4. `logitlens4llms-2025` — extends logit lens automation to
   Qwen-2.5, Llama-3.1, and other modern models (arXiv 2503.11667).

## Key claims to ground (Phase 2 todo)

- **Logit lens definition.** $\text{LogitLens}(\mathbf{h}^{(\ell)}) =
  \text{LayerNorm}[\mathbf{h}^{(\ell)}] W_U$, where $W_U$ is the
  unembedding `[belrose2023-tuned-lens §2 Eq.3]`. Should derive the
  full residual decomposition (Eq. 1–2 of belrose2023) showing that
  $\mathcal{M}_{>\ell}(\mathbf{h}_\ell) = \text{LayerNorm}[\mathbf{h}_\ell
  + \sum_{\ell' = \ell}^L F_{\ell'}(\mathbf{h}_{\ell'})] W_U$ — the
  logit lens drops the residual updates from layer $\ell$ onward.
- **Tuned lens definition.** Train $\mathbf{h}_\ell \mapsto A_\ell
  \mathbf{h}_\ell + b_\ell$ with a distillation loss against final-
  layer logits. The "translator" can be composed with $W_U$ to give a
  layer-specific probe.
- **Causal Basis Extraction (CBE).** Belrose et al. introduce CBE to
  identify the residual-stream directions with highest influence on
  the tuned lens output and ablate them — connection to MI feature
  attribution.
- **DoLa decoding.** $\text{logits}_\text{DoLa}(t) =
  \text{logit}^{(L)}(t) - \text{logit}^{(\ell^*)}(t)$ for an
  adaptively-chosen early layer $\ell^*$.
- **Failure modes.** The logit lens fails on BLOOM/GPT-Neo for early
  layers (representational drift across models with different
  pre-training) `[belrose2023-tuned-lens §1, Figure 1]`. The tuned
  lens recovers interpretability there.
- **Cross-modal extension.** Diffusion Steering Lens (2025, arXiv
  2504.13763) applies the framework to vision transformers.

## Position in the methodology family

Lens techniques are the *scalar / vocabulary* readout: they reduce
intermediate states to a vocabulary distribution. **Probing** (next
note) is the *learned* readout: a classifier reads activations for
some property of interest. **SAEs** are the *unsupervised feature*
readout: decompose activations into a sparse basis. The three are
complementary; lens techniques are the cheapest (no training, just
matrix multiplications) and the most aligned with the model's actual
output space.

## Related notes

- `kb/notes/interpretability/probing.md` — supervised classifier
  readouts of activations.
- `kb/notes/interpretability/mechanistic-interpretability.md` — the
  broader program; lens techniques are an early-and-still-useful
  diagnostic.
- `kb/notes/architecture/transformer-overview.md` — the residual
  stream + unembedding setup that lens techniques exploit.
