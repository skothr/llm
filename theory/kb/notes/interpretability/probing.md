---
topic: interpretability/probing
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - burns2023-truth-geometry
  - alain-bengio-2017             # foundational; Tier A but not yet downloaded
  - tegmark-truth-2025            # follow-up: Probing the Geometry of Truth
  - lp-scaling-2025
---

# Probing

**Status:** stub. Drafted from Phase 1 landscape sweep + first-page
read of `burns2023-truth-geometry.pdf`; needs full Phase 2 treatment.

## What this topic covers

A **probe** is a (typically linear) classifier $g: \mathbb{R}^n \to
\mathcal{Y}$ trained on labeled activations to predict an external
property $y$ (truth, sentiment, syntactic role, etc.) from the
activation $\mathbf{h}^{(\ell, t)}$ at a chosen layer $\ell$ and token
$t$. The historical reference is Alain & Bengio 2017 ("Understanding
intermediate layers using linear classifier probes"). The implicit
claim of the probing methodology is: if a high-accuracy linear probe
exists for $y$, the model **linearly represents** $y$ in its
activations. The catch (the **probing methodology critique**, ca.
2019–2021): a high-accuracy probe could be reading a property the
model *has* but does not *use* — probes are correlational, not causal.

The 2023+ generation of probing work uses **paired-counterfactual /
causal** probing recipes (e.g., causal abstraction, mass-mean
probing in Burns et al. 2023) to address this critique.

## Primary sources to read (in order)

1. `burns2023-truth-geometry` — *The Geometry of Truth: Emergent
   Linear Structure in Large Language Model Representations*
   (Marks & Tegmark, arXiv 2310.06824) — establishes a "truth
   direction" in activation space via mass-mean probing. PDF
   downloaded at `theory/sources/papers/burns2023-truth-geometry.pdf`.
   *Note on key disambiguation:* this is the Marks & Tegmark 2023
   "Geometry of Truth" paper, not the Burns 2023 weak-to-strong
   generalization paper (`burns2023-w2s` in `papers.json`). The
   shared first-author "Burns" was a Phase 1 transcription error
   in the landscape sweep — Phase 2 should add a key
   `marks-tegmark-2023-truth` instead.
2. `alain-bengio-2017` — *Understanding intermediate layers using
   linear classifier probes* (arXiv 1610.01644) — foundational
   probing methodology. Not yet downloaded.
3. `tegmark-truth-2025` — *Probing the Geometry of Truth: Consistency
   and Generalization of Truth Directions* (ACL Findings 2025) —
   extends truth probing to logical transformations and across
   model sizes.
4. `lp-scaling-2025` — *Linear Probe Accuracy Scales with Model Size
   and Benefits from Multi-Layer Ensembling* (arXiv 2604.13386) —
   systematic probe-scaling study; multi-layer ensemble recovers
   signal where single-layer probes fail.

## Key claims to ground (Phase 2 todo)

- **Linear probe definition.** $g(\mathbf{h}) = \sigma(\mathbf{w}^\top
  \mathbf{h} + b)$, trained by minimizing classification loss on a
  labeled dataset of $(\mathbf{h}, y)$ pairs. Layer $\ell$ and
  position $t$ are hyperparameters; multi-layer / multi-position
  ensembles are common.
- **Mass-mean probing (Burns 2023).** Define the truth direction as
  $\mathbf{d}_\text{truth} = \mathbb{E}[\mathbf{h} \mid y =
  \text{true}] - \mathbb{E}[\mathbf{h} \mid y = \text{false}]$. The
  probe is the projection $\mathbf{h}^\top \mathbf{d}_\text{truth}$.
  Simpler and more robust than logistic regression; survives
  generalization tests across logical transformations
  (negation, conjunction, etc.) where logistic-regression probes fail.
- **The correlational-causal gap.** A probe can read $y$ from
  activations even if the model does not *use* $y$ to make
  predictions. Resolution: pair probing with activation patching
  along the probe direction (ablate $\mathbf{d}$ in the residual
  stream and measure behavior change). This connects probing to
  `kb/notes/interpretability/activation-patching.md`.
- **Probe scaling.** Linear-probe accuracy increases with model size
  and is well-predicted by simple scaling laws (lp-scaling-2025).
  Multi-layer ensembles recover signal that single-layer probes
  miss.
- **Probing for non-factual properties.** Recent work (e.g.,
  rhetorical questions, deception detection, sycophancy) extends
  probing to pragmatic / behavioral phenomena.

## Position in the methodology family

Probing is the *supervised-readout* counterpart to the unsupervised
SAE feature decomposition. Where an SAE finds a basis of
"directions the model uses" without a label, a probe finds the
direction *for* a label. Both produce a vector in residual-stream
space; both are correlational unless paired with a causal
intervention. The 2023+ probing literature is converging on
**causal probing** = probe + activation-patch-along-the-probe-
direction, which is the same methodology as ablation along an SAE
feature direction.

## Open questions

- **Probes vs. SAE features for the same concept.** When both
  techniques find "a Paris feature" / "a truth direction", how often
  are the two directions the same vector (cosine similarity)?
  Phase 2 should ground this — it is currently asserted in informal
  reviews but not, to my knowledge, systematically measured at scale.
- **Does mass-mean probing generalize to non-linear properties?**
  Truth (a binary property) admits a linear separating direction;
  more complex properties (e.g., "the answer is in {Paris, London,
  Madrid}") may not.
- **Probing for safety-relevant properties** — deception, refusal,
  sycophancy — is increasingly load-bearing in alignment evaluation
  (`kb/notes/alignment/scheming-and-deceptive-alignment.md`). The
  question of whether such probes generalize from in-distribution
  evaluation prompts to adversarial out-of-distribution behavior is
  open and contested.

## Related notes

- `kb/notes/interpretability/lens-techniques.md` — vocabulary-projection
  readout; the "untrained probe" baseline.
- `kb/notes/interpretability/sparse-autoencoders.md` — unsupervised
  feature decomposition; SAE feature activations are concept-readouts
  without labels.
- `kb/notes/interpretability/activation-patching.md` — the causal
  intervention that turns a correlational probe into a causal claim.
- `kb/notes/alignment/scheming-and-deceptive-alignment.md` — the
  alignment-evaluation context for behavioral probes.
