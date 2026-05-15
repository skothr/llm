# Mid-Seq Native Discriminants Confirm Protocol-Coupling; Map Cross-Protocol Axis Stability

**Date:** 2026-05-14
**Toolkit:** `nla_mid_seq_native_compare.py`
**Inputs:** `vocab_atlas.pt` (eop), `mid_seq_vocab_atlas.pt`, `pairwise_and_hotdims.pt`
**Output:** `mid_seq_native_compare.pt`
**Figures:** `fig33_native_signal_lift.png`, `fig34_cross_protocol_axis_cos.png`
**Linear:** MAIN-70 (followup to MAIN-44)

## What MAIN-70 predicted

MAIN-44 found that 23 end-of-prompt-derived discriminants project mid-seq h's nearly orthogonally (+0.0491 aggregate). The natural follow-up: build mid-seq-NATIVE discriminants (same `mean(cat) - mean(non-cat)` recipe applied to mid-seq captures) and check whether in-protocol signal lifts to match the eop in-protocol +0.4022. Prediction: yes, ~+0.4 (confirming protocol-coupling). Also map per-category cross-protocol axis stability via the 23×23 cosine matrix.

## What actually happened

Computed both discriminant families and crossed each protocol's captures against each:

| captures × discriminants | aggregate signal | argmax accuracy |
|---|---|---|
| eop-h × eop-discr (in-protocol) | +0.4022 | 75.37% |
| eop-h × mid-discr (cross-protocol) | +0.0369 | 6.96% |
| mid-h × eop-discr (cross-protocol, MAIN-44) | +0.0491 | 32.04% |
| **mid-h × mid-discr (in-protocol)** | **+0.5632** | **97.10%** |

Native mid-seq discriminants give the *highest* in-protocol signal of either protocol — 40% higher than eop's in-protocol baseline, and argmax accuracy reaches 97.10% (vs 75.37% at eop). The basis-design is **protocol-coupled by construction**: a per-protocol family of discriminants each yields strong within-protocol classification.

The asymmetric cross-protocol numbers (eop-h × mid-discr = +0.037 < mid-h × eop-discr = +0.049) are mildly suggestive: mid-discriminants generalize slightly *worse* to eop-h than eop-discriminants do to mid-h. Possible cause: mid-seq captures have lower variance in carrier context (every anchor uses the same surrounding prompt), so mid-discriminants have a sharper but more position-specific direction.

## Cross-protocol axis stability (fig34)

The 23×23 cosine matrix `cos(d_eop_C, d_mid_D)` quantifies how each category's discriminant direction transfers across protocols:

| Statistic | Value |
|---|---|
| Mean diagonal (same-category, cross-protocol) | +0.0784 |
| Max diagonal | +0.1704 (`emotion`) |
| Min diagonal | +0.0126 (`p_quote`) |
| Mean off-diagonal | -0.0009 |

Diagonal entries are weakly positive (cosines +0.01 to +0.17), off-diagonal mean is essentially zero — so each category's axis at one protocol points in a direction that's *closer to its own axis at the other protocol than to a random other category's axis*, but only modestly so.

**Top-5 most-stable axes** (largest diagonal cosines) — all content-bearing categories:

| category | cos(d_eop, d_mid) |
|---|---|
| emotion | +0.170 |
| capital | +0.151 |
| country | +0.150 |
| nature | +0.131 |
| refusal | +0.127 |

**Top-5 least-stable axes** — function-word + punctuation:

| category | cos(d_eop, d_mid) |
|---|---|
| p_quote | +0.013 |
| p_special | +0.026 |
| math_op | +0.028 |
| auxiliary | +0.032 |
| preposition | +0.043 |

## What this tells us about layer-20 geometry

The content vs function distinction shows up again as the dominant organizing principle — and now at a deeper level than fig21's PC1. Per-category axes for content concepts (country, emotion, nature) have a modest position-invariant component, while punctuation and function words' axes are *almost entirely* position-determined. This matches the dominant PC1 (content-vs-function, 33.5% variance) found earlier in the sink-removed vocab atlas — but is a stronger statement: not just that content and function words occupy different *regions*, but that the *direction* in which a content category fans out from the population mean is partially preserved across positions, while function-word axes are constructed anew at each position.

This refines the MAIN-44 closure: the basis isn't *purely* end-of-prompt-protocol-coupled. There's a small protocol-invariant component (~+0.15 cosine for content categories). But it's small enough that cross-protocol projection collapses to noise.

## Implications for the glyph viz primitive

A discriminant glyph is **only meaningful for h captured at the same protocol the discriminants were derived from**. Cross-protocol use should be flagged. Two paths forward:

1. **Per-protocol glyph family.** Have multiple discriminant sets (eop, mid-seq, mid-generation, etc.), and pick the matching one based on the capture protocol. Honest but uses N times the visual real estate.
2. **Stable-axis-only glyph.** Build a glyph using only the top-K most-stable axes (the 5 content-category axes here). Smaller glyph, but the axes are interpretable as carrying a position-invariant semantic component. Probably the right primitive for cross-protocol comparison views.

The interpolation flipbook (fig17/fig25) used cross-protocol-by-default discriminants on interpolated h's. The strong t=0.421 transition there (MAIN-25) is **still real** — fig21 confirmed it via cosine-to-vocab-anchors (no discriminants) — but the per-step glyph readings on fig25 should be interpreted with caution since the h's at intermediate t are out-of-protocol for the eop discriminants.

## Cross-arc lessons

MAIN-44 + MAIN-70 together turn what looked like a null result into a productive characterization of the basis. The pattern: **null result + native re-derivation reveals what the failed basis was missing**. Worth keeping as a methodological template — when a basis "fails" cross-protocol, build the native basis and compare directly to map what *is* preserved.
