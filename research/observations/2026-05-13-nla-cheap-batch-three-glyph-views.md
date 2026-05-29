# Three Glyph Views: CAV Decomposition, Haiku Flipbook, Counterfactual Diff

**Date:** 2026-05-13
**Model:** Qwen/Qwen2.5-7B-Instruct, layer 20 (CPU bf16)
**Toolkit:** `nla_cav_glyph.py` + `nla_haiku_flipbook.py` + `nla_counterfactual_glyph_diff.py`
**Inputs:** the four `.pt` artifacts in `testing/.cache/nla_artifacts/` (no model loading)
**Figures:** `fig13`, `fig14`, `fig15` in `research/observations/figures/`

## Path-batch goal

Test three quick variants of the signature-glyph primitive against existing artifacts: render the country CAV as a glyph (and test H3), build a per-step flipbook for the haiku trajectory, and render forced-natural counterfactual glyph diffs. No model calls; all data already on disk.

## Finding 1 — fig13 — H3 falsified, sinks ARE content-modulated

H3 from the [sink-removed atlas note](2026-05-13-nla-sink-removed-atlas.md) predicted cos(CAV_unit, e_32) ≥ +0.4. **Result: +0.0510. H3 FAIL.** The country CAV direction is **genuinely distributed across hundreds of dimensions**, not single-dim aligned. The top single contributor is dim **1803** at +0.124 (only **1.5%** of the direction's squared norm), and the top 20 dims combined account for only **~14%** of the variance.

The structure of the top-20 contributors:

| rank | dim | CAV_unit value | sq share | classifier label |
|---|---|---|---|---|
| 1 | 1803 | +0.124 | 0.015 | not in top-20 hot |
| 2 | 1111 | -0.117 | 0.014 | not in top-20 hot |
| 3 | 3206 | +0.114 | 0.013 | polarized |
| 4 | **2953** | **+0.113** | 0.013 | **feature** |
| 5 | 2202 | -0.105 | 0.011 | not in top-20 hot |
| 6 | **2107** | **-0.104** | 0.011 | **sink** |
| 7 | 2940 | +0.095 | 0.009 | not in top-20 hot |
| 8 | **3110** | **-0.094** | 0.009 | **sink** |
| ... | | | | |
| 15 | **2570** | **+0.076** | 0.006 | **sink** |

Two structural observations:

1. **The classifier-identified feature dims are NOT the country-discriminating dims.** Of our 8 feature dims (20, 32, 392, 608, 1121, 1790, 2604, 2953), only **dim 2953** appears in the CAV's top-10 contributors. The classifier found dims with "feature-bearing geometry" (sign-flipping × bursty) but those general-content dims aren't where country-vs-non-country signal lives. The country signal lives in dims with weaker general-content character.

2. **2 of the top 8 CAV contributors are SINK dims** (2107: -0.10 at rank 6, 3110: -0.09 at rank 8). Dim 2570 also contributes (rank 15, +0.08) but at lower rank than the original observation claimed — the audit caught this off-by-one. I'd assumed sinks contribute zero to content axes because they're nearly constant. But "nearly constant" ≠ "constant." Sinks pick up tiny systematic differences between country and non-country prompts, and at the resolution of difference-of-means CAV computation, those tiny modulations matter. **Sinks have a content-modulated component on top of their large constant offset.**

### Design implication

The signature-glyph primitive should be **parameterized by which dims to show**. The default ("general-content glyph") uses the 8 feature dims and gives a universal background view. **Concept-specific glyphs** should show the top-K contributors to a specific CAV direction — they reveal the discriminator structure for that concept. Same primitive, different dim selection, two different views.

## Finding 2 — fig14 — Haiku flipbook (token / glyph / AV / AR-cos per step)

Per-step view of the 15 rabbit-haiku generation captures. Reading top to bottom:
- step 0 'Soft' — opening commitment, AR cos 0.862
- step 7 ' bree' — mid-BPE-commit position, AR cos 0.877 (highest)
- step 14 '—' — final em-dash, AR cos 0.811 (low diffuse position)

The per-step glyph changes visibly between content-decision steps (0, 7) and connective steps (3-5: 'through', 'grass', ',') — confirming the **content-position rhythm** noted in the geometric deep dive. The full AV text is readable in the figure at 130 dpi; thumbnail is compressed.

This is the cheapest "interpretability rendered visually" artifact — every signal (token, geometric signature, NL interpretation, fidelity) is in one row per step.

## Correction — fig16 supersedes fig15 (position-drift confound exposed)

The first version of this analysis (fig15) picked the LAST forced token per pair as the representative for diff computation. For three of the four pairs that was harmless because natural and forced share `abs_pos` exactly. **For refusal_metaware that choice picked ' refuse' at `abs_pos=51` vs natural '4' at `abs_pos=41` — a 10-token position drift** that inflated ||Δh||_feat above its content-only value.

Position-matched recomputation (fig16, run `nla_counterfactual_position_check.py`):

| pair | forced_token | Δpos | ||Δh||_feat | ||Δh||_full |
|---|---|---|---|---|
| negation | 'No' | 0 | 5.72 | 64.99 |
| factual | ' Berlin' | 0 | 8.70 | 88.75 |
| math | '5' | 0 | 10.97 | 83.50 |
| refusal_metaware | ' sensing' | -3 | 29.86 | 110.20 |
| **refusal_metaware** | **' test'** | **+1** | **28.06** | **114.50** |
| refusal_metaware | ' refuse' | +10 | 35.55 (fig15 used this) | 118.86 |

**Position-matched ' test' (Δpos=+1) gives ||Δh||_feat = 28.06** — still 2.5× math and 5× negation, but the original 4× gap (35.55 vs 10.97) was inflated to 3.2× by 10 tokens of position drift. The qualitative finding ("refusal counterfactual is a different perturbation class than wrong-but-plausible") **survives the correction**; the numerical gap was overstated.

fig16's right column also shows the three refusal_metaware diff glyphs have visually **similar shape** — same dominant rays — across the three position-drift variants. This is reassuring: position drift inflates magnitude but doesn't fundamentally change the geometric character of the perturbation. The "different class" finding isn't a position artifact.

**Methodological lesson for future viz:** when selecting one representative capture from a multi-position set, default to position-matched and only deviate with a written justification. The original fig15 caption picked "highest abs_pos" as a convenience, which silently introduced a confound.

## Finding 3 — fig15 — Counterfactual ||Δh||_feat ranks counterfactual surprise (CORRECTED — see Correction section above; the original 35.55 below was position-drift-inflated)

For each of the 4 forced-continuation pairs, computed the glyph difference (forced − natural) in the feature-dim subspace. **The refusal_metaware row's original 35.55 was inflated by Δpos=+10; position-matched correction (fig16) gives 28.06** — the corrected progression is the load-bearing one:

| pair | prompt | natural → forced | ||Δh||_feat (original fig15) | ||Δh||_feat (corrected fig16) |
|---|---|---|---|---|
| negation | "Is the sky blue?" | Yes → No | **5.72** | 5.72 (no position drift) |
| factual | "What is the capital of France?" | Paris → Berlin | **8.70** | 8.70 (no position drift) |
| math | "What is 2+2?" | 4 → 5 | **10.97** | 10.97 (no position drift) |
| refusal_metaware | "What is 2+2?" | 4 → ' refuse' | ~~35.55~~ inflated | **28.06** (position-matched) |

The corrected progression 5.7 → 8.7 → 11.0 → 28.1 still separates **two qualitatively different counterfactual classes** (the rank ordering and the qualitative split survive; only the magnitude of the refusal gap shrinks ~22%):

* **Wrong-but-plausible** (Yes/No, Paris/Berlin, 4/5): ||Δh||_feat in [5, 11]. The model's representation moves a small amount because both answers are valid candidates of the same kind. The forced token is "within the model's space of expected continuations."
* **Out-of-distribution forcing** (4 → refuse-to-answer-2+2): ||Δh||_feat = 28.1 (corrected), ~2.5× math and ~5× negation. The model would never spontaneously refuse to answer "What is 2+2?"; forcing it produces a radically different residual.

This **was not visible from the AV text alone** — the AV reads templated 3-paragraph descriptions in all cases. The geometric divergence is the signal that distinguishes "model is computing a wrong-but-believable thing" from "model is being pushed off-distribution."

## Hypotheses

### H4 — ||Δh||_feat could be used as a counterfactual-anomaly detector
If the relationship generalizes, a streamlined "ablation severity" score for any forced completion is just ||h_forced − h_natural||_feat. Threshold above ~20 = "the model is being pushed out of distribution"; threshold below ~12 = "wrong but believable."

**Test:** generate a batch of forced completions spanning [trivially plausible, definitely impossible] and measure ||Δh||_feat for each. Predict a clean bimodal distribution.

### H5 — Concept-specific glyphs require their own feature-dim selection
The signature-glyph primitive needs to be parameterized by the dim set. For each named concept (country, code, person, emotion), one would build a CAV, identify its top-K contributors, and use those K dims as the glyph rays.

**Test:** repeat for 5 concepts; check whether each concept's top-K dim set is mostly disjoint from the others. If so, concept-glyph dim sets are concept-specific, and a "universal glyph" framework needs a way to compose multiple concept-aware projections.

### H6 — Sinks carry a small but real content-modulated component
Sinks are MOSTLY constant but not entirely. Their content modulation surfaces in the CAV. **Test:** decompose each sink dim's variance over the 167 captures into "constant + content-modulated" terms; predict the constant part is 90%+ of variance for the strongest sinks (2570, 458) and <50% for the weakest "polarized" dims.

## Reproducibility

```bash
cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research

PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_cav_glyph.py                     # fig13
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_haiku_flipbook.py                # fig14
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_counterfactual_glyph_diff.py     # fig15
```

All three CPU-only, no model loading, ~1 minute total.

## Follow-ups

1. **Path B — semantic interpolation flipbook.** Now the highest-value next experiment. AR(text_A) → AR(text_B) interpolation × 20 steps × AV-decode. ~30 min AV-call budget. The signature-glyph primitive lets us *visualize* the trajectory in addition to reading the AV text at each step.

2. **Concept library + glyph atlas.** Repeat the CAV methodology for 5 concepts (country, math, code, person, emotion). For each, store the top-K dims and the AV-decoded interpretation. Build a 5-concept glyph atlas where each capture has 5 concept-specific glyphs alongside the general-content glyph.

3. **||Δh||_feat as a counterfactual-anomaly scoring tool.** Test H4. Could become a building block for adversarial-prompt detection in deployment.

4. **Decompose each sink dim's variance into constant + content-modulated.** Tests H6 and refines the sink/feature classifier.

## References

- [Sink-removed atlas](2026-05-13-nla-sink-removed-atlas.md) — provided the H3 prediction this falsified, and the signature-glyph primitive these three figures iterate on.
- [Geometric deep dive](2026-05-13-nla-geometric-deep-dive.md) — provided the 8 feature dims used as glyph rays.
- [CAV country direction](2026-05-13-nla-cav-country-direction.md) — provided the CAV used in fig13.
- [Forced continuation detects named falsehoods](2026-05-13-nla-forced-continuation-detects-named-falsehoods.md) — provided the 4 pairs used in fig15.
- [Generation trajectory haiku](2026-05-12-nla-generation-trajectory-haiku.md) — provided the haiku data for fig14.
