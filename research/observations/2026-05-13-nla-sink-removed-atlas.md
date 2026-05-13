# Sink-Removed Atlas + Signature-Glyph Visualization Primitive

**Date:** 2026-05-13
**Model:** Qwen/Qwen2.5-7B-Instruct, layer 20 (CPU bf16)
**Toolkit:** `nla_sink_removed_atlas.py` + `nla_signature_atlas.py`
**Inputs:** `pairwise_and_hotdims.pt` (167 h-vectors + dim-character labels)
**Figures:** `fig7-fig11` in `research/observations/figures/`

## Path-A goal

Test whether removing the 7 sink-classified component indices {277, 458, 1427, 1627, 2107, 2570, 3110} from each h reveals a content-bearing PC layout that the sinks were masking. Then build a visualization primitive that integrates multiple feature dims at once instead of one-coloring-at-a-time.

## Finding

The original interpretation in [the geometric deep dive](2026-05-13-nla-geometric-deep-dive.md) was **wrong about causation**. Removing the sinks barely changes the PCA layout. PC1 variance fraction moves from 16.5% to 15.3%; the country cluster stays in the bottom-left, the haiku cluster on the right. **PC1/PC2 are genuinely content-and-position axes that exist independently of the sinks** — the sinks aren't shaping them.

What sinks DO is add a **constant +0.22 offset to every pairwise cosine**, raising mean off-diag cos from +0.179 (sink-removed) to +0.403 (original) and the floor from -0.069 to +0.054. They are the DC component of the residual stream — they make every comparison look more similar than it actually is, without carrying information.

After sink removal, the true similarity structure emerges:

| pool | orig intra | orig cross | res intra | res cross | intra delta |
|---|---|---|---|---|---|
| aggregate (n=113) | +0.397 | +0.392 | +0.156 | +0.168 | -0.241 |
| haiku_gen (n=15) | +0.530 | +0.322 | +0.412 | +0.120 | -0.118 |
| forced (n=10) | +0.366 | +0.315 | +0.268 | +0.149 | -0.098 |
| country_src (n=8) | +0.858 | +0.482 | +0.777 | +0.257 | -0.081 |
| non_country_src (n=8) | +0.876 | +0.487 | +0.804 | +0.259 | -0.072 |
| country_test (n=13) | +0.800 | +0.465 | +0.692 | +0.237 | -0.108 |

Two consequences:

1. **The aggregate pool's intra-source cosine collapses to +0.156** — barely above its cross-source mean (+0.168). The 113 captures from 8 different prompt domains (factual, math, code, creative_haiku, creative_poem, reasoning_logic, reasoning_word) are essentially **mutually orthogonal in content space**. The "similarity" the original cosine matrix showed was almost entirely sink scaffolding.

2. **The country pool's intra-vs-cross gap widens** from +0.376 to +0.520 (and non_country_src similarly +0.389 → +0.545). After sink removal, country prompts are *more discriminable* from non-country prompts — exactly what you want for a CAV. The earlier CAV's "narrow scope" finding is now reframed: the discriminating direction was always there in the residual; the original cosines just visually masked it.

## The visualization primitive (fig10, fig11)

8-ray "signature glyph" placed at each capture's PC1/PC2 location. Each ray's angle = a specific feature dim (out of 8); ray length = |h[dim]| normalized; color = sign (red positive, blue negative). The 8 feature dims are {20, 32, 392, 608, 1121, 1790, 2604, 2953} — the dim-character classifier's "feature" labels.

What this gives that single-color atlases don't:

* **Local feature signature is visible at a glance.** In the country cluster (fig11 left), all 29 glyphs are nearly identical — every one has a strong red dim-32 ray and faint others. The high intra-cosine of +0.78 becomes visually obvious as glyph-similarity.
* **Temporal trajectories trace a path through glyph space.** In the haiku trajectory (fig11 right), 15 consecutive generation steps trace a winding orange line; the glyph morphs step by step. Step 0 ("Soft") has one signature, step 7 ("bree", mid-BPE-commit position) has another, step 14 ("—") yet another. The trajectory IS a view that doesn't have a standard analog.
* **8 dims in one figure.** Standard scatters can color by one dim at a time; signature glyphs let you read all 8 simultaneously.

## What this points to (the visualization design path)

The signature glyph is a useful primitive. Several composable extensions become possible:

1. **Concept-arithmetic glyph composition.** Compute A + B, A - B, α·A + β·B for arbitrary captures; render the result as a glyph alongside the inputs. The viewer sees both the arithmetic and its semantic location in PC space.

2. **AV-text annotation overlay.** Hover a glyph → see the AV's natural-language interpretation of that h. Static version: pick 6-8 representative glyphs per region and render the AV text in callouts around the atlas.

3. **Interpolation flipbook with glyph morphing.** Path B from the previous plan: interpolate between two text anchors, render the glyph sequence frame by frame. Watching the glyph morph IS the visualization of the semantic gradient.

4. **Sink-removed cosine matrices as default.** Always show the sink-removed pairwise matrix; treat the original as the "raw with DC component" auxiliary view. This becomes a methodology rule: never display raw h-vector cosines without subtracting their sink-subspace projection.

5. **Multi-prompt glyph trajectories.** Capture h's for the same generation under multiple prompts (counterfactual, forced, normal). Render each as a glyph trajectory; overlay them. The divergence between trajectories *is* the model's response to the prompt difference.

## Hypotheses

### H1 — The 5% variance the sinks carry is structural overhead, not content
Sinks carry ~25% of ‖h‖² but only ~5% of within-dataset variance. They're nearly constant. **Test:** randomly perturb the sink components and see if downstream generation degrades. Predict: small perturbations OK, large perturbations break layer-norm stability.

### H2 — The signature glyph is a visual proxy for AV faithfulness
Captures whose glyphs match their cluster mean visually should have higher AR cosine than glyph-outliers. **Test:** define glyph-distance-to-cluster-mean for each capture, regress AR cos on it; predict negative slope.

### H3 — The country CAV direction is dim-32-aligned
Dim 32 lights up red across the entire country cluster but blue/zero elsewhere. The CAV's discriminator may largely be dim 32. **Test:** compute cos(CAV_country, e_32); predict ≥+0.4 (single-dim alignment is unlikely to be perfect but a dominant component is plausible).

## Reproducibility

```bash
cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research

# Path A: sink-removed analysis + 3 comparison figures
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_sink_removed_atlas.py

# Signature-glyph atlas + zoom views
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_signature_atlas.py
```

Both run on CPU, no model loading, ~1 minute total.

## Follow-ups (visualization-focused)

1. **fig12: interactive Bokeh/Plotly atlas.** Hover-to-see-AV-text version of fig10. ~1 hour to build. Lets researchers explore the 167 captures freely.
2. **fig13: concept-arithmetic glyph table.** Compute the country CAV direction (we have it), AV-decode it, render as a glyph next to its source captures. Test H3 directly. ~30 min.
3. **fig14: haiku flipbook with AV captions.** For each of the 15 haiku-gen steps, show: token | AV text | signature glyph. Layout as a vertical strip. Already have all the data on disk.
4. **fig15: counterfactual residual viz.** For the 4 forced-continuation pairs, plot natural-glyph and forced-glyph side by side at each position. Difference glyph in a third column.
5. **Multi-position CAV for H3.** Capture country mentions mid-sentence, build a separate CAV, render its glyph; compare with the existing end-of-prompt CAV's glyph.

The signature-glyph primitive is small and composable — most follow-ups reuse it as a building block rather than inventing new viz primitives.

## References

- [Geometric deep dive](2026-05-13-nla-geometric-deep-dive.md) — identified the 7 sinks and 8 feature dims that this Path A built on. Also corrects that note's "sinks drive PCA layout" claim.
- [CAV country direction](2026-05-13-nla-cav-country-direction.md) — the narrow-scope finding is now reframed in terms of sink-induced cosine inflation rather than CAV methodology.
- Sun et al. 2024, "Massive Activations in Large Language Models" — the surviving-under-chat-template sinks are the layer-20 cousins of their massive-activation phenomenon.
