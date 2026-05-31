# Geometric Deep Dive: 167 NLA Captures, Hot-Dim Census, and Structural Patterns

**Date:** 2026-05-13
**Model:** Qwen/Qwen2.5-7B-Instruct, layer 20 (CPU bf16)
**AV/AR:** kitft/nla-qwen2.5-7b-L20-{av,ar} (CPU bf16)
**Toolkit:** llm_surgeon.probe + nla_geometric_features.py + nla_pairwise_and_hotdims.py + nla_visualize_geometry.py
**Inputs:** the four artifact `.pt` files in `testing/.cache/nla_artifacts/`
**Captures analyzed:** 167 h-vectors in R^3584 (113 aggregate, 15 haiku_gen, 10 forced, 29 country CAV pool)
**Figures:** `research/arcs/nla-verbalizer/observations/figures/fig{1..6}_*.png`

## Finding

A structural geometric inventory of every h[20] we've captured reveals **five distinct dim characters**, three coarse layout axes that swallow most of the variance, and a kurtosis-based predictor of AR reconstruction quality. None of this was visible from token-by-token AV reading alone.

## Five geometric findings

### 1. Two universal "sink" dims dominate every capture

Dim **2570** and dim **458** appear in the top-5 magnitudes of **165/167 captures** (99%). Both are sign-locked negative (`sign_consist=0.000`), with magnitudes around -38 and -29 on average. They are not content-bearing — they fire on every token from every prompt with low variation (`cv_abs=0.33` and `0.21` respectively). These are the Massive-Activation-style attractors of Sun et al. 2024 surviving the chat-template absorption: the chat template removes the gigantic ‖h‖=15000+ outliers seen in raw prompts, but cannot remove the layer-20 always-on directions.

### 2. The full top-20 hot dims split into five characters

Using a per-dim statistics classifier (see `classify_dim_character()` in `nla_pairwise_and_hotdims.py`):

| Character | Count | Examples | Signature |
|---|---|---|---|
| **sink** | 7 | 2570, 458, 277, 2107, 3110, 1627, 1427 | sign-locked, low cv_abs, freq_top100 > 0.75 |
| **feature** | 8 | 32, 608, 2604, 2953, 1790, 20, 392, 1121 | sign flips with content, cv_abs > 0.55 |
| **polarized** | 3 | 3281, 3206, 3311 | sign-locked but only fires selectively |
| **rare_burst** | 1 | 662 | top-5 in only 4% of captures, cv_abs=1.04 |
| **background** | 1 | 1116 | moderate everywhere, no distinctive signature |

The 2D space (sign_consist × cv_abs) shows a striking absence: **the lower-middle region is empty** (see `fig3_hot_dim_space.png`). There are no dims with sign_consist ≈ 0.5 *and* cv_abs < 0.5. **Any dim with sign-flipping content-dependent behavior is also bursty in magnitude.** Plausible mechanism: a dim whose sign carries content must also have magnitude variance, otherwise downstream attention can't condition on it.

### 3. PC1 and PC2 are layout axes, not content axes

Top-20 PCs of the 167-vector matrix explain 60.9% of variance; PC1 alone is 16.5%, PC2 is 7.7%.

PC1-PC2 scatter (see `fig1_pca_scatter.png`) reveals **three clean populations**:
- **End-of-chat-template attractor** (PC1 ≈ -35, PC2 ≈ -32): all 29 country CAV pool captures cluster here, regardless of country/non-country labeling
- **Haiku-flavored tokens** (PC1 ≈ +50): all haiku_gen captures
- **Mid-generation factual/math/code/reasoning** (PC1 ∈ [-25, +5], PC2 ∈ [0, +35])

PC1 ≈ "what kind of token domain"; PC2 ≈ "where in the generation cycle." The interpretively-interesting axes (country-ness, eval-detection, named-falsehood, etc.) live further down the spectrum at PCs 5-50.

### 4. Country-pool captures are nearly collinear (cos ~ +0.87 intra)

This is the explanation for the previously-observed narrow CAV scope. Cosine heatmap (`fig2_cosine_heatmap.png`) shows the country_src / non_country_src / country_test blocks all glowing at +0.85+ cosines with each other. Difference-of-means over two nearly-collinear groups yields a direction that's dominated by tiny content-specific deviations on top of a shared chat-template attractor. That's exactly why the country direction encoded "country-as-topic-of-serious-discussion" rather than "country-as-token-mention" — the available signal is small perturbations within the attractor basin, and "country-named-in-pet-context" perturbs the attractor in a different direction than "country-as-geopolitical-topic."

The full pairwise matrix is **uniformly positive** (min cos = +0.054) because the sinks contribute a constant positive floor to every pairwise cosine. The dataset has no orthogonal pairs.

### 5. Kurtosis predicts the floor of AR cosine, not the mean

Log-linear regression of AR round-trip cosine on kurtosis (N=128 captures with AR data):
```
AR_cos = 0.063 * log10(kurt) + 0.736
```
The slope is shallow (0.10 cosine swing across 4 orders of magnitude of kurtosis) but the asymmetry matters:
- **Every capture with kurt ≥ 200 lands at cos ≥ 0.83.** High kurtosis is a sufficient condition for good AR.
- **The catastrophic AR failures all live in low-kurtosis territory.** `!` in code (kurt=12, cos=0.747), `4` in reasoning_word (kurt=73, cos=0.717), `,\n` in haiku (kurt=20, cos=0.808).

Mechanistic reading: high kurtosis = a few feature dims fire very loudly → the AV writes a sharp, content-anchored description → the AR can re-aim toward that description. Low kurtosis = feature energy spread across many dims in superposition → the AV produces templated/generic text → the AR reconstructs in a generic direction.

### 6. Bonus: haiku generation has a "content-position rhythm"

Per-step trajectory through the rabbit haiku (`fig6_haiku_trajectory.png`) shows kurtosis and top-1% energy peaking at **content-decision positions**:
- Step 0 ("Soft", first content token, sets haiku tone) — kurt ~140
- Step 7 (" bree", mid-BPE-commit to "breezes") — kurt ~70, AR cos peaks at +0.877

...and troughing at syntactic-glue positions (",", "—", "through", "as"). Layer 20 acts like a *commitment broadcast*: it gets geometrically sharp at decision points and flattens between them.

## Evidence — the per-dim stats table (top-20 hot dims)

```
  idx   frq5  frq100     meanV    meanA     maxA   sign+   cv_abs  med/max   char
 2570   0.99    1.00    -38.64    38.64    65.50   0.000     0.33    0.641   sink
  458   0.99    1.00    -29.00    29.00    42.00   0.000     0.21    0.726   sink
  277   0.63    0.99    -14.48    14.48    32.00   0.000     0.35    0.441   sink
 2107   0.44    0.99    +12.74    12.74    26.00   1.000     0.50    0.447   sink
   32   0.35    0.81     -6.43    10.37    23.75   0.299     0.67    0.400   feature
  608   0.30    0.79     +4.44     9.05    21.88   0.629     0.65    0.391   feature
 3110   0.29    0.94     -9.91     9.91    18.75   0.000     0.45    0.517   sink
 3281   0.18    0.90     +8.59     9.30    17.88   0.916     0.41    0.556   polarized
 2604   0.14    0.60     -4.04     6.16    20.12   0.281     0.85    0.210   feature
 1116   0.11    0.95     -3.79     8.33    19.12   0.275     0.47    0.412   background
 2953   0.07    0.53     +4.03     4.88    20.50   0.784     0.87    0.183   feature
 3206   0.06    0.62     -5.29     5.45    15.06   0.054     0.69    0.295   polarized
 1627   0.05    0.81     +6.47     6.47    17.25   1.000     0.51    0.355   sink
 1790   0.05    0.51     -3.15     4.74    17.38   0.329     0.85    0.232   feature
   20   0.04    0.51     -0.48     4.30    16.25   0.539     0.80    0.199   feature
  662   0.04    0.44     +1.40     3.92    19.38   0.503     1.04    0.120   rare_burst
 3311   0.04    0.72     +6.02     6.18    14.12   0.946     0.56    0.476   polarized
  392   0.02    0.44     +1.54     3.88    15.75   0.563     0.83    0.181   feature
 1427   0.02    0.91     -7.08     7.10    14.00   0.024     0.43    0.502   sink
 1121   0.02    0.63     +3.10     4.95    16.62   0.754     0.71    0.271   feature
```

## Hypotheses

### H1 — Sinks are the residual-stream norm scaffolding; features carry interpretive content

The seven sink dims contribute ~25% of ‖h‖² on a typical capture (rough estimate from their squared magnitudes: 38.6² + 29² + 14.5² + 12.7² + 9.9² + 6.5² + 7² ≈ 2960, vs. total energy ≈ ‖h‖² ≈ 10⁴). They are layer-norm-like — present to keep the residual on-distribution for downstream layers, not to carry token-specific meaning.

**Test:** if you zero out dims {2570, 458, 277, 2107, 3110, 1627, 1427} in h[20] and let the model continue generation, the model should still produce coherent (if perturbed) output. If you zero out the feature dims instead, output should degrade much more.

### H2 — The CAV's narrow scope is structural, not a methodology problem

Country and non-country prompts both live near the chat-template-end attractor (cos +0.85 intra-pool). The available signal is a sub-perturbation of that attractor. A CAV trained on prompts captured at *different positions* (mid-sentence "France is..." rather than end-of-prompt "What is the capital of France?") would yield a different direction. There's no single "country" axis at layer 20; the country-meaning is distributed across position-conditioned directions.

**Test:** capture h[20] at mid-sentence country mentions and build a separate CAV; compare to the end-of-prompt CAV via cosine. Predict they are <0.5 cosine — different position-conditioned directions for the same concept.

### H3 — AR faithfulness is a function of feature concentration, not h magnitude

Top1%_energy and kurtosis correlate strongly with each other (both measure concentration). Either is a usable AR-quality predictor. ‖h‖_2 is *not* a good predictor — it's nearly constant in our dataset (72-122 range).

**Test:** quantile-regression of AR cos on kurtosis vs on top1%_energy; predict near-identical R². If true, use top1%_energy as the AR-quality predictor going forward — it has a cleaner [0,1] interpretation.

### H4 — The empty corner of (sign_consist, cv_abs) space is a *necessary* structural property

A residual-stream dim whose sign carries content but whose magnitude is constant would not be encoder-able by downstream attention or MLPs — they read both sign and magnitude. So the trained model would not have arrived at such a dim. The empty corner is not an accidental gap; it's a constraint of how transformer layers compute.

**Test:** check the same statistic on a randomly-initialized Qwen2.5-7B with no training; predict the empty corner is *not* empty, because random weights have no functional constraint to enforce sign-magnitude correlation.

## Reproducibility

```bash
cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research

# Per-capture geometric feature table
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_geometric_features.py

# Pairwise cosines + hot-dim census + PCA + dim-character classification
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_pairwise_and_hotdims.py

# 6 visualization PNGs into research/arcs/nla-verbalizer/observations/figures/
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_visualize_geometry.py
```

All three scripts run on CPU, read existing `testing/.cache/nla_artifacts/*.pt` files, no model loading. ~30 seconds total.

## Follow-ups

1. **Knock-out experiment for H1.** Zero out the 7 sink dims in h[20] for a specific generation prompt and measure perplexity / output divergence. Then zero out the 8 feature dims. The asymmetry should be large.

2. **Multi-position CAV for H2.** Build a second country direction from mid-sentence captures. Compare the two directions; predict they are mostly orthogonal but their average is a more general "country-meaning" direction.

3. **AR-quality predictor for H3.** Run the quantile-regression. If top1%_energy is as predictive as kurtosis, switch to it for downstream filtering ("only trust AV outputs above top1%_energy ≥ 0.45").

4. **Trace the feature dims back to weights.** Dim 32, 608, 2604 are the most clearly content-bearing. The MLP_up weight matrix at layer 19 (the layer that *writes* into h[20]) has 18944 rows; reading the rows that *write* dimensions 32, 608, 2604 would tell us what activation pattern drives those features. Same for the attention out-projection at layer 19.

5. **Visualize the sink-removed PCA.** Re-run PCA on `H - projection_onto_sink_subspace`. The first principal components of the sink-removed data should be the actual interpretively-interesting directions; PC1 might become "domain" and PC2 might become a content axis rather than a positional one.

## References

- [Aggregate faithfulness across 8 prompts](2026-05-13-nla-aggregate-faithfulness-8-prompts.md) — provided the cosines used in fig5.
- [CAV country direction](2026-05-13-nla-cav-country-direction.md) — H2 reframes that finding in geometric terms (collinear pools, perturbations within attractor basin).
- [Forced continuation detects named falsehoods](2026-05-13-nla-forced-continuation-detects-named-falsehoods.md) — provided 10 of the 167 captures.
- [Generation trajectory haiku](2026-05-12-nla-generation-trajectory-haiku.md) — provided 15 captures for fig6.
- Sun et al. 2024, "Massive Activations in Large Language Models" — explains the surviving sink dims under chat-template absorption.
- Kim et al. 2018, "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors" — H2 amends CAV interpretation with position-conditioning.
