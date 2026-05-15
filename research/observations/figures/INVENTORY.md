# Figure Inventory — NLA Research Arc

Catalogue of all 32 figures in this directory. Each entry: what the figure shows, source script, source data, model dependencies, assumptions / preprocessing, and any corrections applied.

**Common assumptions across the arc** (so we don't repeat them per-figure):

- **Model:** all activations are from `Qwen/Qwen2.5-7B-Instruct` (cached at `testing/.cache/models/`), layer 20 (`hidden_states[20]`), CPU bf16.
- **AV / AR:** `kitft/nla-qwen2.5-7b-L20-av` (8B params, h→text) and `kitft/nla-qwen2.5-7b-L20-ar` (5B backbone + Linear(d,d) value head, text→h). Both CPU bf16.
- **Capture protocol** for vocab atlas + country pool: chat-templated user message, capture `h[20][0, -1, :]` at the last position before the assistant turn.
- **Toolkit:** `llm_surgeon.probe.{load_av, load_ar, nla_verbalize, nla_reconstruct, nla_score}` (see `testing/llm_surgeon/probe/_nla.py`).
- **Sink dims:** {277, 458, 1427, 1627, 2107, 2570, 3110} — identified by the `classify_dim_character` heuristic in `nla_pairwise_and_hotdims.py`. "Sink-removed" preprocessing zeros these 7 component indices.
- **Feature dims:** {20, 32, 392, 608, 1121, 1790, 2604, 2953} — same heuristic, content-bearing dims.
- **Audit:** `testing/examples/nla_audit_findings.py` re-derives every load-bearing number from raw `.pt` artifacts. 93 PASS / 0 FAIL as of commit `ee58e62`.

---

## Geometric deep dive (fig1-fig6)

Source script: `testing/examples/nla_visualize_geometry.py`. Data: `geometric_features.pt` (per-capture statistics) + `pairwise_and_hotdims.pt` (167-vector H matrix + classifier output).

### fig1_pca_scatter.png
PCA of all 167 h[20] vectors in PC1-PC2 space, colored by source pool (aggregate / haiku_gen / forced / country_src / non_country_src / country_test). Three distinct populations visible: end-of-chat-template attractor (bottom-left), haiku-flavored tokens (right), mid-generation captures (middle).

Source data: `pairwise_and_hotdims.pt`. **No sink removal** — shows the raw PCA layout.

### fig2_cosine_heatmap.png
Full 167×167 pairwise cosine matrix, ordered by source pool with category boundaries. Reveals the country-pool block as the brightest intra-pool region (+0.85+) and the uniformly-positive nature of the matrix (min cos +0.054).

Source data: `pairwise_and_hotdims.pt`. **No sink removal**.

### fig3_hot_dim_space.png
2D scatter of the top-20 hot dims in `(sign_consist × cv_abs)` space, colored by `classify_dim_character` output. Reveals the empty corner (no sign-flipping low-variance dims) and the natural clustering into sink / polarized / feature / rare-burst / background.

Source: `classify_dim_character()` in `nla_pairwise_and_hotdims.py`.

### fig4_hotdim_activations.png
Per-capture signed values of 4 sink dims (top panel) and 3 feature dims (bottom panel) across all 167 captures, ordered by source. Sinks stay in a fixed half-space; features cross zero.

### fig5_kurtosis_vs_ar.png
Scatter of (kurtosis of h[20] components) vs (AR round-trip cosine) for the 128 AR-data captures, with a log-x linear fit `cos = 0.063·log10(kurt) + 0.736`. Shows kurtosis predicts the *floor* of AR cosine, not the mean.

**Caveat:** 15 of the 128 plotted points are byte-for-byte duplicates between aggregate/creative_haiku and haiku_gen (caught by audit). Deduplicated fit shifts slope by only -0.003. Cosmetic; conclusion unchanged.

### fig6_haiku_trajectory.png
Per-step trajectory through 15 rabbit-haiku generation tokens. Four panels: `||h||_2`, kurtosis (log scale), top-1% energy concentration, AR round-trip cosine. Reveals content-position rhythm (sharp at line-initial + mid-BPE-commit positions, diffuse at syntactic glue).

Source data: `rabbit_haiku_gen_trajectory.pt`.

---

## Sink-removed atlas + signature glyph (fig7-fig11)

Source scripts: `testing/examples/nla_sink_removed_atlas.py` + `testing/examples/nla_signature_atlas.py`. Data: `sink_removed_atlas.pt` (derived from `pairwise_and_hotdims.pt`).

### fig7_pca_sink_vs_clean.png
Side-by-side PCA of original H vs sink-removed H. Layout barely rotates (PC1 drops only 16.5%→15.3%). Surprising result — the sinks add a constant offset but don't shape the PCA layout.

### fig8_cosine_sink_vs_clean.png
Side-by-side pairwise cosine matrix, original vs sink-removed. Striking visual difference: sink removal collapses the aggregate-pool block to near-white (mean cosine +0.40→+0.18), revealing that most captures are content-orthogonal beneath the sink scaffolding.

### fig9_feature_colored_atlas.png
2×2 sink-removed PCA grid colored by: source / h[32] value / h[608] value / AR cosine. Shows different feature dims paint orthogonal spatial gradients on the same PC layout — PC1/PC2 integrate multiple feature dims.

### fig10_signature_atlas.png
Full sink-removed PCA scatter with **8-ray signature glyph** at every point. Each ray = a feature dim from the 8-dim set; ray length = `|h[dim]|`, color = sign. Dense; legible only at full resolution.

**Assumption:** uses the 8 dim-character-classifier `feature` dims as axes. This is the "general-content glyph" primitive.

### fig11_signature_zoom.png
Two zoomed panels: country-pool cluster (29 glyphs nearly identical, confirming intra-pool cosine +0.78) and haiku-gen trajectory (15 glyphs connected by line in temporal order, glyphs morph step by step).

---

## Counterfactual diff (fig13, fig15-fig16)

Source scripts: `testing/examples/nla_cav_glyph.py` (fig13), `nla_counterfactual_glyph_diff.py` (fig15), `nla_counterfactual_position_check.py` (fig16). Data: `country_concept_vector.pt`, `forced_continuation.pt`, `pairwise_and_hotdims.pt`.

### fig13_cav_glyph.png
4-panel glyph view of the country CAV direction: country_mean, non_country_mean, CAV-at-scale-150, raw difference-of-means. Tests H3 (CAV is dim-32-aligned). **Result: H3 falsified** — cos(CAV_unit, e_32) = +0.051, not the predicted ≥+0.4.

**Conclusion correction:** the CAV is genuinely distributed across hundreds of dims; top-1 contributor is dim 1803 at only 1.5% variance share.

### fig15_counterfactual_diff.png
4-row panel of forced-continuation pairs (factual Paris→Berlin, negation Yes→No, math 4→5, refusal_metaware 4→' refuse'). Per row: natural glyph, forced glyph, diff glyph. Reports `||Δh||_feat` per pair.

**Known issue (superseded by fig16):** the refusal_metaware row uses ' refuse' at abs_pos=51 vs natural '4' at abs_pos=41 — a 10-token position drift that inflates `||Δh||_feat` from a true ~28 to 35.55. **fig16 supersedes this row with position-matched comparison.** Preserved per user's "don't overwrite" instruction.

### fig16_counterfactual_position_check.png
Position-matched correction to fig15. Shows all 3 refusal_metaware forced tokens (' sensing' Δpos=-3, ' test' Δpos=+1, ' refuse' Δpos=+10) so the position-drift effect on `||Δh||_feat` is visible. The cleanest comparison (' test' at Δpos=+1) gives `||Δh||_feat = 28.06`, still 2.5× the math diff — qualitative finding (refusal is a different perturbation class) survives.

---

## Haiku flipbook (fig14)

### fig14_haiku_flipbook.png
Per-step view of the 15 haiku-gen captures, one row per token. Each row: step+token label, 8-ray signature glyph, AV text (first paragraph), AR cosine bar. Designed for high-resolution full-screen viewing; at thumbnail scale text is unreadable.

Source script: `testing/examples/nla_haiku_flipbook.py`. Data: `rabbit_haiku_gen_trajectory.pt` + sink dims from `pairwise_and_hotdims.pt`.

---

## Path B — interpolation flipbook (fig17-fig18)

Source script: `testing/examples/nla_interpolation_flipbook.py` (capture) + `nla_render_interpolation_flipbook.py` (render). Data: `interpolation_flipbook.pt`.

Both **AR and AV models loaded** for this run (sequential, ~38 min wall time). 20 interpolation steps between two AR-encoded NL anchors:
- Anchor A: `"Structured factual format with subject 'France' and predicate 'capital'. Final token 'Paris' completes a geography question."`
- Anchor B: `"Structured poetic format with imagery of butterflies in spring. Final token 'flutter' completes a haiku-style line."`

### fig17_interp_flipbook.png
Vertical strip flipbook, 20 rows. Per row: t-marker bar (blue→orange gradient), general-content signature glyph (8 feature dims), interpolation-specific glyph (top-8 dims by `|h_A − h_B|`), AV text (first paragraph). Reveals the **stepwise semantic transition** at t=0.421 — the AV format word flips from "factual" to "poem" within one 5% interpolation step.

### fig18_interp_diagnostic.png
4-panel diagnostic for fig17: `cos(h_t, h_A)` and `cos(h_t, h_B)` curves crossing at t=0.5 (linear-interp sanity); `||h_t||` over t showing midpoint dip from 66.30 down to 60.73 (~8%); `||h_t − h_A||` linear (linear-interp sanity); per-step `||Δh||` constant at 2.734 up to FP precision.

**Correction:** earlier draft of the linked observation said `||h_A|| = 51.94`. Audit caught it — that was actually `||h_A − h_B||`. Truth: `||h_A|| = 65.73, ||h_B|| = 66.30, ||h_A − h_B|| = 51.94`. Fix landed in `2026-05-13-nla-interpolation-flipbook.md` (commit `ee58e62`).

---

## Vocab atlas (fig19-fig22)

Source scripts: `testing/examples/nla_vocab_atlas_capture.py` (capture, ~13 min CPU) + `nla_vocab_atlas_render.py` (render). Data: `vocab_atlas.pt`. **Only the Qwen base model loaded** — no AV/AR.

128 anchor tokens across 23 categories: content (country / capital / nature / codemath / emotion / refusal), function words (article / pronoun / demonstrative / preposition / conjunction / auxiliary / negation / quantifier / wh_word), punctuation (6 sub-categories), numbers (digits + math ops). Each anchor captured at end-of-single-token-user-message position.

### fig19_vocab_atlas.png
Full 128-anchor sink-removed PCA scatter. PC1 (33.5%) separates content-bearing (left) from structural/function (right). PC2 (15.3%) is valence/format. Named anchors labeled.

### fig20_combined_atlas.png
207-vector combined PCA: 128 labeled vocab anchors + 167 faded existing captures in shared sink-removed PC space. Reveals that vocab anchors and existing captures occupy **disjoint regions** — prompt-structure dominates the combined-corpora PCA variance.

**Methodology note surfaced:** PCA on mixed corpora is misleading. Use cosine-to-anchor (translation-invariant) for cross-corpus comparison; fig21 already does this.

### fig21_interp_through_anchors.png
Cosine of the 20 interpolation steps (fig17) onto all 128 vocab anchors, with top-3 nearest anchors per step on the left and a per-category heatmap on the right. **Independent confirmation of the t=0.421 pivot**: top-3 anchors flip from `Madrid/Berlin/Tokyo` (capital category) to `autumn/snow/Berlin` (transition) to `autumn/snow/sky` (nature) at exactly t=0.421.

### fig22_anchor_cosine_matrix.png
Full 128×128 anchor pairwise cosine matrix, ordered by category. Visible block structure: content macro-cluster (top-left), function-words middle, punctuation+numbers+math_op bottom-right. **All categories have intra-cos > +0.84 even after sink removal** — there's a category-attractor subspace separate from the sink subspace (open question, see MAIN-24).

---

## Anchor-projection glyph (fig23-fig26)

Source scripts: `testing/examples/nla_anchor_glyph.py` (fig23/24 — centroid-based) + `nla_discriminant_glyph.py` (fig25/26 — discriminant-based). Data: `vocab_atlas.pt` + `pairwise_and_hotdims.pt`.

### fig23_anchor_glyph_interp.png — DEPRECATED, see fig25
**Problem:** all 23 rays active on every glyph because the category centroids are non-orthogonal (mean cross-cosine +0.85 in centroid space). Every centroid was 77-97% aligned with the grand-mean of all centroids; only a tiny perturbation was category-specific. Made every glyph look the same.

### fig24_anchor_glyph_samples.png — DEPRECATED, see fig26
Same problem as fig23 — all rays active on every sample.

### fig25_discriminant_glyph_interp.png — SUPERSEDES fig23
**Fix applied:** use difference-of-means *discriminant* directions instead of centroids. `d_cat = unit(mean(cat_h) − mean(non-cat_h))`. After the switch, mean cross-discriminant cosine drops from +0.85 to +0.006 (essentially uncorrelated). Glyphs become sparse — only categories the h actually fits have long rays; categories it actively anti-fits have dotted rays.

### fig26_discriminant_glyph_samples.png — SUPERSEDES fig24
Same fix applied to the 6-sample panel. Each sample now shows 2-3 strongly positive rays + 2-3 strongly negative rays + near-zero everywhere else.

---

## Discriminant validation (fig27-fig29)

Source scripts: `testing/examples/nla_discriminant_connectivity.py` (fig27 + fig29) + `nla_discriminant_stability_capture.py` + `nla_discriminant_stability_render.py` (fig28). Data: `vocab_atlas.pt`, `discriminant_stability.pt`.

### fig27_discriminant_connectivity.png
23×23 discriminant pairwise cosine heatmap. Reveals **three macro-clusters**: content (top-left 6 categories), function-words (middle 9), structural punctuation+numbers (bottom-right 8 — most tightly correlated). Off-block regions are deeply blue (anti-correlated). The model's mid-late residual is hierarchically organized.

### fig28_discriminant_stability.png
8 anchors × 4 prefix-length contexts each. Per anchor: 4 glyphs side-by-side plus ctx-cosine + expected-projection statistics. Reveals two stability classes: stable anchors (the, France, function, ".", "7", Paris — ctx-cos > 0.80) and unstable anchors (happy, refuse — ctx-cos < 0.60).

**Major finding from this figure:** `happy → emotion` projection mean +0.083 ± 0.061; `refuse → refusal` mean +0.052 ± 0.130 (std > mean). The discriminants do prompt-topic classification, not token-presence detection.

### fig29_self_validation.png
167×23 heatmap: existing captures projected onto each discriminant, ordered by source pool. Black dots mark the expected category per row. Shows top-5 hit rates per category: country 79%, codemath 73%, nature 56%.

**Caveat surfaced by MAIN-47:** the 34% top-1 number was inflated low by labeling artifacts (country_test pool contains weird-framing prompts intentionally not country-content). See MAIN-68 follow-up for strict-prompt re-validation.

---

## Hierarchical re-discrimination (fig30)

Source script: `testing/examples/nla_hierarchical_classifier.py`. Data: `vocab_atlas.pt` + `pairwise_and_hotdims.pt`.

### fig30_hierarchical_accuracy.png
Bar chart comparing baseline (single discriminant) vs hierarchical (sub-discriminator on sibling pairs) top-1 accuracy per source-mapped category. **Null result**: only 1 of 33 sibling-applicable captures flipped, lifting country 34%→38%, overall 44%→45%.

The diagnostic from MAIN-47's resolution: the 34% baseline was a sum of 3 different failure modes (genuine sibling swap, right-by-other-name, mislabeled tests), only the first of which is fixable by the basis. The hierarchical scheme fixed the only "genuine sibling swap" case in the dataset.

---

## Mid-sequence vocab atlas (fig31, fig32)

Source scripts: `testing/examples/nla_mid_seq_vocab_atlas_{capture,compare,render}.py`. Data: `mid_seq_vocab_atlas.pt`, `mid_seq_compare.pt`, joined with `vocab_atlas.pt` + `pairwise_and_hotdims.pt`. Carrier prompt: `"The text contains many words. Here is one specific word: {anchor} continues throughout subsequent discussion paragraphs."` — anchor lands at token pos 36-38 of ~48 (~75% of sequence, 10-12 trailing-context tokens). Position-finding by prefix-tokenize-and-count (BPE is left-to-right so the left segment's token count is invariant to right context).

### fig31_mid_seq_signal_vs_noise.png
Two-panel bar chart per category. **Top**: within-class signal (`mean over a in C of cos(h_a_sink_removed, d_C)`) at end-of-prompt (blue) vs mid-sequence (orange). **Bottom**: max off-class projection (noise floor) under both protocols. The 23 discriminants `d_C` are derived from the end-of-prompt vocab atlas only; mid-seq h's are projected onto them as a cross-protocol test. **Null result**: mid-seq signal is ~8× weaker than end-of-prompt (aggregate +0.0491 vs +0.4022); MAIN-44 hypothesis rejected. The basis is end-of-prompt-protocol-coupled, not just topic-coupled.

### fig32_mid_seq_argmax_accuracy.png
Argmax-over-23-discriminants classification accuracy per category, end-of-prompt vs mid-sequence. Aggregate drops 75.4% → 32.0% — but 6 categories *increase* under mid-seq (nature 50%→100%, emotion 50%→100%, quantifier 80%→100%, conjunction 17%→67%, pronoun 29%→43%, p_special 0%→25%), suggesting end-of-prompt's high cross-category correlation was eating accuracy for those categories. Most categories lose accuracy because the within-class direction (derived from end-of-prompt mean) doesn't align with mid-seq h.

---

## Pipeline summary

Capture → analyze → visualize → audit cycle:

```
Qwen2.5-7B base model
    │
    ▼
(testing/examples/{nla_aggregate_faithfulness.py, nla_gen_trajectory.py, nla_forced_continuation.py,
                    nla_country_concept_vector.py, nla_vocab_atlas_capture.py,
                    nla_discriminant_stability_capture.py})
    │
    ▼
testing/.cache/nla_artifacts/*.pt  (raw capture data, gitignored)
    │
    ▼
nla_geometric_features.py → geometric_features.pt
nla_pairwise_and_hotdims.py → pairwise_and_hotdims.pt
nla_sink_removed_atlas.py → sink_removed_atlas.pt
    │
    ▼
nla_visualize_geometry.py, nla_signature_atlas.py,
nla_vocab_atlas_render.py, nla_discriminant_*.py,
nla_render_interpolation_flipbook.py, nla_hierarchical_classifier.py
    │
    ▼
research/observations/figures/fig*.png  (committed)
    │
    ▼
nla_audit_findings.py — 93 PASS / 0 FAIL regression test
```

Re-running the audit verifies that all numbers in the observation markdown files match what the artifacts actually contain. If anything drifts, the audit catches it.
