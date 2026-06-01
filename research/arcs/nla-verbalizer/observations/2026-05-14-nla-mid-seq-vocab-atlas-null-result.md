# Mid-Sequence Vocab Atlas: Null Result + Refines MAIN-26

**Date:** 2026-05-14
**Toolkit:** `nla_mid_seq_vocab_atlas_capture.py`, `nla_mid_seq_vocab_atlas_compare.py`, `nla_mid_seq_vocab_atlas_render.py`
**Inputs:** `vocab_atlas.pt` (end-of-prompt), `pairwise_and_hotdims.pt` (sink dim labels)
**Outputs:** `mid_seq_vocab_atlas.pt`, `mid_seq_compare.pt`
**Figures:** `fig31_mid_seq_signal_vs_noise.png`, `fig32_mid_seq_argmax_accuracy.png`
**Linear:** MAIN-44

## What MAIN-44 predicted

MAIN-26 ([discriminant-validation](2026-05-13-nla-discriminant-validation.md)) found the 23-discriminant basis acts as a prompt-TOPIC classifier rather than a token-presence detector — `happy → emotion` projection was only +0.083 ± 0.061 across 4 prefix-length contexts at end-of-prompt positions. **Interpretation at the time:** end-of-prompt h has integrated the full user message into a topic representation, drowning token-specific identity. **Predicted fix:** capture each anchor MID-SEQUENCE (inside a carrier prompt that continues past the anchor) and project onto the existing discriminants. If the integration-overwhelm interpretation holds, `expected_proj` should rise to ≥ +0.3.

## What actually happened

Captured h[20] for all 128 vocab anchors × 23 categories at the anchor token's mid-sequence position inside the carrier:

```
"The text contains many words. Here is one specific word: {anchor} continues throughout subsequent discussion paragraphs."
```

Anchor lands at position 36-38 of ~48 tokens (~75% of the sequence; 10-12 tokens of trailing context). Position-finding via prefix-tokenization (BPE is left-to-right; tokenize `chat_str[:span_end]` to count tokens in the anchor span). All 128 anchors captured with 0 skips.

**Projected mid-sequence h's onto the end-of-prompt-derived discriminants:**

| Protocol | mean within-class signal | mean max-off-class noise | argmax accuracy |
|---|---|---|---|
| end-of-prompt (baseline) | +0.4022 | +0.3911 | 75.37% |
| mid-sequence (this test) | **+0.0491** | +0.0623 | **32.04%** |

The mid-sequence within-class signal is **~8× weaker** than end-of-prompt, not stronger. The MAIN-44 hypothesis is **rejected** — mid-sequence positioning does not enable token-presence detection. It produces signal nearly orthogonal to the basis.

Specifically for the MAIN-26 anchor "happy" (emotion):
- end-of-prompt projection onto emotion discriminant: +0.1755 (argmax correct)
- mid-sequence projection: +0.0759 (still argmax correct, but signal dropped 57%)

## Why the prediction was wrong — what mid-sequence really shows

At mid-sequence, h projects approximately uniformly weakly across the entire 23-discriminant subspace. Expected magnitude of a random unit vector's cosine projection in R^3577 is ~1/√3577 ≈ 0.017; the observed +0.05 is only 3× that floor. **The 23-discriminant basis is more specifically end-of-prompt-protocol-coupled than just "prompt-topic".** Mid-sequence h's live in a different region of layer-20 residual space — different enough that the within-class direction (derived from end-of-prompt geometry) doesn't preferentially align with mid-sequence captures of the same category.

This **refines MAIN-26's claim**:
- MAIN-26: basis classifies prompt-TOPIC, not token-presence
- MAIN-44 refinement: basis is **protocol-coupled to end-of-prompt position** — it doesn't transfer even to the same anchor token captured at a different sequence position. The "topic" interpretation was a category error: the basis isn't tracking topic; it's tracking *whatever h represents at end-of-prompt in this specific chat-template-wrapped protocol*.

## The interesting wrinkle: 6 categories actually classify BETTER mid-seq

While aggregate accuracy drops 75% → 32%, some categories' argmax accuracy *increased* under the mid-sequence protocol:

| category | eop argmax acc | mid argmax acc |
|---|---|---|
| nature | 50% | **100%** |
| emotion | 50% | **100%** |
| quantifier | 80% | **100%** |
| conjunction | 17% | **67%** |
| pronoun | 29% | **43%** |
| p_special | 0% | 25% |

For these categories, mid-sequence h has a *cleaner* directional signal — the discriminant's noise floor falls faster than its signal does. The pattern suggests the end-of-prompt protocol's high cross-category correlation (mean centroid cos +0.85 at eop, per fig25) was eating into accuracy at eop, and the mid-seq subspace removes that confound for some categories while damaging others.

## What we now know about layer-20 geometry

Three protocols, three different h subspaces, ranked by within-class signal on the end-of-prompt-derived basis:

1. **End-of-prompt of {anchor} alone** (vocab atlas, ~3 tokens) — signal +0.40, accuracy 75%
2. **End-of-prompt of stability-scan prefixed prompt** (MAIN-26, ~5-15 tokens) — signal +0.08-0.18, accuracy not measured
3. **Mid-sequence inside long carrier** (this work, ~48 tokens, anchor at pos 36-38) — signal +0.05, accuracy 32%

All three "represent the anchor token" in some sense, but the resulting h's live in measurably different parts of layer-20 space. **The basis is a fingerprint of one specific protocol, not a generic semantic axis.**

## Implications for the glyph viz primitive

The discriminant glyph (fig25/26) is the right primitive **only for h's captured at end-of-single-token-user-message position**. Applying it to h captured at other positions — generation steps mid-trajectory, mid-prompt captures, interpolation-derived h — produces glyphs whose rays don't carry the intended semantic interpretation. The interpolation flipbook (fig17/fig25) is borderline: anchors A and B are AR-encoded h's (not end-of-prompt captures), and the interpolated h's live somewhere those discriminants weren't designed to cover. The strong fig21 result (capitals→nature anchor switch at t=0.421) used cosine-to-vocab-anchor projection rather than discriminants — and that result still holds because it doesn't go through this protocol-coupled basis.

## Next experiments this opens

1. **Build mid-seq-NATIVE discriminants** from these 128 captures and project mid-seq h's onto those. Predict within-class signal ~+0.4 (matching the in-protocol signal at eop). If so, the basis-design protocol-coupling is confirmed and we have a per-protocol family of bases.
2. **Cross-protocol cosine alignment** between eop_category_mean and mid_category_mean per category. Maps how each category's representation drifts across protocols. This would tell us whether category-specific h has a stable "axis" across protocols (just rotated by the protocol) or whether categories are constructed differently at each position.
3. **Single-position vs multi-position discriminants:** Capture each anchor at 4 positions in the same carrier (position k=10, k=20, k=30, k=40). Test whether discriminants built from multi-position pooling generalize better than single-protocol ones. Speaks to whether "category" has a position-invariant axis at all in this layer.

## Cross-arc lessons

This is the **second null result of the arc** (MAIN-47 was the first). Both came from over-extrapolating a single finding. MAIN-26 looked at one position protocol (end-of-prompt of varying-length prefix) and produced an interpretation ("end-of-prompt integration overwhelms token identity") that turned out too narrow. MAIN-44's failure refines it: the integration interpretation can't explain why mid-sequence is also a different subspace — the basis is protocol-coupled in a more general way.

When designing the next round of probes, the methodological discipline: **always test the basis at the protocol it was derived from AND at one alternative protocol before claiming the basis "measures" something interpretable.** Within-protocol signal is necessary but not sufficient — the cross-protocol behavior reveals what the basis *isn't* doing.
