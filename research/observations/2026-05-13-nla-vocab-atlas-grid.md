# Vocab Atlas: 128-Anchor Semantic Basis Grid in h[20] Space

**Date:** 2026-05-13
**Toolkit:** `nla_vocab_atlas_capture.py` + `nla_vocab_atlas_render.py`
**Model:** Qwen/Qwen2.5-7B-Instruct, layer 20 (CPU bf16)
**Inputs:** 128 anchor tokens, each captured at end-of-single-token-user-message in a chat-templated prompt
**Figures:** `fig19_vocab_atlas.png`, `fig20_combined_atlas.png`, `fig21_interp_through_anchors.png`, `fig22_anchor_cosine_matrix.png`
**Data:** `testing/.cache/nla_artifacts/vocab_atlas.pt`

## Goal

User's framing: "map a bunch of relevant tokens' embeddings to get a relative baseline for semantic bases." Build a **named-anchor grid** that gives interpretable reference directions in h[20] space, complementing the anonymous dim indices we've been using as glyph rays.

## Vocabulary (128 anchors, 23 categories)

* **Content** (43): 10 countries, 5 capitals, 12 nature/seasons, 6 codemath, 6 emotion, 4 refusal
* **Function words** (52): 3 articles, 7 pronouns, 4 demonstratives, 10 prepositions, 6 conjunctions, 8 auxiliaries, 3 negations, 5 quantifiers, 6 wh-words
* **Punctuation** (18): 3 enders, 3 internal, 2 quote, 4 bracket, 2 dash, 4 special
* **Numbers/operators** (15): 11 digits, 4 math ops

Protocol: each anchor as the entire user-message text in a chat-templated prompt; capture `h[20][0, -1, :]` — same protocol as the country pool, so the new captures are directly comparable.

Norms ranged from 92 to 103 across all 128 — uniform, no outliers.

## Finding 1 — Hierarchical attractor structure

All 23 categories have intra-category cosine > **+0.84 even after sink removal**:

| Category | n | intra-cos |
|---|---|---|
| capital | 5 | +0.983 |
| country | 10 | +0.981 |
| number | 11 | +0.966 |
| demonstrative | 4 | +0.965 |
| auxiliary | 8 | +0.958 |
| preposition | 10 | +0.943 |
| article | 3 | +0.928 |
| nature | 12 | +0.917 |
| codemath | 6 | +0.896 |
| conjunction | 6 | +0.896 |
| pronoun | 7 | +0.881 |
| emotion | 6 | +0.849 |
| refusal | 4 | +0.847 |

This wasn't the sink offset (those were removed). There's a separate **category-attractor subspace** that pulls members together to +0.85 to +0.98 cosine. The h-vector hierarchy now looks like:

| Attractor level | Effect on cosine |
|---|---|
| Universal sinks (the 7) | adds ~+0.22 to every pair |
| Universal-but-not-sink residue | adds another ~+0.4 baseline |
| Category-specific attractor | adds +0.4 to +0.6 *within* category |
| Within-category content modulation | the remaining +0.01 to +0.15 |

The two loosest categories (refusal +0.847, emotion +0.849) span genuinely different semantic dimensions — "refuse" vs "sorry" vs "cannot" have different connotations; "happy" vs "sad" are opposite valences. Tighter categories (capital +0.983) contain near-substitutable concepts.

## Finding 2 — PC1 of the vocab-only atlas is a content-vs-function axis

PCA on the 128-anchor matrix (sink-removed) has:
* PC1 = 33.5%
* PC2 = 15.3%
* PC3 = 7.1%
* top-3 = 55.9%

Much higher concentration than the 167-capture set (top-3 ~28%). Single-position end-of-prompt captures occupy a lower-dimensional manifold.

PC1 cleanly separates **content-bearing** (left, PC1 < 0) from **structural/function** (right, PC1 > 0). PC2 separates valence/format (top: capitals/emotions/wh-words; bottom: codemath/punctuation/numbers). **The model's mid-late residual is organized along a "what is this token about?" vs "what role does it play?" axis as the dominant content direction.**

Within the labeled atlas (fig19), country and capital clusters overlap — Paris and France are essentially identical at h[20]. The model represents "capital-or-country abstract category" more strongly than token-specific identity at this layer.

## Finding 3 — Interpolation trajectory pivots at t=0.421 in TWO independent measurements

From fig17 (AV-text reading) and fig21 (cosine-to-anchor): both show the **same** transition step.

| t | top-3 nearest vocab anchors | cosine |
|---|---|---|
| 0.000 → 0.368 | Madrid, Berlin, Tokyo | +0.44-0.45 |
| **0.421** | **autumn, snow, Berlin** | **+0.44 (Berlin barely hangs on)** |
| 0.474 → 1.000 | autumn, snow, sky (joy late) | +0.37-0.43 |

**Two independent signals confirm the discrete semantic-region structure** at exactly t=0.421:
1. AV text format-word changes from "factual" to "poem"
2. Top-3 nearest anchors flip from capital-category to nature-category

The transition is *anchor-confirmed*, not just an AV interpretation artifact.

**Additional observation:** Anchor A (which specified "France / Paris") maps to capital-category anchors generally (Madrid, Berlin, Tokyo), **not to Paris specifically**. The AR-encoded h matches the abstract "capital-of-X" attractor rather than the concrete instance. The category attractor is what survives the round-trip.

**Asymmetric anchor matching:** Anchor A → capital-category at cos +0.45 (tight). Anchor B → nature-category at cos +0.37-0.38 (looser). The poetic-nature region is geometrically broader; butterflies-as-specific-subject is one valid filler among many.

## Finding 4 — Prompt-structure dominates PCA when corpora are combined

fig20 (combined 207-vector PCA): the 128 vocab anchors and 167 existing captures occupy **disjoint regions** of PC space (PC1 right vs left). Their cross-similarity in raw PC coordinates is zero, even though many existing captures should be semantically near a vocab anchor.

**Reason:** prompt length and structural position dominate h[20]'s variance after sink removal. Vocab anchors are captured at end-of-single-token-prompt; existing captures are mid-generation or end-of-multi-token-prompts. The chat-template structural setup translates h-vectors differently for different prompt configurations, and that translation is the dominant axis when corpora are mixed.

**Methodology fix:** combined-dataset PCA is misleading; use **cosine-to-anchor** (which is invariant to the structural translation) for cross-corpus comparison. fig21 already does this and produces the correct semantic structure.

## Hypotheses

### H11 — There's a "category subspace" separate from the sink subspace

We removed 7 sink dims. Categories still cluster at +0.85+ intra-cosine. Either:
* (a) There are 7+ more "category" dims we haven't identified, or
* (b) The category attractor is a low-dimensional subspace spanned by linear combinations of many dims, not axis-aligned.

**Test:** for each category, compute the principal direction (mean-of-category, normalized after sink removal). Check whether these 23 directions span a low-rank subspace (small SVD spectrum) or are themselves orthogonal-ish.

### H12 — PC1 of vocab-only ≈ "content vs structural function" is a model invariant

The same vocab capture protocol on a different open-source model (TinyLlama, OpenLLaMA) should produce a similar PC1 axis. **Test:** repeat the capture on TinyLlama-1.1B at its equivalent ~80% depth layer; compare top-2 PCs.

### H13 — Position-conditioning is the largest variance source in h[20]

The reason vocab anchors and existing captures don't overlap in PCA is that h[20] varies more by position than by content. **Test:** capture h[20] at several positions for the same anchor word (start-of-prompt, middle, end, post-template). Predict that within-anchor cross-position cosines are *lower* than within-category cross-anchor cosines — meaning position effects exceed content effects.

## Design implications for visualization

1. **Always use cosine projection, not raw PCA coordinates**, when comparing across capture protocols. PCA conflates structural variation with content variation.

2. **Anchor-based projection is the missing primitive.** Instead of dim-indexed glyphs (anonymous), we can build **anchor-projection glyphs** where each ray is a NAMED concept (country, nature, code, refusal, etc.) and ray length is the projection of h onto the category's centroid direction. Same primitive, semantically labeled rays.

3. **Layer 20 is position-conditioned more than content-conditioned.** This is a finding the vocab atlas surfaces. Mid-generation h's live in a different region than end-of-prompt h's even for the same content. Future analyses should segregate by position.

## Followups

1. **Build the anchor-projection glyph.** For any h, compute its cosine to each category centroid; render as a 23-ray glyph with category-named axes.
2. **Position scan.** Capture h[20] for "France" at five different positions in a longer prompt; quantify position vs content variance.
3. **Concept arithmetic in anchor space.** Compute "country - capital" as a difference of centroids; AV-decode it. Does it read as "country-ness that isn't capital-ness"?
4. **Replicate H12 on TinyLlama.** Confirm or refute model-invariance of the content-vs-function PC1 axis.
5. **Audit extension.** Add vocab-atlas claims to `nla_audit_findings.py`.

## Reproducibility

```bash
cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research

# ~13 min: load Qwen, capture 128 anchors
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_vocab_atlas_capture.py

# ~30s: render fig19-22 from artifacts
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_vocab_atlas_render.py
```

## References

- [Sink-removed atlas](2026-05-13-nla-sink-removed-atlas.md) — established the sink-removal preprocessing this builds on.
- [Interpolation flipbook](2026-05-13-nla-interpolation-flipbook.md) — fig21 here directly verifies fig17's t=0.421 pivot with an independent measurement.
- [Geometric deep dive](2026-05-13-nla-geometric-deep-dive.md) — sink dims [277, 458, 1427, 1627, 2107, 2570, 3110] used in sink removal.
- [CAV country direction](2026-05-13-nla-cav-country-direction.md) — H11 generalizes the CAV idea to 23 simultaneous concept directions.
