# Discriminant Validation: Connectivity, Self-Validation, and Stability

**Date:** 2026-05-13
**Toolkit:** `nla_discriminant_connectivity.py`, `nla_discriminant_stability_capture.py`, `nla_discriminant_stability_render.py`
**Figures:** `fig27_discriminant_connectivity.png`, `fig28_discriminant_stability.png`, `fig29_self_validation.png`
**Data:** `discriminant_stability.pt`

## Motivation

After fixing the "all axes active" issue with discriminant directions (fig25/fig26 successors to fig23/fig24), three validations were needed to confirm the basis is **coherent**, **stable**, and actually **discriminating** between concepts:

1. Are the 23 axes themselves coherent — do semantic siblings produce correlated axes; do opposites produce anti-correlated ones? → **fig27 connectivity**
2. Do existing captures with known content project onto the expected discriminant? → **fig29 self-validation**
3. Does the same anchor word in different contexts produce consistent discriminant readouts? → **fig28 stability**

## Result summary

**Discriminate:** PASS (mean cross-axis cosine +0.006).
**Coherent:** PASS (clear macro-cluster structure in fig27).
**Stable:** MIXED — varies dramatically by category.
**Token-presence detection:** FAIL — the discriminants detect **prompt-topic**, not **token-presence-in-prompt**.

## Finding 1 — Discriminant connectivity reveals three macro-clusters (fig27)

The 23×23 discriminant pairwise cosine matrix has clear block structure:

* **CONTENT macro-cluster** (top-left 6 categories): country, capital, nature, codemath, emotion, refusal. Mutually positive within; anti-correlated with everything below.
* **FUNCTION-WORD macro-cluster** (middle 9): article, pronoun, demonstrative, preposition, conjunction, auxiliary, negation, quantifier, wh_word. Own correlated block.
* **STRUCTURAL macro-cluster** (bottom-right 8): p_ender, p_internal, p_quote, p_bracket, p_dash, p_special, number, math_op. **Tightest block of all** — non-content tokens are represented as one super-category at h[20].

Top semantic siblings (highest discriminant cosine):
- `country ↔ capital`: **+0.938**
- `p_special ↔ math_op`: +0.843
- `preposition ↔ auxiliary`: +0.793

Top semantic opposites (most negative):
- `country ↔ demonstrative`: -0.638
- `nature ↔ auxiliary`: -0.605
- `emotion ↔ preposition`: -0.597

The PC1 of fig19 (content-vs-function axis) is reproduced as a pairwise structure here. Content categories anti-correlate with function categories; function correlates with structural less directly. **The model's mid-late residual is hierarchically organized — macro-categories with sub-categories — and this organization is recoverable from the data.**

## Finding 2 — Self-validation: top-5 hit rates 56-79% (fig29)

Projecting all 167 existing captures onto each discriminant, then asking "did the expected category top the projection?":

| expected | n | top-1 | top-3 | top-5 | mean rank |
|---|---|---|---|---|---|
| country (country pool) | 29 | 34% | 76% | **79%** | 3.6 |
| codemath (aggregate/code,math) | 30 | 53% | 63% | 73% | 3.5 |
| nature (haiku, creative) | 45 | 47% | 49% | 56% | 5.2 |
| refusal (forced/refuse) | 1 | 0% | 100% | 100% | 1.0 |

The 79% top-5 + mean-rank 3.6 for country means: country captures **always end up in the content macro-cluster** but country and capital constantly swap top-1 (they're +0.938 siblings — h's fitting one fit the other). The macro-classification is reliable; sibling distinction is harder.

Nature is weakest because haiku tokens span many adjacent categories (nature, emotion, codemath as a single content cluster).

## Finding 3 — Stability varies dramatically by category (fig28)

8 anchors × 4 prefix-length variants × capture at position -1 of the chat-templated sequence. *Note (added 2026-05-29): position -1 is the trailing token of the `<|im_start|>assistant\n` generation prefix, NOT the anchor token itself — what we measure is the model's "what to say next given this prompt" representation. Comparisons across contexts are internally consistent (all four contexts capture at this same kind of position), so the stability finding holds, but the framing should be "end-of-prompt stability across prefix lengths" rather than "anchor-token stability".*:

| anchor | category | ctx-cos (4 contexts) | expected-projection mean | std |
|---|---|---|---|---|
| `the` | article | **+0.917** | +0.183 | ±0.139 |
| `France` | country | +0.851 | +0.219 | ±0.129 |
| `Paris` | capital | +0.815 | +0.185 | ±0.135 |
| `7` | number | +0.815 | +0.281 | ±0.130 |
| `function` | codemath | +0.801 | **+0.284** | ±0.086 |
| `.` | p_ender | +0.799 | +0.119 | ±0.153 (std > mean) |
| `refuse` | refusal | +0.589 | +0.052 | ±0.130 (std ≫ mean) |
| `happy` | emotion | **+0.399** | +0.083 | ±0.061 |

**Two stability classes** (re-framed 2026-05-29 under the corrected "end-of-prompt response-planning" semantics — see note at the top of this finding):
- **Stable anchors** (ctx-cos > +0.80): function-words, structural tokens, place names, numbers. After these prompts, the model's "what to say next" representation is similar regardless of prefix length — whether the user said "France" alone, "Tell me about France," or "I want to discuss with you the following word, which is France," the model's response-planning state is dominated by the final content token + assistant-turn opener. The prefix prose is integrated but doesn't deflect the plan.
- **Unstable anchors** (ctx-cos < +0.60): `happy` and `refuse`. After these prompts, the model's response-planning state shifts substantially with prefix length. Plausibly: factual/structural anchors have a narrow "what one says about this" distribution (definitional / identifier-like), while emotional and refusal-laden anchors have a wider distribution that the surrounding prose actually narrows — the model has to figure out whether you want a definition, an empathetic response, a meta-discussion, or something else.

## The major finding — the discriminants detect prompt-topic, NOT token-presence

Look at the expected-projection column: even the BEST case (`function` → codemath) is only +0.28. For `happy` → emotion it's +0.083. For `refuse` → refusal it's +0.052 with **std exceeding the mean** (sometimes the projection is *negative*).

Watching the glyphs in fig28: `happy`-as-single-token-message projects onto emotion (top-1), but `happy` inside "Tell me about happy" or "I want to discuss the word happy" projects onto **codemath**. The model isn't representing "the token 'happy' is here" — it's representing "this is a question asking about a word" (which feels factual/definitional, hence codemath).

**The discriminants are doing prompt-TOPIC classification, not token-presence detection.** When the entire user message is about country geography, the h projects onto country. When the entire user message asks about the word "refuse" in a meta way, the h projects onto wh_word or demonstrative — *because the message is asking about a word*, not refusing anything.

This re-frames what our 23-axis basis actually represents:
- **Good for**: classifying the overall topic/register of a complete prompt
- **Bad for**: detecting whether a specific token appears in the prompt
- **Reason**: layer 20 at end-of-prompt has integrated the entire user message into a "what is this prompt about?" representation

## Implication for the visualization research

This is a **scope-clarifying finding**. The discriminant glyphs we built (fig25, fig26, fig28) are valid as a representation of **what the model thinks the prompt is about**, not as a representation of **what tokens the prompt contains**. The interpolation flipbook's cascade (fig21: France → autumn → snow) showed the same thing: as t varies, the model's "this prompt is about" topic shifts, not the model's "tokens in the prompt" list.

For future work:
1. **Build separate per-token discriminants** using mid-sequence captures (h[20] at specific positions other than end-of-prompt) if we want token-presence detection
2. **Treat the existing 23-discriminant basis as a prompt-topic classifier** — that's its actual function
3. **The category-attractor structure (intra-cos +0.85 within categories) is what makes the basis work for topic classification** — every prompt about a country pulls h toward the country attractor regardless of which specific country, hence good top-K detection but weak top-1

## Reproducibility

```bash
cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research

# fig27 + fig29 (cheap, no model loading, ~30 sec)
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_discriminant_connectivity.py

# fig28 — capture (~3 min CPU forward passes) + render (~10 sec)
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_discriminant_stability_capture.py
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_discriminant_stability_render.py
```

## References

- [Discriminant glyph](2026-05-13-nla-vocab-atlas-grid.md) — the primitive being validated here.
- [Vocab atlas](2026-05-13-nla-vocab-atlas-grid.md) — same data the centroids and discriminants are computed from.
- [Sink-removed atlas](2026-05-13-nla-sink-removed-atlas.md) — sink removal applied throughout.
- Kim et al., 2018, "TCAV" — discriminant directions generalize the binary CAV to multi-class.
