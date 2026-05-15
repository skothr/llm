# Concept Arithmetic Atlas: Category-Level Composition Preserves; Specific-Identity Analogies Fail

**Date:** 2026-05-14
**Toolkit:** `nla_concept_arithmetic_atlas.py`, `nla_concept_arithmetic_render.py`
**Inputs:** `vocab_atlas.pt`, AV `kitft/nla-qwen2.5-7b-L20-av`
**Output:** `concept_arithmetic_atlas.pt`
**Figure:** `fig35_concept_arithmetic_atlas.png`
**Linear:** MAIN-48

## What MAIN-48 predicted

Does NLA preserve word2vec-style additive/subtractive structure on layer-20 end-of-prompt h? If `vec(Paris) − vec(France) + vec(Germany) ≈ vec(Berlin)`, then the AV becomes a composable interpretation tool — every axis labeled, every arithmetic combination decodable. Tested 7 combinations (3 canonical analogies, 1 pure subtraction, 2 axis directions, 1 compound concept), rescaled to ||h||=150 (typical eop norm) before AV-decoding.

## What actually happened

The picture is **nuanced — not a clean yes or no.** Three observed patterns:

### A) Specific-identity analogies FAIL

| arithmetic | predicted | AV decoded (key content) |
|---|---|---|
| `Paris − France + Germany` | Berlin | **London** ("about the city of London") |
| `Tokyo − Japan + France` | Paris | **Spain** ("about the country") |
| `Berlin − Germany + United Kingdom` | London | **UK** (just outputs UK) |

None of the three analogies produced the predicted specific capital. Test #1 produced a capital (London) — wrong identity but **right category**. Test #2 produced a country (Spain) — wrong category entirely. Test #3 collapsed back to the +input term (UK).

The pattern is **inconsistent**: sometimes the arithmetic moves the h into the right category but picks a wrong identity within it; sometimes it loses the category direction entirely. word2vec-style analogies do not survive the NLA encoding at this layer.

### B) Category-level axis direction DOES preserve

`country_centroid − capital_centroid` (axis direction) decoded coherently as country-flavored content: "this country's population is diverse", "the world's religions", "a country's facts", "a document" — strongly country-themed rather than city-themed. The category-level axis is preserved through subtractive composition.

### C) Compound (additive) shows dominant-component behavior

`country_centroid + emotion_centroid` decoded as **China** (country-themed). The country centroid is larger in magnitude than the emotion centroid (after sink removal), so its direction dominates the sum. The emotion contribution is invisible. **Additive composition doesn't average the two concepts; it follows the larger one.**

### D) Pure subtraction of similar-magnitude vectors yields noise

`France − Germany` (||raw||=9.24) rescaled to ||h||=150 amplifies noise. Decoded as incoherent content: a Chinese movie reference ("Zhang Ziyi's Dream of Red Chamber"), "Je ne sais quoi" (a French phrase — faint French signal), then drift. The rescaling-of-noise problem.

Similarly `happy − sad` (||raw||=49.98) rescaled to 150 decoded as: "Fibonacci Zoo", "prize display" — also incoherent.

## What this tells us about NLA's representation geometry

NLA layer-20 representations are **categorically structured but not algebraically composable in the word2vec sense**:

1. **Category direction is a robust axis** (consistent with `country − capital` working and with MAIN-70's fig34 showing emotion has the strongest cross-protocol axis stability at +0.17).
2. **Within-category specific identity is encoded differently than across-category position.** The "Paris-ness within capital-ness" direction isn't the same as the "Berlin-ness within capital-ness" direction shifted in some consistent way. word2vec's geometric assumption — that `vec(France) − vec(Paris)` equals `vec(Germany) − vec(Berlin)` — does not hold in this layer.
3. **Sum of two concept vectors collapses to the larger-magnitude one.** The smaller concept contribution gets absorbed without changing the dominant decode. This is not the same as averaging the two concepts.

This refines the MAIN-44 + MAIN-70 picture: the basis is protocol-coupled (per-protocol family of discriminants), and even within a protocol, the arithmetic structure of h is **categorical, not algebraic**. Categories have stable axes (✓); within-category positions are constructed in a way that doesn't compose by subtraction (✗).

## Implications for viz primitives

A "concept arithmetic" UI surface — where the user combines tokens via + and − and watches the result decode — would produce mostly category-level results, not specific-identity transformations. The UI affordance is honest: show that arithmetic moves the h into a category direction (visible in discriminant glyph) but specific decode identity isn't algebraically predictable. Useful for exploring category axes; not useful as a "what would change if I rotate this analogy" probe.

A more productive use: arithmetic to **isolate axes** rather than predict tokens. `mean(country) − mean(capital)` gives a clean country axis. `h(happy) − mean(neutral_emotion)` would give a "happy-specific direction within emotion" — testable in a follow-up.

## Cross-arc lessons

The arc now has three converging findings:
- **MAIN-44/70**: basis is protocol-coupled; content categories have small position-invariant component.
- **MAIN-48 (this)**: within a protocol, arithmetic structure is categorical not algebraic.
- **MAIN-25** (the headline): stepwise category-flip at t=0.421 during linear interpolation between AR-encoded anchors.

Reading these together: at layer 20, h-space has **discrete category attractors** rather than a smooth product-of-axes geometry. Interpolation flips between attractors (MAIN-25); arithmetic moves between attractor regions but doesn't smoothly navigate within them (MAIN-48); discriminants pick out the attractor regions per protocol (MAIN-70). The next probe should test the attractor hypothesis directly: dense interpolation near t=0.421 (MAIN-34) to see if the flip is a sharp discontinuity or a smooth-but-fast transition. That's already filed.
