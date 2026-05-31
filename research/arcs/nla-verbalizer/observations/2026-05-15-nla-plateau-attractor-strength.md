# Plateau Attractor Strength: The Hybrid Plateau Is a Real Basin

**Date:** 2026-05-15
**Toolkit:** `nla_plateau_attractor_test.py`
**Inputs:** `dense_interp_near_pivot.pt` + AR `kitft/nla-qwen2.5-7b-L20-ar`
**Output:** `plateau_attractor_test.pt`
**Figures:** (none — this is a numerical-table finding)
**Linear:** MAIN-71 (part 2 of 2)

## What MAIN-71 tested

MAIN-34 found a stable "Definition + Poem" hybrid plateau spanning t ∈ [0.395, 0.4450] in the linear interpolation between AR-encoded anchors A (factual/geography) and B (poetic/nature). Question: is this plateau a **true attractor basin** (AR re-encoding stays in the basin) or just a transit zone (AR re-encoding drifts toward an anchor)?

Procedure: take plateau-mid h (from dense interp step t=0.420), run its AV text through AR to produce h_pred, compare h_pred to original h and to anchors via cosine. Baseline references: same round-trip for h_A (t=0.000) and h_B (t=1.000).

## Results

| target | ||h_orig|| | ||h_pred|| | cos(h_pred, original) | cos(h_pred, A) | cos(h_pred, B) | cos(h_pred, plateau) |
|---|---|---|---|---|---|---|
| **plateau t=0.420** | 60.79 | 65.68 | **+0.8995** | +0.8386 | +0.8154 | +0.8995 |
| anchor A t=0.000 | 65.73 | 64.65 | +0.8900 | +0.8900 | +0.6416 | +0.8521 |
| anchor B t=1.000 | 66.30 | 65.78 | +0.8989 | +0.6431 | +0.8989 | +0.8151 |

## Interpretation

**Plateau round-trip cosine of +0.8995 passes the strong-attractor threshold (>+0.85).** The plateau-AV text, when AR-encoded back to h-space, lands closer to the original plateau h (+0.8995) than to either anchor (+0.8386 to A, +0.8154 to B). The plateau is a basin, not a transit zone.

But the picture has a subtler layer worth flagging:

**All three round-trip cosines are ~+0.89** — AR self-consistency is approximately uniform across the three test points. The relevant signal isn't the round-trip cosine alone, but the **margin** between cos(h_pred, original) and cos(h_pred, other_anchors):

| target | self-consistency | margin to anchor-rivals |
|---|---|---|
| anchor A | +0.890 vs +0.642 (B) | **+0.248** |
| anchor B | +0.899 vs +0.643 (A) | **+0.256** |
| **plateau** | +0.900 vs +0.839 (A) | **+0.061** |

The plateau's margin is much smaller than the anchors' margins. This is geometrically consistent — the plateau is in between the anchors, so an AR reconstruction in the plateau region is naturally somewhat close to both anchors. But it also means the plateau basin is **narrower** than the anchor basins: a small perturbation in h-space might tip the AR re-encoding out of the plateau and into one of the neighboring basins.

**Norm note:** plateau ||h_orig||=60.79 (the magnitude dip we saw in fig37) but ||h_pred||=65.68 — AR reconstruction restored the magnitude to typical anchor-level (~65-66). The AR value head appears to project to canonical magnitude regardless of the input AV text's source. This means the plateau basin is direction-coupled but not magnitude-coupled to its specific dip-point geometry.

## Stronger statement we can make now

Combining MAIN-25, MAIN-34, MAIN-44/70, MAIN-48, and this work:

**Layer-20 h-space has discrete attractor basins separated by sharp boundaries. Basins include both the named vocab categories AND hybrid combinations not in the original atlas (like "Definition + Poem"). AR re-encoding of a basin's AV-text returns h to the basin's directional region. The basin structure is direction-coupled, not magnitude-coupled. Linear interpolation between two basin-residing h's traverses one or more intermediate basins with sharp boundaries between them.**

*Scope (matches README F1/L9): hold this as a **working hypothesis**, not a settled property of layer 20. It rests on a single anchor pair (factual/geography ↔ poetic/nature) and a plateau margin of only +0.061 over the nearest single-anchor — narrower than the ~+0.25 inter-anchor margins (see the margin note above). The honest framing is "basin candidate at this one location" until other anchor pairs and layers replicate (D5/D6).*

This is the strongest synthesis the arc has produced. It explains:
- Why category arithmetic fails for specific identities (MAIN-48): basins, not algebraic offsets
- Why discriminants are protocol-coupled (MAIN-44/70): each protocol's captures land in protocol-specific subsets of basins
- Why interpolation produces stepwise flips (MAIN-25): basin boundaries are sharp
- Why dense sampling reveals plateaus (MAIN-34): basins have non-zero volume in h-space

## Follow-ups this opens

- **Map more hybrid basins.** Try anchor pairs across different content domains: code↔nature, math↔emotion, factual↔refusal. Each pair likely has its own intermediate basin(s). Atlas-of-basins as a viz primitive.
- **Test margin scaling.** Plateau margin to anchors was 0.061 here. If the plateau h is perturbed by Gaussian noise of varying magnitude, at what perturbation level does AR re-encoding leave the basin? This characterizes basin width quantitatively.
- **Probe basin self-consistency at higher resolution.** Multiple round-trips: h → AV → AR → h' → AV' → AR → h''. Does this iterate converge to a fixed point (stable attractor with well-defined center) or wander?
