# Audit Pass: All Numerical Claims Re-Derived from Raw Sources

**Date:** 2026-05-13
**Toolkit:** `nla_audit_findings.py`
**Result:** **65 PASS / 0 FAIL.** Every load-bearing numerical claim verified from the four raw artifact files (without trusting the cached `pairwise_and_hotdims.pt` or `geometric_features.pt` intermediates).

> **Write-time snapshot.** 65 was the count on 2026-05-13. The suite was later extended (93 → 129 → **178 PASS** as of 2026-05-31, now covering the faithfulness foundation and concept-arithmetic decode identities). See [`figures/INVENTORY.md`](figures/INVENTORY.md) and the arc README for the current count.

## Motivation

A spot-check of fig15 (counterfactual glyph diff) revealed a position-drift confound — fig15's reported `||Δh||_feat = 35.55` for refusal_metaware was inflated from a true 28.06 by picking a forced token 10 positions away from natural. fig16 corrected that. The user's instruction "rigoramatize everything" called for the same scrutiny on all other claims.

## Method

`nla_audit_findings.py` loads the four raw `.pt` files (`aggregate_faithfulness.pt`, `rabbit_haiku_gen_trajectory.pt`, `forced_continuation.pt`, `country_concept_vector.pt`) and **re-derives** every load-bearing number from first principles — without trusting the cached `pairwise_and_hotdims.pt` or `geometric_features.pt` intermediates that downstream scripts used. Each derived value is compared against the published claim with a numerical tolerance (typically 0.005 or 0.01 depending on rounding); each comparison is reported as PASS/FAIL.

## Result

```
SUMMARY:  65 PASS  |  0 FAIL
```

Audits performed:
1. **Dataset size** — all 6 source-pool counts (113+15+10+8+8+13=167) verified
2. **Hot-dim census** — dim 2570 and 458 each at 165/167 in top-5, both at 167/167 in top-100
3. **Per-dim stats** — sign_consist, cv_abs, mean_val for top-20 hot dims; all match published table within 0.005
4. **Classifier labels** — the 7 sinks, 8 features, 3 polarized, 1 rare_burst, 1 background match exactly
5. **PCA variance fractions** — PC1=16.48%, PC2=7.72%, top-20=60.93%; matches published 16.5%/7.7%/60.9%
6. **Pairwise cosines** — mean off-diag +0.4031, min +0.0545, max 1.000; all intra-source values match
7. **Sink-removed cosines** — mean drops to +0.1786, min goes to -0.0686, aggregate intra collapses to +0.156, country intra holds at +0.7772; matches published
8. **CAV alignment (H3)** — direction_unit is exactly unit norm, cos(CAV_unit, e_32) = +0.051; H3 falsified
9. **AR-data deduplication** — confirmed 128 captures plotted but only 113 unique (15 haiku_gen are byte-for-byte duplicates of aggregate/creative_haiku)
10. **Counterfactual diff norms** — all 6 ||Δh||_feat values (3 clean + 3 refusal_metaware variants) match within 0.05

## Two findings the audit surfaced (not in original observations)

### Latent finding 1 — fig5 plots 128 points but only 113 are unique

The 15 haiku_gen orange points in fig5 are **byte-for-byte identical** to 15 of the 113 aggregate blue points (the ones from prompt_id `creative_haiku`). The audit verified this with `torch.allclose(a["h"], b["h"], atol=1e-5)` for all 15 pairs.

Impact on the regression fit:
- With duplicates (N=128): `cos = 0.06299 * log10(kurt) + 0.73559`
- Deduplicated (N=113):    `cos = 0.06035 * log10(kurt) + 0.74171`
- Slope shift: -0.003 (4% relative). Intercept shift: +0.006 (0.8% relative).

**The regression conclusion is unchanged** — kurtosis predicts AR cosine, with high kurt sufficient and low kurt necessary-but-not-sufficient. The asymmetry findings hold. Duplication is a cosmetic concern, not a confound.

### Latent finding 2 — off-by-one in the "sinks in CAV top-K" count

The cheap-batch observation claimed "3 of the top 8 CAV contributors are sink dims (2107, 3110, 2570)." The audit found only **2 of top 8** are sinks: 2107 (rank 6) and 3110 (rank 8). Dim 2570 contributes at rank 15, not top-8.

The qualitative finding — sinks have content-modulated components — survives. The numerical "3 of top 8" was off-by-one; corrected to "2 of top 8" in the cheap-batch observation file.

## What audit DIDN'T cover (worth keeping in mind)

- **Visual layout correctness** — verifying that figure axes/colors/legends match what's in the data is beyond automatic auditing; requires eyeballing the PNG
- **Interpretive claims** — "sinks function as the DC component of the residual," "concept axes recruit different dim subsets" are *interpretations* of the verified numbers, not numbers themselves
- **CAV's methodological soundness** — we verified the CAV direction's mathematics but didn't probe whether the difference-of-means CAV approach is the right tool. SAEs might give cleaner axes; the audit can't tell us that
- **AV / AR faithfulness** — we verified that AR cosines were as claimed, not that the AR model is itself faithful (Anthropic's training + our chat-template handling)

## Reproducibility

```bash
cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_audit_findings.py
```

CPU-only, ~20 seconds. Exits 0 if all PASS, 1 if any FAIL. Use as a regression test before any subsequent analysis: if the audit starts failing, an upstream artifact has drifted from the recorded numbers.

## References

- [Cheap-batch glyph views](2026-05-13-nla-cheap-batch-three-glyph-views.md) — the off-by-one was found here (3 → 2 sinks in top-8 CAV contributors)
- [Sink-removed atlas](2026-05-13-nla-sink-removed-atlas.md) — all 7 sink dims and intra/cross cosines verified
- [Geometric deep dive](2026-05-13-nla-geometric-deep-dive.md) — per-dim stats, classifier, PCA variance fractions all verified
- fig16 — the position-drift correction that initiated this audit pass
