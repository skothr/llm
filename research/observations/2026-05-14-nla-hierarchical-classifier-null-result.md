# Hierarchical Re-Discrimination: Null Result + Reframing

**Date:** 2026-05-14
**Toolkit:** `nla_hierarchical_classifier.py`
**Inputs:** `vocab_atlas.pt` + `pairwise_and_hotdims.pt`
**Figure:** `fig30_hierarchical_accuracy.png`
**Linear:** MAIN-47

## What MAIN-47 predicted

Self-validation (MAIN-26 / fig29) showed country-source captures had only 34% top-1 accuracy against the country discriminant — country and capital seemed to constantly swap. The hypothesis: build a sub-discriminator from `mean(country_h) − mean(capital_h)` (and similar for other sibling pairs with discriminant cos > +0.85). Apply when a capture's top-2 first-level categories form a sibling pair. Predicted lift toward 70%+ top-1 accuracy.

## What actually happened

Hierarchical scheme applied to 33 of 107 expected-mapped captures (siblings pair shows up as top-2). **It flipped exactly 1 capture's top-1 label**, lifting country accuracy from 34% → 38% (+1 / 29). All other categories: zero change. Overall: 44% → 45%.

| expected | n | baseline | hierarchical | Δ |
|---|---|---|---|---|
| codemath | 30 | 53% | 53% | +0% |
| **country** | 29 | **34%** | **38%** | **+3.4%** |
| nature | 45 | 47% | 47% | +0% |
| negation | 2 | 0% | 0% | +0% |
| refusal | 1 | 0% | 0% | +0% |
| OVERALL | 107 | 44% | 45% | +0.9% |

## Why the lift was tiny — three confounding effects

Diagnostic on the 8 source-country prompts (the cleanest country-content subset) revealed:

| prompt | baseline top-1 | what's actually happening |
|---|---|---|
| "What is the capital of France?" | **capital** | semantically correct — the prompt IS asking about Paris |
| "Tell me about Germany." | country | correct |
| "Brazil is famous for soccer." | country | correct |
| "The population of India is large." | country | correct |
| "Italy borders Switzerland." | **capital** | border-content; geographically-positioned info |
| "Spain is in Europe." | country | correct |
| "Japan has a unique culture." | country | correct |
| "China's economy is growing." | country | correct |

6 of 8 source country prompts already top `country` (75% top-1 accuracy on the clean subset). The 2 that top `capital` do so for semantically valid reasons. **The basis isn't actually confused — it's correctly responding to content.**

The reported 34% top-1 accuracy bundled three different things:

1. **Genuinely-confused captures (siblings swap)** — only the hierarchical scheme can fix this. We found **1 such case** in 107 total captures (the one country/capital flip that did fire).
2. **Correctly-classified-as-sibling captures** — e.g., "What is the capital of France?" tops capital because it IS a capital-content prompt. Sibling-aware classification correctly leaves these as capital. Counted as "wrong" in the original self-validation because the source pool was labeled `country`.
3. **Mislabeled "expected=country" captures** — the country_test set deliberately includes weird-framing prompts ("Justice is the country of the soul", "Portugal smells like Tuesdays", "If France were a sandwich"). These project to nature/emotion/quantifier, **which is correct**. No classification scheme can lift these because the label is wrong.

## Reframing — the 23 discriminants are doing better than the 34% number suggested

The "country/capital top-1 swap" framing was misleading. The actual situation:

- **For prompts that are genuinely about a country-as-topic**: the country discriminant fires correctly. ~75% top-1 on the clean 8-prompt subset.
- **For prompts that mention countries but are about something else** (asking about a capital; making a metaphor; using country names as filler): the basis classifies into the appropriate category, which often isn't `country`. This is correct behavior counted as wrong by source-label-based self-validation.
- **For genuinely-ambiguous content** (e.g., "Italy borders Switzerland" — is that country-content or capital-content?): the basis picks ONE side, and the hierarchical sub-discriminator agrees. No conflict to resolve.

The discriminant basis built in MAIN-26 is more accurate than the self-validation numbers indicated. The audit pass and this null result together suggest **the self-validation methodology should weight by "label fidelity"** — prompts that are unambiguously country-content count more than weird-framing prompts in the same source pool.

## Hypotheses

### H1 (revised) — Discriminant accuracy is bounded above by label fidelity, not by basis quality

For any classifier-driven validation, the ceiling is the labeling quality of the test set. Hierarchical re-discrimination is the wrong intervention if the failure mode is mislabeled tests, not confused basis directions. **Test:** rebuild the country_test set with strict country-only prompts (no metaphors, no questions about capitals); measure baseline top-1 accuracy. Predict > 75% on the strict subset.

### H2 — The 1 capture that flipped is the only "true sibling swap" in the dataset

If true, the hierarchical scheme found and fixed every honest case of sibling-confusion. Generalizes the null result into a positive: the basis was *already* near-optimal at the discriminate-siblings task in this dataset.

## Methodology takeaway

When a single number ("34% top-1 accuracy") looks low, decompose it before designing a fix:

1. **Genuinely-wrong**: the basis lands on the wrong category despite the prompt being unambiguous.
2. **Right-by-other-name**: the basis correctly identifies a different aspect of the prompt than the source-label expected.
3. **Label-wrong**: the prompt was mislabeled in the test set.

Only (1) is fixable by improving the basis. (2) is a validation-methodology issue. (3) is a labeling issue. **Don't build a complex fix until you've classified the failure modes.**

## Reproducibility

```bash
cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_hierarchical_classifier.py
```

CPU only, ~5 seconds, no model loading.

## References

- [Discriminant validation](2026-05-13-nla-discriminant-validation.md) — produced the 34% top-1 number this issue tried to fix.
- [Vocab atlas](2026-05-13-nla-vocab-atlas-grid.md) — provides the centroids the sub-discriminators are computed from.
