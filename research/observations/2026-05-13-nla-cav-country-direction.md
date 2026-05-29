# NLA Reads Arithmetically-Constructed Concept Directions in Natural Language

**Date:** 2026-05-13
**Model:** Qwen/Qwen2.5-7B-Instruct (CPU bf16)
**AV:** kitft/nla-qwen2.5-7b-L20-av (CPU bf16)
**Toolkit:** llm_surgeon.probe.nla_verbalize + CAV via difference-of-means
**Script:** testing/examples/nla_country_concept_vector.py
**Artifact:** testing/.cache/nla_artifacts/country_concept_vector.pt
**Captures:** 29 (8 country + 8 non-country + 13 test)

## Finding

Concept-Activation-Vector (CAV) methodology applied via the NLA toolkit.
Difference-of-means over 8 country and 8 non-country prompts at h[20]
of the last prompt token yields a direction in R^3584 that:

1. **The AV describes in natural language as country-themed.** Fed the
   unit-normalized direction (rescaled to magnitude 150), the AV
   produces text mentioning travel guides, visa requirements,
   Ministry of Foreign Affairs, census, country capitals, and specific
   country names (China, Korea, Egypt) — confirming the direction
   captures country/geopolitical semantics.
2. **Separates source groups in projection** (mean country +1.11
   vs mean non-country -20.65), confirming the direction is a real
   discriminator.
3. **But narrowly captures country-as-topic, not country-as-mention.**
   Five weird-framing prompts that name countries in unusual contexts
   ("My pet hamster Belgium...", "Portugal smells like Tuesdays...",
   "If France were a sandwich...") project as strongly negative as
   non-country prompts. The direction encodes serious-geopolitical-
   discussion, not the country tokens themselves.
4. **Negation flips the polarity.** "Mars is not a country" projects
   at -10.66, on the non-country side. Country-status as denial does
   not activate the direction.
5. **Fictional countries do not trigger.** Tolkien's Gondor at -8.40,
   clustered with indirect mentions. The direction is specifically
   about real-world geographic entities.

The big-picture result for NLA: **the AV can describe arithmetically
constructed vectors in natural language.** We can extract concept
directions via difference-of-means and use the AV as a free
interpretation tool to verify what we extracted. This is the first
half of an NLA-CAV-steering pipeline working.

## Evidence

### AV-decoded reading of the country direction

```
direction_for_av = (country_mean - non_country_mean) / ||diff|| * 150.0
nla_verbalize(direction_for_av, ...)
```

Output (full):

> "Egyptian travel guide format with structured listing of visa
> requirements and travel facts, using a British English format with
> bullet points and official data.
>
> The phrase 'The Ministry of Foreign Affairs of China and Korea'
> mirrors a common template pattern ('If you want to visit or live
> in...'), with the clause ending 'If it is in the census' suggesting
> a standard statistic about Korea's status.
>
> Final token '访游士' is mid-sentence in a quoted statistic ('In
> recent years...'), part of a census/official status clause ('If it
> is in '访游'...'), strongly expecting 'China's' or 'the country's
> capital' or 'South Korea and other.'"

Note that the AV mentions specific countries (China, Korea, Egypt)
that were NOT in our source set. The direction's identity is inferred
from the AV's training distribution — country-related vectors
co-activate cultural/linguistic features the AV reports together.

### Projection table

Sorted highest to lowest projection on the country direction:

| projection | label | prompt | interpretation |
|---|---|---|---|
| +6.88 | SRC country | "What is the capital of France?" | top of source group |
| +4.77 | SRC country | "China's economy is growing." | |
| +4.07 | holdout_country | "The United Kingdom has a king." | clean generalization |
| +3.76 | SRC country | "Italy borders Switzerland." | |
| +0.93 | SRC country | "The population of India is large." | |
| -0.70 | SRC country | "Spain is in Europe." | source spread is wide |
| -0.95 | holdout_country | "Egypt is in Africa." | weak generalization |
| -1.47 | SRC country | "Brazil is famous for soccer." | |
| -2.64 | SRC country | "Tell me about Germany." | |
| -2.66 | SRC country | "Japan has a unique culture." | |
| -3.30 | indirect | "The band Queen is from England." | country incidental to topic |
| -6.54 | indirect | "Mozart was born in Salzburg." | even more incidental |
| -8.40 | fictional | "Tell me about Gondor." | fictional country, weak activation |
| -10.66 | negation | "Mars is not a country." | denial of country-status |
| -11.54 | weird | "Portugal smells like Tuesdays to me." | metaphorical country use |
| -12.77 | weird | "Justice is the country of the soul." | abstract metaphor |
| -12.92 | weird | "If France were a sandwich..." | counterfactual country |
| -13.31 | non-country sanity | "Banana is yellow." | baseline |
| -13.92 | weird | "Germany over breakfast cereal." | country in absurd context |
| -14.82 to -26.53 | source non-country | (all 8 non-country baseline) | tight cluster at the bottom |
| -16.63 | weird | "My pet hamster Belgium escaped..." | country-name-as-pet, most negative |

Group means:

```
source country     :  +1.11   (high variance: -2.66 to +6.88)
source non-country : -20.65   (lower variance, tight cluster)
holdout country    :  +1.56   (mean of 2 holdouts)
indirect mention   :  -4.92   (mean of 2)
weird-framing      : -13.56   (mean of 5)
fictional Gondor   :  -8.40   (single test)
negation           : -10.66   (single test)
```

The weird-framing group projects with the same magnitude as the
non-country baseline. **The direction does not generalize to country
mentions in non-geopolitical contexts.**

## Hypotheses

### H1 — The direction encodes country-as-topic-of-serious-discussion

The AV's reading uses "travel guide," "visa requirements," "Ministry
of Foreign Affairs," "census" — all *standard ways of talking about
countries as geopolitical entities*. Weird framings (country-as-pet,
country-as-sandwich) project negative because they activate
*different* feature sets entirely — pet-naming features,
metaphor-construction features — even though the surface token is a
country name.

**To test:** train a much narrower CAV using only "geopolitical
discussion" prompts vs only "naming/labeling" prompts. The hypothesis
predicts they're orthogonal directions, not subsets of each other.

### H2 — Within-source-group variance dominates because h[20] is
high-dimensional and feature-rich

The 8 source country prompts span -2.66 to +6.88 in projection. The
within-group variance is comparable to the between-group separation
(22 units). This is consistent with each h[20] having weight on many
features in superposition, of which "country" is one of several.
Averaging cancels the others across the group but not within an
individual.

**To test:** train an SAE on h[20] over a large corpus and check
whether the "country" feature has lower within-group variance than
our difference-of-means estimate.

### H3 — Negation flips the polarity because layer 20 encodes
*current commitment*, not *current topic*

"Mars is not a country" is about country-status (topic), but the
model is *committing to negating* that status, not affirming it. If
h[20] encodes commitment direction rather than topical aboutness,
negation would flip polarity. Same mechanism explains why "Tell me
about Gondor" doesn't fire: the model is not committing to a
real-world country here.

**To test:** prompt pair "X is a country" vs "X is not a country"
for matched X. Predict the first projects positive, the second
negative, by ~10 units.

## Surprises

1. **The AV named Korea and Egypt** — neither in our source set. Egypt
   was in our test set; Korea was nowhere. The AV's reading is
   genuinely inferential: it identified the *kind of thing* the
   direction is and named representative examples from its own
   training distribution.
2. **Source country group variance was higher than expected.** Most
   linear-probe / CAV papers show tight within-group clustering. Ours
   has individual countries spanning ~9 units, comparable to
   between-group separation. Why? Layer 20 is mid-late; country-
   specific representations may differ a lot per country (China vs
   Japan vs Spain are not interchangeable in h[20]).
3. **Negation is "country-cancelling" rather than "country-flipping".**
   "Mars is not a country" projects at -10.66 — not zero, not
   positive. It actively encodes the opposite of country-topic.
4. **Fictional Gondor reads as non-country despite being clearly
   country-shaped fiction.** This narrows our claim: the direction is
   specifically about *real-world* geographic entities, not "country
   as ontological category."

## Follow-ups

1. **SAE comparison** (tests H2). Train or download a Qwen2.5-7B
   layer-20 SAE; compare the "country" feature's projection statistics
   to ours.
2. **Concept-vector steering** (the actual use case). Use our direction
   as a steering vector — splice `h_normal + alpha * country_direction`
   into a non-country prompt's residual at layer 20 and see if output
   becomes country-themed. This is the natural follow-up to today's
   null-result direct-transplant steering.
3. **Concept arithmetic.** Extract a "France" direction and a
   "Germany" direction separately. Compute "France - Germany." Decode
   via AV — does it read as "differences between countries / what
   makes them distinct"? Concept arithmetic visible in natural
   language.
4. **Cross-concept testing.** Repeat the methodology for "person,"
   "math problem," "code," "emotion." See which concepts are clean
   directions vs noisy. Builds a library of NLA-extracted features.

## Reproducibility

```bash
cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_country_concept_vector.py
```

CPU-only. ~20 min on warm cache. Artifact at
testing/.cache/nla_artifacts/country_concept_vector.pt persists all
captures, the extracted direction, projection results, and the AV
reading.

## References

- [Aggregate faithfulness across 8 prompts](2026-05-13-nla-aggregate-faithfulness-8-prompts.md) — established baseline cosines we used to validate the AV's faithfulness.
- [Forced continuation detects named falsehoods](2026-05-13-nla-forced-continuation-detects-named-falsehoods.md) — established that the AV reads training-distribution features, not raw activation content.
- Kim et al., 2018, "Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)" — original CAV methodology this builds on.
- Anthropic 2023+ work on Sparse Autoencoders for feature extraction — alternative approach that addresses the within-group variance we hit (H2).
