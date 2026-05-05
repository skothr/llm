---
paper_key: belinkov2022-probing-survey
title: "Probing Classifiers: Promises, Shortcomings, and Advances"
authors: Belinkov
year: 2022
venue: Computational Linguistics 48(1), pp. 207-219
arxiv: 2102.12452
url: https://aclanthology.org/2022.cl-1.7/
local_pdf: null
type: excerpts
note: PDF not yet downloaded; excerpts from ACL Anthology / arXiv 2102.12452 abstract (verified via WebSearch 2026-05-04). The methodology review that consolidates the correlational critique of probing and surveys causal alternatives. Used in `kb/notes/interpretability/probing.md` §3 as the canonical reference for the probe-as-correlation-not-causation gap.
---

# Excerpts — Belinkov 2022, "Probing Classifiers: Promises, Shortcomings, and Advances"

## Abstract {#abstract}

> Probing classifiers have emerged as one of the prominent
> methodologies for interpreting and analyzing deep neural network
> models of natural language processing. The basic idea is simple — a
> classifier is trained to predict some linguistic property from a
> model's representations — and has been used to examine a wide
> variety of models and properties. However, recent studies have
> demonstrated various methodological limitations of this approach.
> This article critically reviews the probing classifiers framework,
> highlighting their promises, shortcomings, and advances.

(Verified via ACL Anthology 2022.cl-1.7 abstract and arXiv 2102.12452.)

## The shortcomings list (operational)

Belinkov's review organizes the methodological critique into:

1. **Probe expressivity confounder.** A sufficiently powerful probe
   can learn a property from low-information features that the
   model itself does not use. The control-task baseline (Hewitt &
   Liang 2019) is the standard mitigation: train the same probe
   architecture on randomly-permuted labels; if it still does well,
   the probe is too expressive to attribute its accuracy to the
   model's representation.

2. **Spurious correlation confounder.** A probe can read a feature
   that *correlates with* the target property in the probing
   dataset, not the property itself. Generalization tests across
   distributions (e.g., probe trained on `cities`, evaluated on
   `neg_cities`) detect this.

3. **Vestigial information confounder.** Information about the
   property may be in the activations *without being used* by the
   downstream model. Probing is correlational; the model's
   *use* of the information requires causal intervention.

4. **The advances section** surveys causal probing methods —
   Amnesic Probing (Elazar et al. 2021), causal-mediation analysis
   (Vig et al. 2020), interchange interventions (Geiger et al.
   2020) — that pair probes with activation interventions to test
   whether the probe direction is causally implicated.

[PDF-VERIFY] The exact taxonomy of "advances" sections and the
specific recommendations Belinkov makes for probing-protocol design
need PDF verification before propagating as hard claims. The
abstract-level critique above is sufficient to ground the §3
correlation-vs-causation gap reference in
`kb/notes/interpretability/probing.md`.
