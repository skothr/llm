---
paper_key: hewitt2019-structural-probe
title: "A Structural Probe for Finding Syntax in Word Representations"
authors: Hewitt, Manning
year: 2019
venue: NAACL 2019
arxiv: null
url: https://aclanthology.org/N19-1419/
local_pdf: null
type: excerpts
note: PDF not yet downloaded; excerpts from ACL Anthology abstract and Hewitt's lab page (verified via WebSearch 2026-05-04). The structural-probe paper generalizes linear probing from binary classification to geometric / structural probing of higher-arity properties (parse trees). Foundational for the §1.2 structural-probe definition in `kb/notes/interpretability/probing.md`.
---

# Excerpts — Hewitt & Manning 2019, "A Structural Probe for Finding Syntax in Word Representations"

## Abstract — the structural probe {#abstract}

> Recent work has improved our ability to detect linguistic knowledge
> in word representations. However, current methods for detecting
> syntactic knowledge do not test whether syntax trees are represented
> in their entirety. In this work, we propose a *structural probe*,
> which evaluates whether syntax trees are embedded in a linear
> transformation of a neural network's word representation space. The
> probe identifies a linear transformation under which squared L2
> distance encodes the distance between words in the parse tree, and
> one in which squared L2 norm encodes depth in the parse tree. Using
> our probe, we show that such transformations exist for both ELMo and
> BERT but not in baselines, providing evidence that entire syntax
> trees are embedded implicitly in deep models' vector geometry.

(Verified from ACL Anthology N19-1419, NAACL 2019.)

## Operational form

The two structural probes:

1. **Distance probe.** Learn $\mathbf{B} \in \mathbb{R}^{k \times d}$
   such that for any two tokens $i, j$ in the sentence:
   $$\|\mathbf{B}(\mathbf{h}_i - \mathbf{h}_j)\|_2^2 \approx d_T(w_i, w_j)$$
   where $d_T$ is the parse-tree distance (number of edges on the
   path between $w_i$ and $w_j$).

2. **Depth probe.** Learn $\mathbf{B}$ such that
   $\|\mathbf{B}\mathbf{h}_i\|_2^2 \approx \mathrm{depth}_T(w_i)$
   (depth of token $i$ in the parse tree).

Both probes have a single learned linear transformation per
(model, layer); the rank $k$ is a hyperparameter.

The methodological contribution beyond Alain & Bengio 2017: the
probe target is a **structured** quantity (tree distance / tree
depth), not a single class label. This generalizes from
"is property $y$ in the activations" to "is the *structure* of
property $y$ in the activations".

[PDF-VERIFY] Specific findings — the layer at which BERT's distance
probe peaks, the rank at which the probe achieves its best fit,
the comparison numbers against baselines — need PDF verification
before propagating as hard claims. The abstract-level claim above
is sufficient to ground the §1.2 reference.
