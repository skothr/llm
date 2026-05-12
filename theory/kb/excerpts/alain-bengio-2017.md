---
paper_key: alain-bengio-2017
title: "Understanding intermediate layers using linear classifier probes"
authors: Alain, Bengio
year: 2016
venue: ICLR 2017 Workshop (arXiv 1610.01644)
arxiv: 1610.01644
local_pdf: theory/sources/papers/alain-bengio-2017.pdf
type: excerpts
note: PDF download blocked (curl/wget restricted by sandbox in this Phase 2.5 deepening pass); full abstract retrieved verbatim via WebFetch on arXiv 1610.01644 abstract page (2026-05-05). The foundational paper for the linear-classifier-probe methodology, applied to vision models (Inception v3, ResNet-50). Key findings used in the probing note: (i) the formal probe definition, (ii) the monotonic-linear-separability finding.
---

# Excerpts — Alain & Bengio 2017, "Understanding intermediate layers using linear classifier probes"

## #abstract — full verbatim

> Neural network models have a reputation for being black boxes. We
> propose to monitor the features at every layer of a model and measure
> how suitable they are for classification. We use linear classifiers,
> which we refer to as "probes", trained entirely independently of the
> model itself. This helps us better understand the roles and dynamics
> of the intermediate layers. We demonstrate how this can be used to
> develop a better intuition about models and to diagnose potential
> problems. We apply this technique to the popular models Inception v3
> and Resnet-50. Among other things, we observe experimentally that the
> linear separability of features increase monotonically along the depth
> of the model.

(Verified verbatim from arXiv 1610.01644 abstract via WebFetch
2026-05-05.)

## The probe definition (operational)

A linear probe in Alain & Bengio's setup is a logistic regression
classifier trained on the activations at a chosen layer, with the
underlying network's parameters held fixed. The probe has access
*only* to the layer's output; it cannot see earlier layers or the
input.

The headline operationalization: probe accuracy at layer $\ell$
indicates how *linearly decodable* the class label is from
$\mathbf{h}^{(\ell)}$. The monotonic-increase finding (probe accuracy
is non-decreasing with $\ell$ for class-label probes on
ImageNet-trained CNNs) is the canonical "models hierarchically
abstract" result that subsequent NLP probing work
(Tenney et al. 2019, Hewitt & Manning 2019) inherits and refines.

[PDF-VERIFY] The exact form of the probe (sigmoid vs. softmax),
regularization choice, and the layer-grain at which the
monotonic-increase finding holds need PDF verification before
propagating into hard claims. The abstract-level claim above is
sufficient to ground the §1 / §2 references in
`kb/notes/interpretability/probing.md`.
