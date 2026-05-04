---
paper_key: leask2025-sae-not-canonical
title: "Sparse Autoencoders Do Not Find Canonical Units of Analysis"
authors: Leask, Bussmann, Pearce, Bloom, Tigges, Al Moubayed, Sharkey, Nanda
year: 2025
venue: ICLR 2025
arxiv: 2502.04878
local_pdf: theory/sources/papers/sae-not-canonical-2025.pdf
type: excerpts
note: Verbatim quotations from the v1 arXiv PDF (Feb 2025). The "negative result" paper for the SAE/monosemanticity program: introduces SAE stitching and meta-SAEs to show that (a) larger SAEs find features that smaller SAEs do not (incompleteness), and (b) latents in a large SAE often decompose into combinations of latents from other SAEs (non-atomicity). Direct challenge to the "canonical features" framing of Bricken 2023 / Templeton 2024.
---

# Excerpts — Leask et al. 2025, "Sparse Autoencoders Do Not Find Canonical Units of Analysis"

## Abstract — the headline negative result {#abstract}

> A common goal of mechanistic interpretability is to decompose the
> activations of neural networks into features: interpretable properties
> of the input computed by the model. Sparse autoencoders (SAEs) are a
> popular method for finding these features in LLMs, and it has been
> postulated that they can be used to find a *canonical* set of units:
> a unique and complete list of atomic features. We cast doubt on this
> belief using two novel techniques: SAE stitching to show they are
> incomplete, and meta-SAEs to show they are not atomic. SAE stitching
> involves inserting or swapping latents from a larger SAE into a
> smaller one. Latents from the larger SAE can be divided into two
> categories: *novel latents*, which improve performance when added to
> the smaller SAE, indicating they capture novel information, and
> *reconstruction latents*, which can replace corresponding latents in
> the smaller SAE that have similar behavior. The existence of novel
> features indicates incompleteness of smaller SAEs. Using meta-SAEs —
> SAEs trained on the decoder matrix of another SAE — we find that
> latents in SAEs often decompose into combinations of latents from a
> smaller SAE, showing that larger SAE latents are not atomic.

## §1 What "canonical" means here {#sec-1-canonical}

> Mechanistic interpretability aims to reverse-engineer neural networks
> into human-interpretable algorithms. […] A key challenge of
> mechanistic interpretability is identifying the correct *units of
> analysis* — fundamental components that can be individually
> understood and collectively explain the network's function. Ideally,
> these units would be *unique*, with no variations (Bricken et al.,
> 2023); *complete*, encompassing all necessary features (Elhage et
> al., 2022); and *atomic* or *irreducible*, indivisible into smaller
> components (Engels et al., 2024). We refer to a set of units with
> all of these properties as **canonical**.

## §1 The two techniques {#sec-1-techniques}

> In summary, our contributions are:
>
> 1. **SAE stitching**, as a method for comparing latents across
>    different sizes of SAE. Latents in a larger SAE are either novel
>    latents, missing in smaller SAEs, or reconstruction latents,
>    similar to some latents in smaller SAEs.
> 2. **Meta-SAEs**, as an approach for decomposing the decoder
>    directions of SAEs into interpretable, monosemantic meta-latents.

## §1 The Einstein decomposition example {#sec-1-einstein}

> We find that meta-latents are similar to latents in smaller SAEs,
> demonstrating that latents from larger SAEs can be interpreted as the
> composition of latents from smaller SAEs. […] [Figure 1] Decomposition
> of an SAE latent representing "Einstein" into a set of interpretable
> meta-latents. […] revealing how a complex concept can be represented
> as a sparse combination of meta-latents.

The Einstein latent decomposes into meta-latents for "scientist",
"Germany", "famous person", "Starts With E-" — i.e. the SAE's "Einstein"
direction is *not* atomic; it's a sparse combination of more elementary
features.

## §1 The "feature splitting" framing it overturns {#sec-1-feature-splitting}

> One challenge to the theory that SAEs identify a canonical set of
> units is the phenomenon of *feature splitting*, where latents from
> smaller SAEs "split" into multiple, more fine-grained latents in
> larger SAEs (Bricken et al., 2023). For example, [Bricken et al.,
> 2023] find a base64 feature splits into three features in a larger
> SAE: activating on letters, digits, and encoded ASCII in base64 text.
> […] Currently, the effect of dictionary size on the features has not
> been systematically studied, in part because we lack good methods to
> compare latents found in SAEs of different sizes.

## §1 Why SAEs may still be useful {#sec-1-still-useful}

> Our empirical results suggest that simply training larger SAEs is
> unlikely to result in a canonical set of units for all mechanistic
> interpretability tasks, and that the choice of dictionary size is
> subjective. We suggest taking a pragmatic approach to applying SAEs
> to mechanistic interpretability tasks, trying SAEs of several widths
> to see which is best suited. We are uncertain whether canonical units
> of analysis exist, but our results suggest that alternative approaches
> should be explored.

[CONTRADICTION] This directly contradicts the "true features" framing
in Bricken et al. 2023 ("Towards Monosemanticity") and the implicit
assumption in Templeton et al. 2024 ("Scaling Monosemanticity") that
larger SAEs converge toward a canonical decomposition of model
activations. As of 2026, the debate is unresolved — Anthropic's
2025 circuit-tracing work uses SAEs/transcoders pragmatically (a
useful basis, not a unique one), which is consistent with this paper's
recommendations.
