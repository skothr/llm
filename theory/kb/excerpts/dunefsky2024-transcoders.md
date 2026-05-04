---
paper_key: dunefsky2024-transcoders
title: "Transcoders Find Interpretable LLM Feature Circuits"
authors: Dunefsky, Chlenski, Nanda
year: 2024
venue: NeurIPS 2024
arxiv: 2406.11944
local_pdf: theory/sources/papers/dunefsky2024_transcoders.pdf
type: excerpts
note: Verbatim quotations from the v2 arXiv PDF (Nov 2024). Transcoders: a wide ReLU MLP that learns to mimic the input/output behavior of an MLP sublayer with sparse hidden activations. Unlike SAEs (which decode an activation back to itself), transcoders give an *input-invariant* factorization of MLP computations into sparse feature circuits. The architectural substrate of Anthropic's 2025 attribution-graph framework.
---

# Excerpts — Dunefsky et al. 2024, "Transcoders Find Interpretable LLM Feature Circuits"

## Abstract — what transcoders are and what they enable {#abstract}

> A key goal in mechanistic interpretability is circuit analysis:
> finding sparse subgraphs of models corresponding to specific behaviors
> or capabilities. However, MLP sublayers make fine-grained circuit
> analysis on transformer-based language models difficult. In
> particular, interpretable features—such as those found by sparse
> autoencoders (SAEs)—are typically linear combinations of extremely
> many neurons, each with its own nonlinearity to account for. Circuit
> analysis in this setting thus either yields intractably large circuits
> or fails to disentangle local and global behavior. To address this we
> explore *transcoders*, which seek to faithfully approximate a densely
> activating MLP layer with a wider, sparsely-activating MLP layer. We
> introduce a novel method for using transcoders to perform weights-based
> circuit analysis through MLP sublayers. The resulting circuits neatly
> factorize into input-dependent and input-invariant terms.

## §1 The input-invariance argument {#sec-1-input-invariance}

> SAEs cannot tell us about the general input-output behavior of an MLP
> across all inputs.
>
> To address why input-invariance is desirable, consider the following
> example: say that one has a post-MLP SAE feature and wants to see how
> it is computed from pre-MLP SAE features. Doing e.g. patching on one
> input shows that a pre-MLP feature for Polish last names is important
> for causing the post-MLP feature to activate. But on other inputs,
> would features other than the Polish last name feature also cause the
> post-MLP feature to fire (e.g. an English last names feature)? Could
> there be other inputs where the Polish last names feature fires but
> the post-MLP feature does not? We can see that without
> input-invariance, it is difficult to make general claims about model
> behavior.

## §3.1 Transcoder architecture and loss {#sec-3-1}

> Transcoders aim to learn a "sparsified" approximation of an MLP
> sublayer: they approximate the output of an MLP sublayer as a sparse
> linear combination of feature vectors. Formally, the transcoder
> architecture can be expressed as
>
> $$\mathbf{z}_{\text{TC}}(\mathbf{x}) = \text{ReLU}(\mathbf{W}_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}) \tag{3}$$
> $$\text{TC}(\mathbf{x}) = \mathbf{W}_{\text{dec}} \mathbf{z}_{\text{TC}}(\mathbf{x}) + \mathbf{b}_{\text{dec}} \tag{4}$$
>
> where $\mathbf{x}$ is the input to the MLP sublayer […] $d_{\text{features}}$
> is the number of feature vectors in the transcoder, and $d_{\text{model}}$
> is the dimensionality of the MLP input activations. Usually,
> $d_{\text{features}}$ is far greater than $d_{\text{model}}$.
>
> Each feature in a transcoder is associated with two vectors: the $i$-th
> row of $\mathbf{W}_{\text{enc}}$ is the **encoder feature vector** of
> feature $i$, and the $i$-th column of $\mathbf{W}_{\text{dec}}$ is the
> **decoder feature vector** of feature $i$. The $i$-th component of
> $\mathbf{z}_{\text{TC}}(\mathbf{x})$ is called the **activation** of
> feature $i$.

Loss:

> $$\mathcal{L}_{\text{TC}}(\mathbf{x}) = \underbrace{\|\text{MLP}(\mathbf{x}) - \text{TC}(\mathbf{x})\|_2^2}_{\text{faithfulness loss}} + \underbrace{\lambda_1 \|\mathbf{z}_{\text{TC}}(\mathbf{x})\|_1}_{\text{sparsity penalty}} \tag{5}$$

## §1 The transcoder vs. SAE distinction in one picture {#sec-1-comparison}

> [Figure 1 caption] A comparison between SAEs, MLP transcoders, and
> MLP sublayers for a transformer-based language model. SAEs learn to
> reconstruct model activations, whereas transcoders imitate sublayers'
> input-output behavior.

This is the architectural distinction: an SAE on the MLP input gives
you a sparse decomposition of the activation $x$ (the residual stream
right before the MLP); an SAE on the MLP output gives you a sparse
decomposition of $\text{MLP}(x)$. A transcoder gives you a sparse
*function approximator* for $\text{MLP}(\cdot)$ itself — input-invariant.

## §1 Headline empirical result {#sec-1-headline}

> Our contributions. Our main contributions are (1) to introduce a
> method for circuit analysis using transcoders, (2) to confirm that
> transcoders are a faithful and interpretable approximation to MLP
> sublayers, and (3) to demonstrate the utility of our circuit analysis
> method on detailed case studies.
> […] Because SAEs are the standard method for finding sparse
> decompositions of model activations, we compare transcoders to SAEs
> on models up to 1.4 billion parameters and verify that transcoders
> are on par with SAEs or better with respect to these properties.
