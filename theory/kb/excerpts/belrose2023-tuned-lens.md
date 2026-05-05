---
paper_key: belrose2023-tuned-lens
title: "Eliciting Latent Predictions from Transformers with the Tuned Lens"
authors: Belrose, Furman, Smith, Halawi, Ostrovsky, McKinney, Biderman, Steinhardt
year: 2023
venue: arXiv (also COLM 2024 cite for downstream applications)
arxiv: 2303.08112
local_pdf: sources/papers/belrose2023_tuned-lens.pdf
type: excerpts
note: Excerpts verified from local PDF (`sources/papers/belrose2023_tuned-lens.pdf`, v6 2025-11-11). Equations transcribed from §2 and §3; §4 CBE math from page 7 (Eq. 10-12). Anchors cover §1 framing, §2 logit-lens definition + failure modes, §3 tuned-lens definition + loss, §4 causal basis extraction, §5 applications.
---

# Excerpts — Belrose et al. 2023, "Eliciting Latent Predictions from Transformers with the Tuned Lens"

## §1 Iterative-inference framing {#sec-1-iterative}

> We analyze transformers from the perspective of iterative inference,
> seeking to understand how model predictions are refined layer by
> layer. To do so, we train an affine probe for each block in a
> frozen pretrained model, making it possible to decode every hidden
> state into a distribution over the vocabulary.

> We decode these latent predictions through early exiting, converting
> the hidden state at each intermediate layer into a distribution over
> the vocabulary. This yields a sequence of distributions we call the
> *prediction trajectory*, which exhibits a strong tendency to
> converge smoothly to the final output distribution, with each
> successive layer achieving lower perplexity.

Figure 1 (page 1) shows side-by-side logit-lens vs. tuned-lens
predictions on GPT-Neo-2.7B prompted with the Vaswani et al. 2017
abstract. The logit lens fails to elicit interpretable predictions
before layer 21; the tuned lens succeeds throughout.

## §2 The logit lens — definition and failures {#sec-2-logit-lens}

The logit lens is defined by setting all post-layer-$\ell$ residual
updates to zero in the residual decomposition (Eq. 2):

> $$\mathcal{M}_{>\ell}(\mathbf{h}_\ell) = \mathrm{LayerNorm}\!\left[\mathbf{h}_\ell + \underbrace{\sum_{\ell'=\ell}^{L} F_{\ell'}(\mathbf{h}_{\ell'})}_{\text{residual update}}\right] W_U \quad (2)$$

> $$\mathrm{LogitLens}(\mathbf{h}_\ell) = \mathrm{LayerNorm}[\mathbf{h}_\ell] W_U \quad (3)$$

(both equations transcribed from page 2, with the residual-update
brace in the original.)

### §2 residual decomposition {#sec-2-decomposition}

> The transformer layer at index $\ell$ updates the representation as
> follows: $\mathbf{h}_{\ell+1} = \mathbf{h}_\ell + F_\ell(\mathbf{h}_\ell)$
> [Eq. 1]. […] Applying Equation 1 recursively, the output logits
> $\mathcal{M}_{>\ell}$ can be written as a function of an arbitrary
> hidden state $\mathbf{h}_\ell$ at layer $\ell$ [Eq. 2 above].

### §2 Unreliability and bias of the logit lens {#sec-2-failures}

> Beyond GPT-Neo, the logit lens struggles to elicit predictions from
> several other models released since its introduction, such as BLOOM
> (Scao et al., 2022) and OPT 125M (Zhang et al., 2022) (Figure 14).

> Moreover, the type of information extracted by the logit lens varies
> both from model to model and from layer to layer, making it
> difficult to interpret. For example, we find that for BLOOM and
> OPT 125M, the top 1 prediction of the logit lens is often the input
> token, rather than any plausible continuation token, in more than
> half the layers (Figure 18).

> **Bias.** Even when the logit lens is useful, we find that it is a
> *biased* estimator of the model's final output: it systematically
> puts more probability mass on certain vocabulary items than the
> final layer does. […] This is concerning because it suggests we
> can't interpret the logit lens prediction trajectory as a belief
> updating in response to new evidence. The beliefs of a rational
> agent should not update in an easily predictable direction over
> time, since predictable updates can be exploited via Dutch books.

> In Figure 3 we evaluate the bias for each layer of GPT-Neo-2.7B. We
> find the bias of the logit lens can be quite large: around 4 to 5
> bits for most layers. As a point of comparison, the bias of Pythia
> 160M's final layer distribution relative to that of its larger
> cousin, Pythia 12B, is just 0.0068 bits.

## §3 The tuned lens — definition and loss {#sec-3-tuned-lens}

> One problem with the logit lens is that, if transformer layers learn
> to output residuals that are far from zero *on average*, the input
> to LogitLens may be out-of-distribution and yield nonsensical
> results. […] Our first change to the method is to replace the
> summed residuals with a learnable constant value $\mathbf{b}_\ell$
> instead of zero:
> $$\mathrm{LogitLens}^{\mathrm{debiased}}_\ell(\mathbf{h}_\ell) = \mathrm{LogitLens}(\mathbf{h}_\ell + \mathbf{b}_\ell) \quad (7)$$

> One simple, general way to correct for drifting covariance is to
> introduce a learnable change of basis matrix $A_\ell$, which learns
> to map from the output space of layer $\ell$ to the input space of
> the final layer. We have now arrived at the *tuned lens* formula,
> featuring a learned affine transformation for each layer:
> $$\mathrm{TunedLens}_\ell(\mathbf{h}_\ell) = \mathrm{LogitLens}(A_\ell \mathbf{h}_\ell + \mathbf{b}_\ell) \quad (8)$$

> We refer to $(A_\ell, \mathbf{b}_\ell)$ as the *translator* for
> layer $\ell$.

(Eq. 8 transcribed from page 4, top right.)

### §3 Loss — KL distillation against the final layer {#sec-3-loss}

> **Loss function.** We train the translators to minimize KL between
> the tuned lens logits and the final layer logits:
> $$\arg\min \mathbb{E}_x \left[D_{\mathrm{KL}}(f_{>\ell}(\mathbf{h}_\ell) \| \mathrm{TunedLens}_\ell(\mathbf{h}_\ell))\right] \quad (9)$$
> where $f_{>\ell}$ refers to the rest of the transformer after layer
> $\ell$. This can be viewed as a distillation loss, using the final
> layer prediction distribution as a soft label (Sanh et al., 2019).
> It ensures that the probes are not incentivized to learn extra
> information over and above what the model has learned, which can
> become a problem when training probes with ground truth labels
> (Hewitt and Liang, 2019).

> **Implementation details.** […] We initialized all translators to
> the identity transform, and use a weight decay of $10^{-3}$.

> **Benefits over traditional probing.** Unlike Alain and Bengio
> (2016), who train early exiting probes for image classifiers, we
> do not learn a new unembedding for each layer. This is important,
> since it allows us to shrink the size of each learned matrix from
> $|\mathcal{V}| \times d$ to $d \times d$, where $|\mathcal{V}|$
> ranges from 50K (GPT-2, Pythia) to over 250K (BLOOM). We observe
> empirically that training a new unembedding matrix requires
> considerably more training steps and a larger batch size than
> training a translator, and often converges to a worse perplexity.

## §4 Causal Basis Extraction {#sec-4-cbe}

(Section 4.1, page 7.)

> More specifically, let $f$ be a function (such as the tuned lens)
> that maps latent vectors $\mathbf{h} \in \mathbb{R}^d$ to logits
> $\mathbf{y}$. Let $r(\mathbf{h}, \mathbf{v})$ be an erasure
> function which removes information along the span of $\mathbf{v}$
> from $\mathbf{x}$. In this work we use $r(\mathbf{h}, \mathbf{v})$
> = mean ablation, which sets $\langle \mathbf{h}, \mathbf{v}\rangle$
> to the mean value of $\langle \mathbf{h}, \mathbf{v}\rangle$ in the
> dataset (see Appendix D.1). We define the *influence* $\sigma$ of a
> unit vector $\mathbf{v}$ to be the expected KL divergence between
> the outputs of $f$ before and after erasing $\mathbf{v}$ from
> $\mathbf{h}$:
> $$\sigma(\mathbf{v}; f) = \mathbb{E}_\mathbf{h}\left[D_{\mathrm{KL}}(f(\mathbf{h}) \| f(r(\mathbf{h}, \mathbf{v})))\right] \quad (10)$$

> We seek to find an orthonormal basis $B = (\mathbf{v}_1, \ldots, \mathbf{v}_k)$
> containing principal features of $f$, ordered by a sequence of
> influences $\Sigma = (\sigma_1, \ldots, \sigma_k)$ for some
> $k \le d$. In each iteration we search for a feature $\mathbf{v}_i$
> with maximum influence that is orthogonal to all previous features
> $\mathbf{v}_j$:
> $$\mathbf{v}_i = \arg\max_{\|\mathbf{v}\|_2 = 1,\;\langle\mathbf{v},\mathbf{v}_j\rangle = 0\;\forall j < i} \sigma(\mathbf{v}; f) \quad (11)$$

(Equations 10-11 transcribed from page 7.)

### §4 CBE causal validation {#sec-4-cbe-validation}

> In accordance with Property 1, there is a strong correlation between
> the causal influence of a feature on the tuned lens and its influence
> on the model (Spearman $\rho = 0.89$). Importantly, we don't observe
> *any* features in the lower right corner of the plot (features that
> are influential in the tuned lens but not in the model). The model
> is somewhat more "causally sensitive" than the tuned lens: even the
> least influential features never have an influence under
> $2 \times 10^{-3}$ bits.

(From page 7, Figure 8 caption + immediately preceding paragraph,
discussing CBE features ablated at layer 18 of Pythia 410M.)

## §5 Applications {#sec-5}

### Eliciting secret knowledge

> Cywiński et al. (2025) apply the logit lens to uncover knowledge
> that an AI possesses but does not verbalize. Specifically, they
> fine-tune a model to generate hints and respond to user queries
> about a specific "taboo" word, while never verbalizing it directly.
> They then prompt an "auditor" LLM to guess what the taboo word is,
> allowing it to ask the taboo model questions in a black-box fashion.
> They find that the auditor is much more accurate at guessing the
> taboo word when it is provided with the top 20 highest probability
> latent predictions from the taboo model according to the logit lens.

> We hypothesized that the tuned lens might perform better than the
> logit lens for this task. The results were mixed: the tuned lens
> outperforms at some layers and underperforms at others.

### §5.3 Detecting prompt injections {#sec-5-3}

> We hypothesize that the prediction trajectory of the tuned lens on
> anomalous inputs should be different from the trajectories on
> normal inputs, and that this could be used to detect anomalous
> inputs. […] To test this, we focus on *prompt injection attacks*,
> a recently discovered vulnerability in large language models where
> untrusted inputs from a malicious user cause the model to behave in
> unexpected or dangerous ways. […] We then flatten these
> trajectories into feature vectors and feed them into two standard
> outlier detection algorithms — isolation forest (iForest) and local
> outlier factor (LOF).

(Belrose et al. report near-perfect detection accuracy across the
nine multiple-choice tasks tested.)
