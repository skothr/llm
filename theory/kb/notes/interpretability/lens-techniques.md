---
topic: interpretability/lens-techniques
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - belrose2023-tuned-lens
  - nostalgebraist2020-logit-lens   # Tier B / C: LessWrong post; discovery-only
  - chuang2023-dola
secondary_sources:
  - marks-tegmark-2023-truth        # uses logit-lens-style projection in §3
  - meng2022-rome                   # logit-lens-of-causal-trace inspiration
related_topics:
  - interpretability/probing
  - interpretability/mechanistic-interpretability
  - interpretability/sparse-autoencoders
  - interpretability/activation-patching
  - architecture/transformer-overview
---

# Lens techniques

A **lens** is a function that takes a transformer's intermediate
hidden state at layer $\ell$ and returns a distribution over the
vocabulary, *as if* the model had early-exited at $\ell$. Lenses are
the cheapest interpretability readout — no training data, no
classifier, no causal intervention; just a matrix multiplication
through the model's own unembedding. The two canonical lenses are the
**logit lens** (nostalgebraist 2020, LessWrong) and the **tuned lens**
(Belrose et al. 2023). The lens framework is also the substrate for at
least one inference-time application — **DoLa** (Chuang et al. 2023) —
which contrasts logits across layers to suppress hallucination.

This note keeps strictly to the canonical math, the failure modes
documented in primary sources, and the layer-readout family of
methods. Probing (which trains a *new* classifier on activations) is
in `kb/notes/interpretability/probing.md`; SAEs (which decompose
activations into a sparse basis) are in
`kb/notes/interpretability/sparse-autoencoders.md`.

## 1. Formal definition

### 1.1 Setup — the iterative-inference view

Belrose et al. 2023 frame transformers as **iterative inference**:
each layer makes a small additive update to the residual stream, and
the *prediction trajectory* across layers tends to converge smoothly
to the final-layer output `[belrose2023-tuned-lens §1;
kb/excerpts/belrose2023-tuned-lens#sec-1-iterative]`.

Concretely, for a pre-LayerNorm transformer $\mathcal{M}$, the residual
update at layer $\ell$ is

$$\mathbf{h}_{\ell+1} = \mathbf{h}_\ell + F_\ell(\mathbf{h}_\ell) \tag{1}$$

where $F_\ell$ is the residual output of layer $\ell$ (attention +
MLP). Splitting $\mathcal{M}$ into "first-$\ell$ layers" and
"after-$\ell$ layers", the function from a hidden state $\mathbf{h}_\ell$
at layer $\ell$ to the final-layer logits is

$$\mathcal{M}_{>\ell}(\mathbf{h}_\ell) = \mathrm{LayerNorm}\!\left[\mathbf{h}_\ell + \underbrace{\sum_{\ell'=\ell}^{L} F_{\ell'}(\mathbf{h}_{\ell'})}_{\text{residual update}}\right] W_U \tag{2}$$

`[belrose2023-tuned-lens §2 Eq.2; kb/excerpts/belrose2023-tuned-lens#sec-2-decomposition]`,
where $W_U \in \mathbb{R}^{d \times |\mathcal{V}|}$ is the unembedding
matrix (often weight-tied with the input embedding).

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $\mathbf{h}_\ell \in \mathbb{R}^d$ | residual-stream activation at the entry of layer $\ell$, single token |
| $F_\ell$ | residual output of layer $\ell$ (block contribution) |
| $\mathcal{M}_{\le \ell}, \mathcal{M}_{>\ell}$ | first-$\ell$ layers / after-$\ell$ layers as functions |
| $L$ | total number of transformer layers |
| $W_U \in \mathbb{R}^{d \times \|\mathcal{V}\|}$ | unembedding matrix |
| $\mathcal{V}$ | vocabulary |
| $\mathrm{LayerNorm}$ | the model's own final pre-unembed LayerNorm (or RMSNorm) |
| $A_\ell \in \mathbb{R}^{d \times d}, \mathbf{b}_\ell \in \mathbb{R}^d$ | tuned-lens "translator" parameters at layer $\ell$ |

### 1.2 Logit lens (nostalgebraist 2020)

The **logit lens** drops the residual update from layer $\ell$ onward
in Eq. (2) — i.e., it sets $\sum_{\ell'=\ell}^L F_{\ell'} = 0$:

$$\boxed{\;\mathrm{LogitLens}(\mathbf{h}_\ell) := \mathrm{LayerNorm}[\mathbf{h}_\ell]\, W_U\;} \tag{3}$$

`[belrose2023-tuned-lens §2 Eq.3; kb/excerpts/belrose2023-tuned-lens#sec-2-logit-lens]`.
The output is a vector of logits over $\mathcal{V}$; applying softmax
gives the layer-$\ell$ "as-if-final" distribution. The technique was
introduced by nostalgebraist on LessWrong in 2020
`[nostalgebraist2020-logit-lens FORUM-SIGNAL]` and reused widely
(Halawi et al. 2023, Geva et al. 2022, Dar et al. 2022, Millidge &
Black 2022) before any peer-reviewed treatment
`[belrose2023-tuned-lens §2; kb/excerpts/belrose2023-tuned-lens#sec-2-logit-lens]`.

### 1.3 Tuned lens (Belrose et al. 2023)

The **tuned lens** inserts a learned affine **translator**
$(A_\ell, \mathbf{b}_\ell)$ at layer $\ell$ that maps $\mathbf{h}_\ell$
into the input space the unembedding expects:

$$\boxed{\;\mathrm{TunedLens}_\ell(\mathbf{h}_\ell) := \mathrm{LogitLens}(A_\ell \mathbf{h}_\ell + \mathbf{b}_\ell)\;} \tag{4}$$

`[belrose2023-tuned-lens §3 Eq.8; kb/excerpts/belrose2023-tuned-lens#sec-3-tuned-lens]`.
$A_\ell$ is initialized to the identity, so a freshly initialized
tuned lens is identical to the logit lens; training pulls the
translator away from the identity to reduce KL to the final-layer
output. Crucially, the unembedding $W_U$ is **frozen** — there is
exactly one $W_U$ per model, shared across all $L$ tuned lenses
`[belrose2023-tuned-lens §3 implementation;
kb/excerpts/belrose2023-tuned-lens#sec-3-tuned-lens]`.

The training objective is a per-layer distillation loss against the
final-layer prediction:

$$\mathcal{L}(\ell) = \mathbb{E}_{\mathbf{x}}\!\left[D_{\mathrm{KL}}\!\left(\mathcal{M}_{>\ell}(\mathbf{h}_\ell)\,\big\|\,\mathrm{TunedLens}_\ell(\mathbf{h}_\ell)\right)\right] \tag{5}$$

`[belrose2023-tuned-lens §3 Eq.9; kb/excerpts/belrose2023-tuned-lens#sec-3-loss]`.
Using the model's own $\mathcal{M}_{>\ell}$ output as the soft label
(rather than ground-truth tokens) is what the authors call a
**distillation** loss; it ensures the translator learns nothing the
model itself does not already know
`[belrose2023-tuned-lens §3 distillation;
kb/excerpts/belrose2023-tuned-lens#sec-3-loss]`.

## 2. Mechanism — what the lens computes

### 2.1 Step-by-step (logit lens)

For a single token at layer $\ell$ with hidden state
$\mathbf{h}_\ell \in \mathbb{R}^d$:

1. **Apply the model's own final-pre-unembed normalization** to
   $\mathbf{h}_\ell$. Most modern LLMs use RMSNorm or pre-LayerNorm;
   the lens uses *the same* normalization the model uses just before
   $W_U$. Skipping this step gives garbage, because $\mathbf{h}_\ell$
   has the wrong scale `[belrose2023-tuned-lens §2 footnote-2;
   kb/excerpts/belrose2023-tuned-lens#sec-2-logit-lens]`.
2. **Multiply by $W_U$** to get raw logits $\in \mathbb{R}^{|\mathcal{V}|}$.
3. **Softmax** for a probability distribution over $\mathcal{V}$;
   $\arg\max$ for the layer-$\ell$ "top-1 token".

Total cost per (layer, token): one LayerNorm + one matmul of shape
$d \times |\mathcal{V}|$. The output is the token the model would
predict if it were forced to commit *now*, ignoring all later
processing.

### 2.2 Step-by-step (tuned lens)

Same as logit lens, but with an extra step 1.5: apply the translator
$A_\ell \mathbf{h}_\ell + \mathbf{b}_\ell$ before the LayerNorm + $W_U$
chain. Cost: one extra $d \times d$ matmul per layer.

Storage: tuned-lens parameters total $L \cdot (d^2 + d)$ — a fraction
of the model's own size (a $d \times d$ matrix per layer is roughly
the size of one attention block's $W^Q$). For Pythia 12B with
$d = 5120$ and $L = 36$, this is ~944M extra parameters
`[belrose2023-tuned-lens §3.1 sizing;
kb/excerpts/belrose2023-tuned-lens#sec-3-loss]`. Belrose et al. note
that this is **dramatically smaller** than training a per-layer
unembedding $|\mathcal{V}| \times d$, which is what Alain & Bengio's
1610.01644 probing recipe would imply
`[belrose2023-tuned-lens §3.2 size-comparison;
kb/excerpts/belrose2023-tuned-lens#sec-3-tuned-lens]`.

### 2.3 The prediction trajectory

Belrose et al. coin the term **prediction trajectory** for the
sequence $\{\mathrm{Lens}_\ell(\mathbf{h}_\ell)\}_{\ell=0}^L$ across
layers `[belrose2023-tuned-lens §1;
kb/excerpts/belrose2023-tuned-lens#sec-1-iterative]`. Empirically the
trajectory:

- **converges smoothly** to the final-layer prediction with depth;
- **for clean, in-distribution prompts**, agrees with the final
  prediction many layers before the end;
- **for prompt-injected or anomalous inputs**, deviates from the
  smooth trajectory in detectable ways (the basis for a prompt-
  injection detector with near-perfect accuracy
  `[belrose2023-tuned-lens §5.3;
  kb/excerpts/belrose2023-tuned-lens#sec-5-3]`).

The trajectory framing is the diagnostic value of lens techniques:
a lens reading at one layer is a snapshot, but the *evolution* across
layers is a fingerprint of the model's iterative computation.

## 3. Why the logit lens fails (and the tuned lens fixes it)

Belrose et al. document three failure modes of the raw logit lens
`[belrose2023-tuned-lens §2 unreliability;
kb/excerpts/belrose2023-tuned-lens#sec-2-failures]`:

### 3.1 Bias

The logit lens systematically puts more probability mass on certain
vocabulary items than the model's final output does. Measured as the
KL divergence between the logit-lens marginal and the final-layer
marginal, the bias on Pythia 12B is **4–5 bits per byte for most
layers**, vs. 0.0068 bits between Pythia 160M's and Pythia 12B's
final outputs `[belrose2023-tuned-lens §2 bias;
kb/excerpts/belrose2023-tuned-lens#sec-2-failures]`. The implication:
the prediction trajectory under the logit lens cannot be interpreted
as Bayesian belief updating, because a rational agent should not have
predictable updates over time.

### 3.2 Representational drift

Hidden states at different layers may live in slightly drifted
covariance subspaces. The covariance matrix at the final layer often
shifts sharply relative to earlier layers, suggesting the logit lens
could "misinterpret" earlier representations
`[belrose2023-tuned-lens §3 drift;
kb/excerpts/belrose2023-tuned-lens#sec-3-tuned-lens]`. The tuned-lens
translator $A_\ell$ explicitly absorbs this change-of-basis.

### 3.3 Cross-model unreliability

The logit lens works "reasonably well" on GPT-2 (Radford 2019) but
fails to elicit interpretable predictions before later layers on
BLOOM (Scao et al. 2022) and GPT-Neo (Black et al. 2021)
`[belrose2023-tuned-lens §2 unreliability;
kb/excerpts/belrose2023-tuned-lens#sec-2-failures]`. Figure 1 of
Belrose et al. shows the logit lens producing nonsense ("@G:")
predictions in the bottom 21 layers of GPT-Neo-2.7B, while the tuned
lens recovers coherent intermediate predictions throughout
`[belrose2023-tuned-lens §1 figure-1;
kb/excerpts/belrose2023-tuned-lens#sec-1-iterative]`.

## 4. Causal Basis Extraction (CBE) — making lenses causal

The lens is correlational by default: it tells you what *would*
happen if the model committed at layer $\ell$, not what state at
layer $\ell$ the model *uses*. Belrose et al. close part of this gap
with **Causal Basis Extraction (CBE)**
`[belrose2023-tuned-lens §4.1;
kb/excerpts/belrose2023-tuned-lens#sec-4-cbe]`.

CBE iteratively finds an orthonormal basis $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$
in residual-stream space, ordered by influence. Each $\mathbf{v}_i$ is
defined as

$$\mathbf{v}_i = \arg\max_{\|\mathbf{v}\|_2 = 1,\; \langle \mathbf{v}, \mathbf{v}_j\rangle = 0\;\forall j < i} \sigma(\mathbf{v}; f) \tag{6}$$

where $\sigma(\mathbf{v}; f) = \mathbb{E}_\mathbf{h}\!\left[D_{\mathrm{KL}}(f(\mathbf{h}) \| f(r(\mathbf{h}, \mathbf{v})))\right]$
is the expected KL divergence between the function $f$'s output
before and after **mean-ablating** the direction $\mathbf{v}$ from
$\mathbf{h}$ `[belrose2023-tuned-lens §4.1 Eq.10-11;
kb/excerpts/belrose2023-tuned-lens#sec-4-cbe]`. Setting $f =
\mathrm{TunedLens}_\ell$ identifies the directions most causally
influential on the lens output.

The validation experiment: ablate each $\mathbf{v}_i$ in the *model's
own* hidden state and measure the effect on $\mathcal{M}_{>\ell}$. On
Pythia 410M layer 18, the Spearman correlation between
"influence-on-lens" and "influence-on-model" is $\rho = 0.89$
`[belrose2023-tuned-lens §4.1 Figure 8;
kb/excerpts/belrose2023-tuned-lens#sec-4-cbe]`. This is the **causal
fidelity** of the tuned lens: the directions it considers important
are the same directions the rest of the model considers important.

## 5. Variants and lineage

| Variant | Year | Translator? | Loss | Key property |
|---|---|---|---|---|
| **Logit lens** (nostalgebraist) | 2020 | none (identity) | n/a (no training) | works on GPT-2; fails on BLOOM/GPT-Neo |
| **Logit lens$^\text{ext}$** (Black et al.) | 2021 | retain final transformer layer | n/a | partial recovery on GPT-Neo `[belrose2023-tuned-lens §2 Eq.4]` |
| **Tuned lens** (Belrose et al.) | 2023 | per-layer affine $(A_\ell, \mathbf{b}_\ell)$ | KL distillation to final layer | unbiased; lower perplexity than logit lens; works across BLOOM, GPT-Neo, OPT, Pythia, GPT-NeoX-20B |
| **DoLa** (Chuang et al.) | 2023 | none — uses logit lens at two layers | n/a; inference-time method | factuality-improving decoding |

`[belrose2023-tuned-lens §3, §5;
kb/excerpts/belrose2023-tuned-lens#sec-3-tuned-lens]`,
`[chuang2023-dola FORUM-SIGNAL: arxiv 2309.03883 abstract]`.

### 5.1 DoLa — lens techniques as a decoding strategy

**DoLa** (Decoding by Contrasting Layers) is the cleanest production
application of lens techniques. It contrasts logits at the final
layer $L$ against logits at an adaptively-chosen "premature" layer
$\ell^\star$:

$$\mathrm{logits}_{\mathrm{DoLa}}(t) = \mathrm{logit}^{(L)}(t) - \mathrm{logit}^{(\ell^\star)}(t) \tag{7}$$

with $\ell^\star$ chosen per-token as the layer whose logit
distribution maximally differs (in JS divergence) from the final
layer `[chuang2023-dola FORUM-SIGNAL: arxiv 2309.03883 abstract]`.
The intuition: tokens whose probability rises most sharply between
the early layer and the final layer are the ones the *later* layers
specifically committed to; reweighting toward those tokens preserves
the model's late-stage factual signal while suppressing
hallucination-shaped tokens that are already high at early layers.

DoLa requires no fine-tuning and adds only a modest decoding-time
overhead. Reported gains: improved TruthfulQA scores on LLaMA-7B/13B/33B
`[chuang2023-dola FORUM-SIGNAL]`. The point of including it here:
DoLa is evidence that the logit-lens framework is not just a
diagnostic — the layerwise-readout structure carries decision-relevant
information that can be **fed back into generation**.

### 5.2 Other extensions and applications (from belrose2023 §5)

- **Eliciting latent knowledge via tuned lens** (Cywiński et al.
  2025; cited in `[belrose2023-tuned-lens §5.1;
  kb/excerpts/belrose2023-tuned-lens#sec-5]`). Top-$k$ tuned-lens
  predictions help an auditor LLM guess a "taboo" word the target
  model is trained to suppress. Tuned lens outperforms logit lens.
- **Prompt-injection detection** (Belrose et al. §5.3). Tuned-lens
  prediction trajectories on prompt-injected inputs deviate from the
  on-distribution trajectories detectably; isolation forest / LOF
  outlier detection on the trajectories achieves near-perfect
  detection accuracy `[belrose2023-tuned-lens §5.3;
  kb/excerpts/belrose2023-tuned-lens#sec-5-3]`.
- **Layer-of-acquisition diagnostic.** Belrose §5.4 finds that
  data points which require many training steps to learn also tend
  to be classified at *later* layers in the trained model — the
  prediction trajectory extends deeper for harder examples.

## 6. Intuitions and analogies

[ANALOGY] A lens is **early-exit decoding without retraining**. The
canonical "early exit" method (Schuster et al. 2022) trains a
dedicated head at each layer to predict the final output; the logit
lens skips the training and just reuses $W_U$. The tuned lens is
halfway: it trains a small *change of basis* at each layer but
reuses $W_U$. The analogy returns to canonical form via Eq. (3):
the lens output is exactly the unembed-of-the-mid-layer-state, with
or without an affine pre-rotation.

[INTUITION] Why the unembedding works at intermediate layers at all:
the residual stream is *additively updated* at each layer. The
final-layer hidden state is $\mathbf{h}_L = \mathbf{h}_\ell +
\sum_{\ell'=\ell}^{L-1} F_{\ell'}(\mathbf{h}_{\ell'})$. If
$\sum_{\ell'} F_{\ell'}(\mathbf{h}_{\ell'})$ is small relative to
$\mathbf{h}_\ell$, the unembedding of $\mathbf{h}_\ell$ approximates
the unembedding of $\mathbf{h}_L$. The Mathematical Framework's
**residual-stream-as-bus** picture (Elhage et al. 2021) is the
substrate for this: every layer reads from and writes to the same
linear bus, and the unembedding is just the canonical readout from
that bus. The lens is what happens when you tap the bus at layer
$\ell$ instead of at $L$.

[INTUITION] Why training a translator helps: even on a residual-stream
architecture, hidden states at different layers occupy different
covariance subspaces (Belrose §3 "representational drift"). The logit
lens implicitly assumes the layer-$\ell$ basis is *the same* as the
layer-$L$ basis — true for GPT-2, only approximately true for BLOOM
and GPT-Neo. The tuned lens learns the change-of-basis explicitly. The
fact that this is a *single affine map* per layer (not a deeper
network) is informative: the basis drift is approximately linear, not
learned-feature-dependent.

[ANALOGY] DoLa is to the logit lens what **pseudoderivative** is to
a function: it doesn't read the layer-$\ell$ output as a prediction,
it reads the *change* between layer-$\ell$ and layer-$L$ as a signal
about what the later layers specifically committed to. The analogy
returns to canonical form: $\mathrm{logits}_\mathrm{DoLa}$ is exactly
the difference of two LogitLens evaluations applied to the same
token, so the underlying object is still the layerwise unembed, just
read differentially across depth.

## 7. Frontier and open questions (as of 2026-05)

- **Tuned lenses for modern open-source frontier models.** Belrose
  et al. ship lenses for Pythia, GPT-Neo, GPT-NeoX-20B, OPT, BLOOM,
  GPT-2, and (via lens-transfer) Vicuna-13B. The 2025-2026
  generation (Llama 3 / 4, Qwen-2.5 / 3, DeepSeek-V3, Gemma 3) is
  largely uncovered by published lenses as of writing. The
  `logitlens4llms-2025` extension targets Qwen-2.5 / Llama-3.1; broader
  open lenses are an outstanding need.
- **[CONTRADICTION] Layer-of-prediction vs. layer-of-decision.**
  Halawi et al. 2023 argue that on certain tasks, predictions
  extracted from earlier layers are **more robust** to incorrect
  demonstrations than the final layer. Belrose §5.2 reproduces this
  on three models (BLOOM 560M, Neo 1.3B, Neo 2.7B), but flags that
  *without* peaking at ground truth labels, choosing the right layer
  is hard `[belrose2023-tuned-lens §5.2;
  kb/excerpts/belrose2023-tuned-lens#sec-5]`. So the early-layer
  prediction is sometimes more reliable than the final, but the
  lens-user has no a-priori way to know that.
- **Lens-based interventions vs. activation patching.** Causal Basis
  Extraction is a lens-derived ablation method. Its relationship to
  full activation patching
  (`kb/notes/interpretability/activation-patching.md`) is partly
  formal (CBE finds influential directions; patching tests
  influential states) and partly empirical (Spearman $\rho = 0.89$
  between the two on Pythia 410M is suggestive but not conclusive).
  Whether CBE recovers the "right" causal directions on harder tasks
  (multi-token answers, multi-step reasoning) is open.
- **Lens with non-canonical unembeddings.** The lens framework
  assumes a clean residual-stream-with-final-unembedding architecture.
  Modern instructed models with auxiliary heads (reward heads, value
  heads) or modified output projections (BLT, infini-attention)
  complicate the picture. No published lens for byte-level
  transformers (BLT, Pagnoni et al. 2024) exists.
- **Cross-modal / vision-transformer extension.** "Diffusion Steering
  Lens" (2025, arXiv 2504.13763) extends the lens framework to
  diffusion / vision transformers `[FORUM-SIGNAL]`. Whether the
  iterative-inference structure that makes lenses work on
  language-LLMs also holds for diffusion-style architectures is an
  active question.
- **The post-RoPE / MLA architectural shift.** As MoE / MLA / NSA
  architectures spread (DeepSeek-V2/V3, Llama 4), the residual
  stream is still well-defined but the layer structure is more
  irregular (mixture-of-experts at FFN, latent compression at K/V).
  Whether lens techniques extend cleanly to these is partially
  empirical and not yet documented in print.

## 8. See also

- `kb/notes/interpretability/probing.md` — probes are the
  *trained* readout that lenses contrast with: probes train a new
  classifier on activations; lenses reuse the model's own
  unembedding. Belrose §3.2 explicitly compares the two and notes
  that translators are dramatically smaller than per-layer
  unembeddings would be `[belrose2023-tuned-lens §3.2;
  kb/excerpts/belrose2023-tuned-lens#sec-3-tuned-lens]`.
- `kb/notes/interpretability/sparse-autoencoders.md` — SAEs find a
  *basis* of features in activations; lenses read out a single
  vocabulary distribution. Both bottom out in residual-stream
  geometry.
- `kb/notes/interpretability/activation-patching.md` — patching is
  the canonical *causal* method; CBE is a lens-derived ablation.
- `kb/notes/interpretability/mechanistic-interpretability.md` — the
  broader program. The logit lens was the field's first
  layer-readout diagnostic and remains the cheapest.
- `kb/notes/architecture/transformer-overview.md` — the
  residual-stream + LayerNorm + unembedding setup that lens
  techniques exploit. Without weight-tied / shared-unembedding
  decoder-only architecture, the lens framework needs adaptation.
