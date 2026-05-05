---
topic: interpretability/probing
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - alain-bengio-2017
  - hewitt2019-structural-probe
  - belinkov2022-probing-survey
  - marks-tegmark-2023-truth
secondary_sources:
  - belrose2023-tuned-lens          # lens-vs-probe comparison; §3.2
  - vennemeyer2025-sycophancy-not-one-thing  # difference-in-means probing applied to sycophancy
related_topics:
  - interpretability/lens-techniques
  - interpretability/sparse-autoencoders
  - interpretability/activation-patching
  - interpretability/mechanistic-interpretability
  - alignment/sycophancy
---

# Probing

A **probe** is a (typically linear) classifier trained on **frozen**
neural network activations to predict an external property — truth,
sentiment, syntactic role, factuality, sycophancy, etc. The implicit
claim of the probing methodology is: if a high-accuracy probe exists
for property $y$ in activations $\mathbf{h}$, then the model
**linearly represents** $y$ in $\mathbf{h}$. Probing is the
*correlational supervised* readout — contrasting with the
*correlational unsupervised* readout of SAEs (which find directions
without labels) and the *causal* readout of activation patching
(which intervenes on activations to test what the model *uses*).

The 2016–2021 wave of probing work established the methodology
(Alain & Bengio 2017; Hewitt & Manning 2019). The 2022 review
(Belinkov 2022) cataloged the **probing methodology critique**:
high-accuracy probes can read properties the model *has* but does not
*use*. The 2023+ wave (Marks & Tegmark 2023; Burns 2023;
Vennemeyer 2025) responds with **causal** probing recipes that pair
probes with activation interventions.

## 1. Formal definition

### 1.1 The probing setup

Let $\mathcal{M}$ be a frozen pre-trained transformer with $L$ layers,
hidden size $d$. For an input sequence $\mathbf{x}$, denote the
activation at layer $\ell$, position $t$ as
$\mathbf{h}^{(\ell, t)}(\mathbf{x}) \in \mathbb{R}^d$.

A **linear probe** is a classifier
$g: \mathbb{R}^d \to \mathcal{Y}$ of the form

$$g(\mathbf{h}) = \sigma(\mathbf{w}^\top \mathbf{h} + b) \tag{1}$$

with $\mathbf{w} \in \mathbb{R}^d$, $b \in \mathbb{R}$, $\sigma$ a
sigmoid (binary $\mathcal{Y} = \{0,1\}$) or softmax (multiclass)
output `[alain-bengio-2017 §1; kb/excerpts/alain-bengio-2017#sec-1]`.
The probe is trained on a labeled dataset
$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}$ by minimizing classification
loss

$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{|\mathcal{D}|}\sum_{i} \mathcal{L}_{\mathrm{CE}}(g(\mathbf{h}^{(\ell, t)}(\mathbf{x}_i)),\, y_i) + \lambda \|\mathbf{w}\|_2^2 \tag{2}$$

with the model's own parameters held fixed. The layer $\ell$ and
position $t$ are *hyperparameters*; multi-layer / multi-position
ensembles are common.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $\mathcal{M}$ | frozen transformer being probed |
| $\mathbf{h}^{(\ell, t)} \in \mathbb{R}^d$ | activation at layer $\ell$, position $t$ |
| $\mathbf{w}, b$ | probe parameters (linear case) |
| $g$ | probe function $\mathbb{R}^d \to \mathcal{Y}$ |
| $\mathcal{Y}$ | probe label space (binary, multiclass, or vector for structural probes) |
| $\mathbf{d}_y$ | "direction" for property $y$ — either $\mathbf{w}$ from (1) or $\boldsymbol{\mu}_+ - \boldsymbol{\mu}_-$ from §2.3 |
| $\mathbf{B}$ | structural-probe linear transformation $\in \mathbb{R}^{k \times d}$ |

The *probe accuracy* on held-out data is the headline diagnostic. The
*probe accuracy delta* over a control baseline (e.g., probe trained
on randomly initialized model) operationalizes "the model contains
information about $y$" `[alain-bengio-2017 §2;
kb/excerpts/alain-bengio-2017#sec-1]`.

### 1.2 The structural probe (Hewitt & Manning 2019)

A **structural probe** generalizes the linear probe from binary
classification to a *geometric* probe of higher-arity structure. The
canonical example: probe whether a syntactic dependency tree is
embedded in the activation geometry.

Hewitt & Manning's syntactic distance probe defines a learned linear
transform $\mathbf{B} \in \mathbb{R}^{k \times d}$ such that, under
the squared $L_2$ norm in $\mathbf{B}$-space, the distance between two
words' activations approximates the **parse-tree distance** between
those words `[hewitt2019-structural-probe §abstract;
kb/excerpts/hewitt2019-structural-probe#abstract]`:

$$d_\mathbf{B}(\mathbf{h}_i, \mathbf{h}_j)^2 := \|\mathbf{B}(\mathbf{h}_i - \mathbf{h}_j)\|_2^2 \approx d_T(w_i, w_j) \tag{3}$$

where $d_T(w_i, w_j)$ is the number of edges on the parse-tree path
from token $i$ to token $j$. A second probe variant maps each token's
hidden state to a scalar "tree depth" via a learned norm
$\|\mathbf{B}\mathbf{h}\|_2^2$. Hewitt & Manning find that such
$\mathbf{B}$ exists for ELMo and BERT — but **not for randomly
initialized baselines** — providing evidence that whole syntax trees
are embedded implicitly in deep model geometry.

The structural-probe pattern is the methodological frontier of
probing for *non-binary* properties: instead of asking "does the
model represent $y$?" the question becomes "does the model represent
the *structure* of $y$?".

### 1.3 Mass-mean (difference-in-means) probing

The **mass-mean probe** is a parameter-free alternative used heavily
in the 2023+ truth-direction literature. Given a labeled set
$\{(\mathbf{h}_i, y_i \in \{+, -\})\}$, define

$$\mathbf{d}_{\mathrm{MM}} = \boldsymbol{\mu}_+ - \boldsymbol{\mu}_- = \mathbb{E}_{i: y_i = +}[\mathbf{h}_i] - \mathbb{E}_{i: y_i = -}[\mathbf{h}_i] \tag{4}$$

The probe is the projection $\mathbf{h}^\top \mathbf{d}_{\mathrm{MM}}$,
thresholded at the midpoint
$\tfrac{1}{2}(\boldsymbol{\mu}_+ + \boldsymbol{\mu}_-)^\top \mathbf{d}_{\mathrm{MM}}$.
Marks & Tegmark 2023 argue that despite its simplicity, mass-mean
"performs as well as other probing techniques while identifying
directions which are more causally implicated in model outputs"
`[marks-tegmark-2023-truth §abstract;
kb/excerpts/marks-tegmark-2023-truth#abstract]`. The causal-fidelity
property is the load-bearing claim — see §3.

## 2. Mechanism — the probing protocol

### 2.1 Step-by-step

For probing a property $y$ at layer $\ell$, position $t$:

1. **Curate** a labeled dataset $\{(\mathbf{x}_i, y_i)\}$ where the
   $\mathbf{x}_i$ are *minimal-pair* style (vary only on $y$, not on
   confounders) `[marks-tegmark-2023-truth §2 datasets;
   kb/excerpts/marks-tegmark-2023-truth#sec-2-datasets]`.
2. **Cache** activations $\{\mathbf{h}^{(\ell, t)}_i\}_i$ via one
   forward pass per example (cost: $\sim |\mathcal{D}|$ forward
   passes).
3. **Fit** a probe — logistic regression, mass-mean, structural
   probe — on a train split.
4. **Evaluate** accuracy on held-out test split.
5. **(Optional) generalization tests**: train on dataset $\mathcal{D}_1$
   (e.g., `cities`), evaluate on $\mathcal{D}_2$ (e.g., `neg_cities`,
   `cities_conj`). Probes that fail to transfer are likely reading
   surface confounders, not the abstract property
   `[marks-tegmark-2023-truth §1 contributions;
   kb/excerpts/marks-tegmark-2023-truth#sec-1-contributions]`.
6. **(Optional) causal validation**: ablate $\mathbf{d}_y$ from the
   residual stream during a forward pass; measure behavioral change
   (§3 below).

### 2.2 Layer / position selection — where to probe

The probe-accuracy curve as a function of $\ell$ tells a structural
story about *where* a property is represented:

- **Surface features** (token identity, basic syntactic role) probe
  highest in **early** layers `[hewitt2019-structural-probe §abstract;
  kb/excerpts/hewitt2019-structural-probe#abstract]`.
- **Mid-level features** (syntactic structure, named entity classes)
  peak in **middle** layers, consistent with Alain & Bengio's
  "linear separability increases monotonically with depth" finding
  `[alain-bengio-2017 §abstract;
  kb/excerpts/alain-bengio-2017#sec-1]` — though that result is for
  vision models, not LLMs.
- **Semantic / pragmatic features** (truth, factuality, sentiment,
  speaker stance) often peak in **late-mid** layers and stay accurate
  to the end. Marks & Tegmark 2023 find the truth direction
  localizes around layer 12–15 of LLaMA-2-13B and stays salient
  through layer 35 `[marks-tegmark-2023-truth §3 layers;
  kb/excerpts/marks-tegmark-2023-truth#sec-3-layers]`.

Position-token selection: for free-form text, the **end-of-statement
punctuation token** (period, ".") is often the most-informative probe
position because it summarizes the preceding clause — Marks & Tegmark
note this "summarization behavior" was previously observed by
Tigges et al. 2023 `[marks-tegmark-2023-truth §3 summarization;
kb/excerpts/marks-tegmark-2023-truth#sec-3-layers]`.

### 2.3 The mass-mean recipe in practice

Mass-mean probing on the LLaMA-2-13B `cities` dataset (true / false
"the city of X is in Y" statements):

1. Cache residual-stream activations at the period-token, layer
   $\ell = 12$, for ~1500 true and ~1500 false statements.
2. Compute $\boldsymbol{\mu}_+ \in \mathbb{R}^{5120}$ and
   $\boldsymbol{\mu}_- \in \mathbb{R}^{5120}$ as the per-class means.
3. Form $\mathbf{d}_{\mathrm{MM}} = \boldsymbol{\mu}_+ - \boldsymbol{\mu}_-$.
4. Classify a new statement by sign of $\mathbf{h}^\top
   \mathbf{d}_{\mathrm{MM}} - c$, with $c$ the midpoint threshold.

No optimization, no regularization, no hyperparameter tuning. The
direction is computed in closed form. The simplicity is the point:
absence of moving parts means the discovered direction is a property
of the data, not a property of the optimizer.

## 3. The correlation–causation gap

The methodological critique of probing — articulated by Hewitt &
Liang 2019 ("designing and interpreting probes") and consolidated by
Belinkov 2022 — is: **high-accuracy probes are correlational**. A
probe $g$ that classifies $y$ from $\mathbf{h}$ at layer $\ell$ tells
you *information about $y$ is present* in $\mathbf{h}$. It does *not*
tell you the model *uses* that information at layer $\ell$ to compute
its output. Two confounders motivate this:

1. **Spurious correlate.** $\mathbf{h}$ might encode a feature $z$
   correlated with $y$ in the probe's training data; the probe reads
   $z$, not $y$. Generalization tests across distributions detect
   this (Marks & Tegmark §1, §4).
2. **Vestigial information.** The model might compute $y$ at some
   stage, then the information might persist in $\mathbf{h}$ without
   being used by downstream layers. The probe reads the *trace*, not
   the *driver*.

Belinkov 2022's *Probing Classifiers: Promises, Shortcomings, and
Advances* `[belinkov2022-probing-survey §abstract;
kb/excerpts/belinkov2022-probing-survey#abstract]` consolidates this
critique and reviews "advances" — primarily, **causal** approaches
that go beyond correlational accuracy.

### 3.1 Causal probing — the 2023+ pattern

The resolution: **pair probes with activation interventions** to test
whether the probe's direction is *causally* used by the model. Two
interventions are standard:

- **Direction ablation.** Project $\mathbf{h}$ onto the orthogonal
  complement of $\mathbf{d}_y$:
  $\mathbf{h} \mapsto \mathbf{h} - (\mathbf{h}^\top \hat{\mathbf{d}}_y)\hat{\mathbf{d}}_y$
  with $\hat{\mathbf{d}}_y$ unit-norm. Run the rest of the model from
  this ablated state. If model behavior on the property changes
  significantly, the direction is causally used.
- **Direction addition / subtraction (steering).** Add $\alpha
  \hat{\mathbf{d}}_y$ to $\mathbf{h}$ at a chosen layer; measure
  output change. This is the **activation-steering** family
  (Zou et al. 2023, Rimsky et al. 2024, Vennemeyer 2025).

Marks & Tegmark 2023 perform direction-ablation in LLaMA-2-13B and
report:

> Causal evidence obtained by surgically intervening in a LLM's forward
> pass, causing it to treat false statements as true and vice versa.
> `[marks-tegmark-2023-truth §abstract;
> kb/excerpts/marks-tegmark-2023-truth#abstract]`

The direction found by mass-mean probing flips the model's TRUE/FALSE
prediction when ablated — this is the **strong** version of "the
model linearly represents truth": the linear direction is not just
correlated with truth, *intervening on it changes the model's output
about truth*.

### 3.2 The probe vs. patching duality

Activation patching (`kb/notes/interpretability/activation-patching.md`)
also yields causal claims, but at the granularity of *components*
(MLPs, attention heads), not *directions* in residual-stream space.
The two methodologies are complementary:

- **Patching** isolates the *component* that carries a behavior.
- **Probing + ablation along $\mathbf{d}_y$** isolates the
  *direction within that component* that carries the property.

Marks & Tegmark §3 first localize truth representations to a small
group of hidden states via patching, *then* probe within those
hidden states for the truth direction
`[marks-tegmark-2023-truth §3;
kb/excerpts/marks-tegmark-2023-truth#sec-3-layers]`. This is the
canonical "patch then probe" pipeline.

## 4. Variants and lineage

### 4.1 Probe families

| Variant | Year | Probe form | Best for | Key limitation |
|---|---|---|---|---|
| **Linear logistic regression** | 1990s, Alain & Bengio 2017 for NN | $\sigma(\mathbf{w}^\top \mathbf{h} + b)$ | binary properties | needs regularization to avoid overfitting; correlational |
| **MLP probe** | 2010s | small feedforward net on $\mathbf{h}$ | complex non-linear properties | "probe leakage": a powerful probe can learn $y$ from low-information features `[hewitt-liang-2019, control-tasks]` |
| **Structural probe** (Hewitt & Manning) | 2019 | $\|\mathbf{B}(\mathbf{h}_i - \mathbf{h}_j)\|^2 \approx d_T(w_i, w_j)$ | tree / graph structure | confounded with task-difficulty when comparing layers |
| **Control / random-init baseline** | Hewitt & Liang 2019 | same probe, randomly-initialized model | required as ablation | inflates "accuracy delta" when model is poorly initialized |
| **Logistic regression** (Marks & Tegmark) | 2023 | linear, regularized | truth-style binary | correlational unless paired with ablation |
| **Mass-mean** (Marks & Tegmark) | 2023 | $\boldsymbol{\mu}_+ - \boldsymbol{\mu}_-$ | binary properties; closed-form | requires balanced classes for fair $\mathbf{d}$ |
| **Difference-in-means + steering** (Vennemeyer 2025) | 2025 | $\mathbf{d} = \boldsymbol{\mu}_+ - \boldsymbol{\mu}_-$, then activation-steer | causal probing of behavioral phenomena | within-model only; cross-model transfer untested |
| **Contrastive / unsupervised probes** (Burns CCS 2022) | 2022 | unsupervised consistency loss | truth-style without labels | known generalization issues to negation `[marks-tegmark-2023-truth §1.1; kb/excerpts/marks-tegmark-2023-truth#sec-1-related]` |

### 4.2 Lineage thread: from Alain & Bengio to truth directions

1. **Alain & Bengio 2017** — *Understanding intermediate layers using
   linear classifier probes* (arXiv 1610.01644). Foundational
   methodology paper. Probes vision models (Inception v3, ResNet-50);
   establishes that linear separability of class labels increases
   monotonically with depth.
2. **Hewitt & Manning 2019** — structural probe for syntax in BERT /
   ELMo. Generalizes from binary classification to geometric probing
   of tree structure. NAACL 2019.
3. **Hewitt & Liang 2019** — control tasks: a probe should be tested
   against a random-label baseline to control for "probe expressivity".
4. **Tenney et al. 2019** — *BERT Rediscovers the Classical NLP
   Pipeline*. Layer-wise probing for POS, NER, dependency parsing,
   coref shows BERT roughly recapitulates the classical pipeline
   layer-by-layer.
5. **Belinkov 2022** — *Probing Classifiers: Promises, Shortcomings,
   and Advances*. Computational Linguistics 48(1). Methodology
   review; consolidates the correlational critique
   `[belinkov2022-probing-survey §abstract;
   kb/excerpts/belinkov2022-probing-survey#abstract]`.
6. **Burns et al. 2022 (CCS)** — *Discovering Latent Knowledge in
   LLMs Without Supervision*. Contrastive consistency search; finds
   "truth direction" without labels. Subsequent work finds
   generalization issues, especially under negation.
7. **Marks & Tegmark 2023** — *The Geometry of Truth*. Establishes
   linear truth representation in LLaMA-2 family, validates with
   patching, recommends mass-mean
   `[marks-tegmark-2023-truth §abstract;
   kb/excerpts/marks-tegmark-2023-truth#abstract]`.
8. **Zou et al. 2023, Rimsky et al. 2024** — Representation Engineering
   / activation steering. Probing-derived directions used at
   inference-time to *control* model behavior.
9. **Vennemeyer et al. 2025** — Difference-in-means probing applied
   to sycophancy decomposition; finds three orthogonal directions
   (agreement / praise / genuine-agreement) that can be steered
   independently. See `kb/notes/alignment/sycophancy.md`.

## 5. Intuitions and analogies

[ANALOGY] A probe is a **psychophysics experiment on a frozen
model**: like flashing stimuli at a subject and asking whether they
can press button A or B, you flash inputs at the model, freeze its
brain (literally — gradients off), and ask whether a *cheap detector*
applied to one of its layers can recover a label. The analogy returns
to canonical form via Eq. (1): the "cheap detector" is the linear
classifier $\mathbf{w}^\top \mathbf{h}$; "can recover" is the probe
accuracy.

[INTUITION] Why "linear" probes specifically? The model's own
unembedding is a linear map. If the model uses a property $y$ to
make its decision, the property must be linearly decodable by the
unembedding from the final hidden state — otherwise the model
couldn't act on it. So a *linear* probe is in some sense the right
test: it asks "is the property in a form the model could itself use".
Deeper / non-linear probes can find information the model has but
*could not use* — that's the "probe leakage" failure mode.

[ANALOGY] The mass-mean direction is the **first principal component
of the class-difference distribution**. It's the axis along which the
two classes are maximally separated in expectation. The analogy
returns to canonical form via Eq. (4): $\mathbf{d}_{\mathrm{MM}} =
\boldsymbol{\mu}_+ - \boldsymbol{\mu}_-$ is the mean of the +
distribution minus the mean of the − distribution; projecting onto
this axis is exactly the LDA-style "class-difference projection"
familiar from classical pattern recognition.

[CONTRADICTION] Whether the **same direction** that probing finds is
the *direction the model uses*. Marks & Tegmark §3 argue yes (their
ablation experiments work). But Marks & Tegmark §4 also document
**stark misalignment** between truth directions for different
datasets:

> The axes of separation for various true/false datasets align often,
> but not always […] in early layers, `cities` and `neg_cities`
> separate antipodally, before rotating to lie orthogonally […], and
> finally aligning in later layers.
> `[marks-tegmark-2023-truth §4;
> kb/excerpts/marks-tegmark-2023-truth#sec-4-misalignment]`

Different prompt formats yield different truth directions until late
layers, suggesting the model's "truth representation" is itself
multi-component, not a single axis. The 2025 Vennemeyer result that
sycophancy decomposes into three independently-steerable directions
generalizes this: behaviorally-singular properties may be
representationally-plural.

[INTUITION] Causal probing changes the question from
"*does* the model represent $y$" to "*does the model use* its
representation of $y$". Probing alone answers the first; probing +
direction-ablation answers the second. The former is necessary but
not sufficient for downstream interpretability claims; the latter is
what alignment-relevant claims require.

## 6. Frontier and open questions (as of 2026-05)

- **Do probes and SAE features find the same directions?**
  Both methodologies produce a vector
  $\mathbf{d} \in \mathbb{R}^d$ for a "concept". Are the truth
  direction (Marks & Tegmark) and the truth-related SAE latent
  (Templeton 2024) the *same* vector, up to scale? As of 2026 this
  question is informally addressed in Neuronpedia-style writeups but
  not, to our knowledge, systematically measured at scale. The
  cosine similarity between probe directions and SAE latents on
  matched concepts is a missing benchmark.
  See `kb/notes/interpretability/sparse-autoencoders.md`.
- **[CONTRADICTION] Generalization of mass-mean probing to non-binary
  properties.** Truth (binary) admits a linear separating direction.
  Multi-class properties ("the answer is in {Paris, London, Madrid,
  Berlin}") do not have a single mass-mean direction; whether
  multi-class linear probes recover causally-implicated subspaces
  the way mass-mean does for binary is unresolved.
- **Probing for safety-relevant properties.** Behavioral probes for
  deception, refusal-circumvention, sycophancy, and scheming
  (`kb/notes/alignment/safety-evaluation.md`,
  `kb/notes/alignment/sycophancy.md`) are increasingly load-bearing
  in alignment evaluation. The question of whether such probes
  generalize from in-distribution evaluation prompts to adversarial
  out-of-distribution behavior is **the** open question for
  alignment uses of probing.
- **Probe-derived steering in production.** Activation-steering at
  inference time (representation engineering, Vennemeyer 2025) is
  cheap and avoids retraining; whether it composes with RLHF /
  RLAIF post-training in a stable way is being actively studied.
  Side-effect risk: steering one direction may drag along
  unrelated directions in the residual stream's covariance subspace.
- **Probing across model families.** A truth direction in
  LLaMA-2-13B and a truth direction in Qwen-2.5-14B almost certainly
  do not have the same coordinates (different bases). Cross-family
  probe transfer is not generically supported. Methodological
  question: is there a "canonical truth direction" up to model-
  specific basis, or are truth directions fundamentally
  model-dependent?
- **Probe scaling laws.** Linear-probe accuracy for fixed properties
  appears to scale with model size — but the scaling exponent is not
  rigorously characterized. A scaling-law-style study of
  $\text{ProbeAcc}(N, D)$ for canonical properties (truth,
  sentiment, syntactic depth) across model scales is a missing piece.

## 7. See also

- `kb/notes/interpretability/lens-techniques.md` — **vocabulary-projection
  readout**; the "untrained probe" baseline. Lenses reuse the model's
  own $W_U$; probes train a fresh classifier. Belrose §3.2 explicitly
  compares the two
  `[belrose2023-tuned-lens §3.2;
  kb/excerpts/belrose2023-tuned-lens#sec-3-tuned-lens]`.
- `kb/notes/interpretability/sparse-autoencoders.md` — **unsupervised
  feature decomposition**. SAE feature activations are concept-readouts
  *without* labels; probes are concept-readouts *for* labels. Both
  produce directions in $\mathbb{R}^d$.
- `kb/notes/interpretability/activation-patching.md` — the **causal
  intervention** that turns a correlational probe into a causal
  claim. The "patch then probe" pipeline is the 2023+ standard for
  causal probing.
- `kb/notes/interpretability/mechanistic-interpretability.md` — the
  broader program. Probing is the supervised cousin of MI's
  feature-extraction methods; MI uses probing-derived directions as
  starting points for mechanism hypothesis.
- `kb/notes/alignment/sycophancy.md` — Vennemeyer 2025's
  difference-in-means probing of sycophancy is the paradigmatic
  alignment-application of probing.
- `kb/notes/alignment/safety-evaluation.md` — behavioral probes for
  deception / refusal / scheming feed into safety evaluation
  pipelines.
