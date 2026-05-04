# Interpretability — glossary additions (Phase 2)

These fragments are intended to be merged into `theory/kb/glossary.md`
under the indicated section headings (or new sections). Format: one
`## Section heading` block per topic cluster, each followed by
`- **Term** — definition. citation.` lines, matching the pilot
glossary format (cf. "Attention variants and KV-cache compression").

## Mechanistic interpretability — core vocabulary

- **Mechanistic interpretability (MI)** — the research program of
  reverse-engineering neural networks into human-readable algorithms,
  at the level of circuits (sub-graphs) and features (directions).
  Distinct from behavioral interpretability (input/output black-box
  evaluation) and post-hoc rationalization. `[wang2022-ioi §1; meng2022-rome §1]`.
- **Circuit** — a sub-graph $C$ of a model's computational graph $M$
  that, when used in place of $M$ via knockouts of $M \setminus C$,
  approximately reproduces $M$'s behavior on a target task
  `[wang2022-ioi §2; kb/excerpts/wang2022-ioi#sec-2-definition]`.
- **Feature** — a direction $\mathbf{d} \in \mathbb{R}^n$ in some
  activation space such that the projection $\mathbf{x}^\top \mathbf{d}$
  (or equivalently, an SAE latent activation with decoder column
  $\mathbf{d}$) corresponds to a human-interpretable property. The
  unit of analysis at the *what* level; circuits operate at the *how*
  level.
- **Faithfulness, completeness, minimality** — the three quantitative
  validity criteria for a circuit. Faithfulness: $C$ matches $M$'s
  performance on the task. Completeness: $C$ contains every node
  needed. Minimality: every node in $C$ is necessary `[wang2022-ioi §4;
  kb/excerpts/wang2022-ioi#sec-4-criteria]`.
- **Polysemanticity** — the property that an individual neuron
  responds to multiple unrelated concepts. The motivation for SAEs
  (which lift polysemanticity by inflating the basis). `[leask2025-sae-not-canonical §1]`.
- **Monosemanticity** — the property that a feature responds to a
  single, human-interpretable concept. The (contested) goal of SAE
  training; "Towards Monosemanticity" (Bricken 2023) is the founding
  reference. [CONTRADICTION] Whether SAEs achieve true monosemanticity
  is challenged by `[leask2025-sae-not-canonical §1; kb/excerpts/leask2025-sae-not-canonical#sec-1-canonical]`.
- **Superposition** — the hypothesis (Elhage 2022) that a network with
  $n$ neurons can represent more than $n$ approximately-orthogonal
  features as long as they fire sparsely. SAEs lift superposition by
  inflating the dictionary size $M \gg n$. `[gao2024-topk-saes §1]`.
- **Residual stream** — the linear data bus running through a
  decoder-only transformer's layers; every sublayer (attention, MLP)
  reads from and additively writes to it. The substrate that lens
  techniques and SAEs decompose. Vocabulary established in Elhage et
  al. 2021 (Mathematical Framework for Transformer Circuits).

## Activation patching — methodology vocabulary

- **Activation patching** — the canonical causal-intervention method
  in MI: identify which model components matter for a behavior by
  swapping their activations between forward passes on paired
  (clean, corrupted) prompts and measuring the effect on the output
  `[meng2022-rome §2; kb/excerpts/meng2022-rome#sec-2; zhang2023-apatching §2.1]`.
  Also known as causal tracing, interchange intervention, causal
  mediation analysis, or representation denoising.
- **Causal tracing** — Meng et al. 2022's specific implementation of
  activation patching: clean run + Gaussian-noised corrupted run +
  patched-with-clean-state run, computed over every (token, layer)
  cell to produce a heatmap of indirect effects
  `[meng2022-rome §2.2; kb/excerpts/meng2022-rome#sec-2-2]`.
- **Indirect effect (IE)** — the change in the output metric when the
  activation at a specific (component, position, layer) is restored
  from a cached clean run during an otherwise-corrupted forward pass.
  $\text{IE} = \mathbb{P}_{*,\,\text{clean }h_i^{(l)}}[r] -
  \mathbb{P}_*[r]$ `[meng2022-rome §2 effects; kb/excerpts/meng2022-rome#sec-2-effects]`.
- **Average indirect effect (AIE)** — IE averaged over a dataset of
  paired prompts; the standard metric for component importance.
- **Path patching** — a refinement of activation patching that
  isolates the direct effect of a component $C$ on a downstream
  component $R$, holding fixed the indirect routes through other
  components. Introduced in Wang et al. 2022 (IOI)
  `[wang2022-ioi §3.1; kb/excerpts/wang2022-ioi#sec-3-1-path-patching]`.
- **Attribution patching** — gradient-based first-order approximation
  of the patching effect: $\text{IE}_C \approx \langle \nabla_h m,
  \Delta h \rangle$. Cost: 1 forward + 1 backward pass for all
  components. Tier B (Nanda blog); approximation breaks in non-linear
  regions.
- **Gaussian Noising (GN) / Symmetric Token Replacement (STR)** — the
  two corruption methods for activation patching. GN adds
  $\mathcal{N}(0, 3\sigma_\text{textset})$ to subject-token embeddings
  `[meng2022-rome §2]`; STR replaces the subject with a similar token
  (e.g., "Eiffel Tower" → "Colosseum") preserving sequence length
  `[zhang2023-apatching §2.1; kb/excerpts/zhang2023-apatching#sec-2-1-corruption]`.
  STR is the recommended default; GN can drive activations
  off-distribution `[zhang2023-apatching §1 findings]`.
- **Logit difference (LD)** — an activation-patching metric:
  $\text{LD}(r, r') = \text{Logit}(r) - \text{Logit}(r')$. The
  patching effect, normalized to $[0, 1]$, is $[\text{LD}_\text{pt} -
  \text{LD}_*] / [\text{LD}_\text{cl} - \text{LD}_*]$
  `[zhang2023-apatching §2.1 metrics; kb/excerpts/zhang2023-apatching#sec-2-1-metrics]`.
  Recommended over raw probability because it is sensitive to
  components that *push against* the correct answer.
- **Mean ablation** — knocking out a node by replacing its activation
  with the mean activation across a reference distribution; preferred
  to zero ablation, which is arbitrary `[wang2022-ioi §2;
  kb/excerpts/wang2022-ioi#sec-2-definition]`.
- **Knockout** — generic term for setting a component's contribution
  to a fixed reference value (zero or mean) for circuit-validation
  purposes `[wang2022-ioi §2.1]`.

## Sparse autoencoders — architecture vocabulary

- **Sparse autoencoder (SAE)** — a wide ($M \gg n$) two-layer
  encoder-decoder trained to reconstruct a model's activations
  $\mathbf{x} \in \mathbb{R}^n$ as a sparse linear combination of
  learned dictionary directions $\{\mathbf{d}_i\}_{i=1}^M$
  `[rajamanoharan2024-jumprelu §2 Eq.1-2; kb/excerpts/rajamanoharan2024-jumprelu#sec-2]`.
- **Dictionary / latents** — the columns $\mathbf{d}_i$ of
  $\mathbf{W}_\text{dec}$; each is a feature direction in
  $\mathbb{R}^n$. Templeton 2024 distinguishes "feature" (the
  conceptual entity the model represents) from "latent" (the SAE's
  learned approximation). `[gemma-scope-2024 §2.1]`.
- **Expansion factor** — the ratio $M / n$ of dictionary size to
  activation dimension. Typical values 8–1000.
- **L0 sparsity** — the headline sparsity metric: the number of
  non-zero feature activations per token. The $x$-axis of the
  reconstruction-fidelity Pareto plots
  `[rajamanoharan2024-jumprelu §1 Pareto; kb/excerpts/rajamanoharan2024-jumprelu#sec-1-pareto]`.
- **TopK SAE** — SAE variant with $\sigma = \text{TopK}(\cdot)$:
  zero all but the top $K$ pre-activations. Direct $L_0$ control, no
  L1 penalty needed. OpenAI 2024 `[gao2024-topk-saes §2.3 Eq.2;
  kb/excerpts/gao2024-topk-saes#sec-2-3]`.
- **JumpReLU SAE** — SAE variant with a discontinuous activation
  $\text{JumpReLU}_\theta(z) = z \cdot H(z - \theta)$ and
  per-feature learned threshold $\theta$. Trained with L0 penalty via
  straight-through estimators. SOTA reconstruction at fixed L0 on
  Gemma 2 9B `[rajamanoharan2024-jumprelu §3 Eq.4, §1 Pareto]`.
- **Gated SAE** — SAE variant with two encoder kernels, one for
  gating (binary on/off) and one for magnitude estimation.
  Architecturally equivalent (with weight sharing) to JumpReLU
  `[rajamanoharan2024-jumprelu §2]`.
- **Straight-through estimator (STE)** — a backward-pass surrogate
  used to backprop through the discontinuous JumpReLU and the L0
  penalty. The forward pass uses the discontinuity; the backward
  pass uses a smooth (kernel-density-estimated) gradient
  `[rajamanoharan2024-jumprelu Abstract; rajamanoharan2024-jumprelu §3]`.
- **Dead latent** — an SAE feature that has stopped activating across
  the training distribution (typically because its encoder weights
  collapsed during early training). Up to 90% of latents in a large
  SAE can be dead without mitigation `[gao2024-topk-saes §2.4;
  kb/excerpts/gao2024-topk-saes#sec-2-4]`.
- **AuxK loss** — auxiliary loss added by Gao et al. 2024 to revive
  dead latents: model the residual reconstruction error using the
  top-$k_\text{aux}$ dead latents `[gao2024-topk-saes §2.4]`.
- **Shrinkage** — the bias introduced by the L1 penalty in ReLU SAEs:
  positive feature activations are systematically underestimated, so
  reconstructed activations have smaller magnitudes than true values.
  Avoided by TopK and JumpReLU `[gao2024-topk-saes §2.3 benefits]`.
- **SAE stitching** — Leask et al. 2025: insert/swap latents from a
  larger SAE into a smaller one to compare bases. Reveals novel
  latents (in the larger but not the smaller SAE) and reconstruction
  latents (in both with similar behavior)
  `[leask2025-sae-not-canonical §1; kb/excerpts/leask2025-sae-not-canonical#sec-1-techniques]`.
- **Meta-SAE** — an SAE trained on the *decoder matrix* of another
  SAE. Reveals that latents in a larger SAE often decompose into
  sparse combinations of meta-latents, contradicting the "atomic
  features" assumption `[leask2025-sae-not-canonical §1 Einstein;
  kb/excerpts/leask2025-sae-not-canonical#sec-1-einstein]`.
- **Canonical features (contested)** — the (challenged) hypothesis
  that SAEs find a unique, complete, atomic decomposition of model
  activations into features. [CONTRADICTION] Leask et al. 2025
  argue against canonicality with both stitching and meta-SAE
  evidence `[leask2025-sae-not-canonical §1; kb/excerpts/leask2025-sae-not-canonical#sec-1-canonical]`.

## Transcoders and circuit tracing

- **Transcoder** — a wide ReLU MLP (encoder–decoder) trained to
  approximate the input/output behavior of an MLP sublayer:
  $\text{TC}(\mathbf{x}) = \mathbf{W}_\text{dec}\,\text{ReLU}(\mathbf{W}_\text{enc}
  \mathbf{x} + \mathbf{b}_\text{enc}) + \mathbf{b}_\text{dec}$, with
  L1 sparsity penalty on the hidden activations
  `[dunefsky2024-transcoders §3.1 Eq.3-5; kb/excerpts/dunefsky2024-transcoders#sec-3-1]`.
  Distinct from an SAE: a transcoder approximates a *function* (the
  MLP); an SAE approximates a *vector* (an activation).
- **Cross-layer transcoder** — Lindsey et al. 2025: a transcoder
  whose decoder writes into the residual stream of *all subsequent*
  layers, not just the immediate next one. Makes a feature's
  skip-layer contributions explicit in the attribution graph.
  HTML-only at transformer-circuits.pub.
- **Input-invariance** — the property of a transcoder (and *not* of
  an SAE-on-MLP-output) that the connections between features can be
  stated independently of any specific input. Necessary for
  weight-based circuit analysis through MLPs
  `[dunefsky2024-transcoders §1; kb/excerpts/dunefsky2024-transcoders#sec-1-input-invariance]`.
- **Attribution graph** — the output of circuit tracing: a sparse
  directed graph from input embeddings to output unembedding, with
  feature activations as nodes and Jacobian-weighted attribution
  scores as edges. Computed in 1 forward + 1 backward pass on the
  transcoder-replaced model (Lindsey 2025).
- **Circuit tracing** — Anthropic's 2025 methodology: replace MLPs
  with cross-layer transcoders, then compute attribution graphs by
  Jacobian back-attribution from a target node. Synthesis of SAE
  feature decomposition + activation patching causal intervention.
  `[lindsey2025-circuit-tracing FORUM-SIGNAL — HTML-only]`.

## Lens techniques — vocabulary

- **Logit lens** — apply the final LayerNorm + unembedding $W_U$ to
  an intermediate residual-stream activation $\mathbf{h}^{(\ell)}$
  to read out a layer-$\ell$ vocabulary distribution. nostalgebraist
  2020 (LessWrong, Tier B). Formal expression and decomposition in
  `[belrose2023-tuned-lens §2 Eq.3]` (PDF p.2).
- **Tuned lens** — Belrose et al. 2023: per-layer affine
  "translators" $\mathbf{h}_\ell \mapsto A_\ell \mathbf{h}_\ell +
  b_\ell$ trained with a distillation loss against final-layer
  logits. More reliable than the raw logit lens, especially on
  BLOOM/GPT-Neo `[belrose2023-tuned-lens §1; PDF p.1-2]`.
- **Causal Basis Extraction (CBE)** — Belrose et al. 2023's algorithm
  for finding residual-stream directions with highest influence on
  the tuned lens output. Connection between lens analysis and MI
  feature attribution.
- **DoLa** — *Decoding by Contrasting Layers*: an inference-time
  application of lens analysis. The decoded logits are
  $\text{logit}^{(L)}(t) - \text{logit}^{(\ell^*)}(t)$ for an
  adaptively chosen early layer $\ell^*$, improving factuality.
  Chuang et al. 2023.

## Probing — vocabulary

- **Probe** — a (typically linear) classifier $g(\mathbf{h}) =
  \sigma(\mathbf{w}^\top \mathbf{h} + b)$ trained on labeled
  activations to predict an external property $y$. Foundational
  reference: Alain & Bengio 2017.
- **Mass-mean probing** — Marks & Tegmark 2023: define the probe
  direction as $\mathbf{d}_\text{class} = \mathbb{E}[\mathbf{h} \mid
  y = c_1] - \mathbb{E}[\mathbf{h} \mid y = c_0]$, then project
  activations onto $\mathbf{d}_\text{class}$. Simpler and more
  generalization-robust than logistic regression on the same
  features `[marks-tegmark-2023-truth §1]`.
- **Truth direction** — the canonical mass-mean probe direction for
  truth-value of a statement, derived in Marks & Tegmark 2023. A
  small linear subspace separates true and false statements across
  many models and datasets.
- **Correlational vs. causal probing** — a probe demonstrates that
  the model *represents* a property; an activation patch along the
  probe direction demonstrates that the model *uses* it. The 2023+
  literature is converging on always pairing the two.
