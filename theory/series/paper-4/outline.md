# Paper 4 — *The internal computation can be partially read*

**Working title:** *The internal computation can be partially read: lenses, probes, sparse autoencoders, activation patching, and circuit tracing as a method-pluralist program, 2017–2026.*

**Thesis:** Five interpretability methods — **lens techniques**, **probes**, **sparse autoencoders (SAEs)**, **activation patching**, and **circuit tracing** — make different commitments about what counts as a *feature*, a *circuit*, and an *explanation*. Method-pluralism is principled, not a sign of immaturity: each method answers a different question about the model's internal computation, and the five together let the field make claims it could not make with any one in isolation. Cross-method evidence converges in some cases (the GPT-2-small IOI circuit; ROME's mid-layer factual-recall localization) and diverges in others (the geometry of "truth" across datasets, the canonicality of SAE features). We name the points of convergence, the points of divergence, and what evidence would resolve each.

**Length target:** ~70 pages.

**Audience:** ML researchers and engineers with graduate-level math who want a single grounded reference for the modern interpretability stack — not a vendor-neutral survey, but a method-by-method comparison that takes the disagreements between methods seriously rather than papering them over. Assumes Paper 1's residual-stream / FFN / attention notation and Paper 2's training-pipeline vocabulary (probes are typically read off training-time activations).

## Section structure (target 14 sections)

For each section: **(thesis sentence)** • [page budget] • {primary KB anchors} • {key contradictions to surface}.

### §1 — Introduction (4 pages)

Interpretability is method-plural because the underlying object — a frontier transformer's internal computation — is plural. We list the five methods, the question each answers, and the dimensions along which they commit (correlational vs causal; component-level vs feature-level; supervised vs unsupervised; static-readout vs intervention). Preview of the convergence/divergence pattern documented in §11. What "partially read" means: we can extract human-legible structure for *some* behaviors, on *some* models, with cross-method agreement on *some* claims — not a complete reverse-engineering, and the field knows it.

- Anchors: `kb/notes/interpretability/{lens-techniques,probing,sparse-autoencoders,activation-patching,mechanistic-interpretability,circuit-tracing}.md` (the six method notes), `kb/index/topics.md` interpretability cluster, `kb/index/contradictions.md` § interpretability.
- Closing question for §11: when do two methods *have* to agree, and what does it mean when they don't?

### §2 — What we mean by "internal computation": substrate and notation (5 pages)

The residual stream as a linear additive bus (Elhage et al. 2021 "Mathematical Framework"); attention as QK + OV decompositions; MLP/FFN as key-value associative memory candidate; LayerNorm/RMSNorm and the unembedding $W_U$ as the canonical readout. Tensor-shape table reused from Paper 1: $B \times S \times D$, head-split $B \times H \times S \times d_h$, vocabulary $|\mathcal{V}|$, layer count $L$. Define what a **feature** is in this paper: a direction $\mathbf{d} \in \mathbb{R}^d$ in residual-stream space (not a neuron, not necessarily a basis vector). Define what a **circuit** is: a subgraph of the model's computational graph satisfying the Wang-2022 *faithfulness / completeness / minimality* criteria.

- Anchors: `kb/notes/interpretability/mechanistic-interpretability.md` §1; `kb/notes/architecture/transformer-overview.md` §residual-stream (Paper 1 §2 cross-ref); `kb/excerpts/wang2022-ioi.md#sec-2-definition` (circuit definition), `#sec-4-criteria` (the three criteria).
- Math to transcribe: residual update $\mathbf{h}_{\ell+1} = \mathbf{h}_\ell + F_\ell(\mathbf{h}_\ell)$ `[belrose2023-tuned-lens §2 Eq.1]`, the unembed-from-layer-$\ell$ identity $\mathcal{M}_{>\ell}(\mathbf{h}_\ell) = \mathrm{LN}[\mathbf{h}_\ell + \sum_{\ell'\geq \ell} F_{\ell'}] W_U$ `[belrose2023-tuned-lens §2 Eq.2; kb/excerpts/belrose2023-tuned-lens#sec-2-decomposition]`.
- Notation table fixed for the rest of the paper: $\mathbf{h}^{(\ell, t)}$ for activation at layer $\ell$, position $t$; $\mathbf{d}_y$ for a direction associated with property $y$; $f_i^{(\ell, t)}$ for SAE/transcoder feature $i$ activation; $C$ for a circuit subgraph.

### §3 — Lens techniques: the cheapest readout (5 pages)

Lenses project intermediate hidden states through the model's *own* unembedding $W_U$ to obtain a layer-$\ell$ "as-if-final" distribution over the vocabulary. Logit lens (nostalgebraist 2020, LessWrong, FORUM-SIGNAL) is the no-training baseline; tuned lens (Belrose 2023) inserts a per-layer affine **translator** $(A_\ell, \mathbf{b}_\ell)$ trained by KL distillation against the final-layer output. We document why the logit lens fails on BLOOM and GPT-Neo (bias, representational drift, cross-model unreliability), why the tuned lens fixes it, and how DoLa (Chuang 2023) turns the lens framework into a decoding-time hallucination-suppression strategy by *contrasting* logits across layers.

- Anchors: `kb/notes/interpretability/lens-techniques.md` §1-5; `kb/excerpts/belrose2023-tuned-lens.md#sec-1-iterative`, `#sec-2-logit-lens`, `#sec-2-failures`, `#sec-3-tuned-lens`, `#sec-3-loss`, `#sec-4-cbe`.
- Math to transcribe: $\mathrm{LogitLens}(\mathbf{h}_\ell) = \mathrm{LN}[\mathbf{h}_\ell] W_U$ (Eq. 3); $\mathrm{TunedLens}_\ell(\mathbf{h}_\ell) = \mathrm{LogitLens}(A_\ell \mathbf{h}_\ell + \mathbf{b}_\ell)$ (Eq. 4); the tuned-lens distillation loss $\mathcal{L}(\ell) = \mathbb{E}_\mathbf{x}[D_\mathrm{KL}(\mathcal{M}_{>\ell}(\mathbf{h}_\ell) \| \mathrm{TunedLens}_\ell(\mathbf{h}_\ell))]$ (Eq. 5); the DoLa contrast $\mathrm{logits}_\mathrm{DoLa}(t) = \mathrm{logit}^{(L)}(t) - \mathrm{logit}^{(\ell^*)}(t)$.
- Diagnostic: the **prediction trajectory** $\{\mathrm{Lens}_\ell(\mathbf{h}_\ell)\}_{\ell=0}^L$ and what its smoothness / convergence rate tell us about a prompt (clean vs prompt-injected; easy vs late-layer-acquisition).
- Cost: $L \cdot (d^2 + d)$ tuned-lens parameters — for Pythia 12B with $d=5120, L=36$, ~944M params, dramatically smaller than per-layer probing unembeddings.
- Contradiction `[CONTRADICTION]`: layer-of-prediction vs layer-of-decision (Halawi 2023 — sometimes early-layer predictions are *more* robust than the final layer; choosing the right layer has no a-priori recipe).
- Frontier-2026 gap: tuned lenses for Llama 3/4, Qwen 2.5/3, DeepSeek-V3, Gemma 3 are not all published. `logitlens4llms-2025` is the partial cover.

### §4 — Probing: the supervised correlational readout (6 pages)

A **probe** is a (typically linear) classifier trained on **frozen** activations to predict an external property — truth, sentiment, syntactic role, factuality, sycophancy. The implicit claim: high-accuracy probe $\Rightarrow$ model linearly represents the property. We trace four probe families: (i) Alain–Bengio 2017 linear logistic regression (foundational, originally on vision models); (ii) Hewitt–Manning 2019 **structural probe** with $d_\mathbf{B}(\mathbf{h}_i, \mathbf{h}_j)^2 \approx d_T(w_i, w_j)$ for parse-tree distance in BERT/ELMo; (iii) Burns 2022 CCS for unsupervised consistency-based truth probing; (iv) Marks–Tegmark 2023 **mass-mean** $\mathbf{d}_\mathrm{MM} = \boldsymbol{\mu}_+ - \boldsymbol{\mu}_-$, parameter-free and closed-form. We document the layer-position story (surface features early, syntax mid, semantic/pragmatic late-mid) and the period-token "summarization" position trick.

- Anchors: `kb/notes/interpretability/probing.md` §1-4; `kb/excerpts/{alain-bengio-2017,hewitt2019-structural-probe,belinkov2022-probing-survey,marks-tegmark-2023-truth}.md` (all PDF-grounded).
- Math to transcribe: Eq. 1 (linear probe); Eq. 2 (cross-entropy + L2 training loss); Eq. 3 (structural-probe squared $L_2$ in $\mathbf{B}$-space); Eq. 4 (mass-mean direction).
- Lineage table: 8 probe families × {year, probe form, best-for, key limitation}, lifted from `probing.md` §4.1.
- Practical recipe: the 6-step probing protocol (curate minimal-pair dataset → cache activations → fit probe → evaluate held-out → generalization tests → optional causal validation).
- Cross-paper thread: probes are read from training-time activations (Paper 2 pipeline §1 pre-training output stage).

### §5 — The correlation–causation gap, and how the field closed it (5 pages)

The methodological critique of probing — Hewitt–Liang 2019, consolidated by Belinkov 2022 *Probing Classifiers: Promises, Shortcomings, and Advances*: high-accuracy probes are **correlational**. Two confounders: **spurious correlate** (probe reads $z$ correlated with $y$) and **vestigial information** (model computed $y$ but no longer uses it downstream). The 2023+ resolution: pair probes with activation interventions. **Direction ablation** $\mathbf{h} \mapsto \mathbf{h} - (\mathbf{h}^\top \hat{\mathbf{d}}_y)\hat{\mathbf{d}}_y$ tests whether the direction is causally used; **direction addition** $\mathbf{h} \mapsto \mathbf{h} + \alpha \hat{\mathbf{d}}_y$ is the activation-steering family (Zou 2023, Rimsky 2024, Vennemeyer 2025). Marks–Tegmark 2023 is the canonical worked example: ablating $\mathbf{d}_\mathrm{MM}$ flips LLaMA-2-13B's TRUE/FALSE behavior. We connect this to §8: the "patch-then-probe" pipeline localizes truth representations to a small group of hidden states *via patching*, then probes within those states.

- Anchors: `kb/notes/interpretability/probing.md` §3; `kb/excerpts/{belinkov2022-probing-survey,marks-tegmark-2023-truth}.md#abstract`, `marks-tegmark-2023-truth#sec-3-layers`, `#sec-4-misalignment`.
- Math to transcribe: direction-ablation operator (orthogonal projector $I - \hat{\mathbf{d}}_y \hat{\mathbf{d}}_y^\top$); direction-addition steering rule.
- Cross-paper thread `[Paper 3 §11]`: CoT faithfulness as a probing question — is a probe of "the model is using its CoT" a correlational artifact, or causally implicated?
- Cross-paper thread `[Paper 5]`: behavioral probes for deception, refusal-circumvention, sycophancy, scheming as alignment-evaluation surface.

### §6 — Truth directions and the "geometry of truth" debate (4 pages)

The most-studied probing target since 2022. Marks–Tegmark 2023 *The Geometry of Truth* establishes a linear truth representation in LLaMA-2 family at layers 12–15, validates with patching, and recommends mass-mean. Burns 2022 CCS finds a truth-like direction without labels but suffers documented generalization issues under negation. The unresolved issue: **the same model's truth direction is dataset-dependent in early/mid layers, only aligning across `cities`/`neg_cities`/`cities_conj` in late layers** (Marks–Tegmark §4 "stark misalignment"). The 2025 Vennemeyer result that *sycophancy* decomposes into three independently-steerable directions (agreement / praise / genuine-agreement) generalizes the multi-component finding to other behavioral phenomena: **behaviorally-singular properties may be representationally-plural**. We catalog what the field has converged on (truth is linearly decodable from late-mid layers in LLaMA-family models on factual statements) and what remains open (cross-prompt-format alignment, cross-model-family transfer, multi-class generalization of mass-mean).

- Anchors: `kb/notes/interpretability/probing.md` §3.1, §6 misalignment; `kb/excerpts/marks-tegmark-2023-truth.md#sec-3-layers`, `#sec-4-misalignment`, `#sec-1-related` (Burns CCS critique).
- Contradiction `[CONTRADICTION]` block: dataset-dependent truth direction; CCS generalization-under-negation failure; the "is there a canonical truth direction" question and what would resolve it.
- Cross-paper thread: behavioral probes feed Paper 5 (alignment threats — sycophancy, scheming, alignment-faking) as detection methodology.

### §7 — Sparse autoencoders: the unsupervised feature framing (6 pages)

An SAE decomposes activations $\mathbf{x} \in \mathbb{R}^n$ into a sparse linear combination of $M \gg n$ learned directions: $\mathbf{f}(\mathbf{x}) = \sigma(\mathbf{W}_\mathrm{enc}\mathbf{x} + \mathbf{b}_\mathrm{enc})$, $\hat{\mathbf{x}} = \mathbf{W}_\mathrm{dec}\mathbf{f}$, with $\sigma$ chosen to enforce sparsity. The four production families: ReLU+L1 (Bricken 2023, Templeton 2024), Gated (Rajamanoharan 2024), TopK (Gao 2024), JumpReLU+$L_0$-via-STE (Rajamanoharan 2024), BatchTopK (Bussmann 2024). The $L_0$-vs-reconstruction Pareto frontier as the *only* axis to optimize during training. Documented failure modes: ReLU+L1 *shrinkage* (positive activations biased toward zero), TopK *dead-latent* problem (≤90% dead at large $M$ without AuxK), and reparameterization-non-invariance of L1 without unit-norm decoder columns. Gemma Scope (DeepMind 2024) as the open-release scale demonstration: >2000 SAEs covering >30M features across Gemma 2 2B/9B/27B at ~20% of GPT-3 pretraining compute.

- Anchors: `kb/notes/interpretability/sparse-autoencoders.md` §1-4; `kb/excerpts/{gao2024-topk-saes,rajamanoharan2024-jumprelu,gemma-scope-2024}.md` (PDF-grounded).
- Math to transcribe: encoder/decoder definition (Eq. 1); reconstruction + sparsity loss (Eq. 2); TopK activation (Eq. 3); JumpReLU activation $\mathrm{JumpReLU}_\theta(z) = z\, H(z - \theta)$ (Eq. 4); $L_0$ penalty via STE (Eq. 5).
- Comparison table: 5 SAE variants × {encoder activation, sparsity term, Pareto position, dead-latent risk, compute}, lifted from `sparse-autoencoders.md` §2.5.
- Application headline: Templeton 2024 "Scaling Monosemanticity" finds abstract / safety-relevant features (a "Golden Gate Bridge" feature, a "code-bug" feature, a "deception" feature usable for steering) on Claude 3 Sonnet. (HTML-only source; quotes pending Phase 2 manual transcription.)
- Superposition hypothesis (Elhage 2022 *Toy Models*): why $M \gg n$ is the right design.

### §8 — Are SAE features canonical? (4 pages)

The **decomposition uniqueness** question. Leask et al. 2025 (ICLR) *Sparse Autoencoders Do Not Find Canonical Units of Analysis* provides two pieces of evidence: (i) **meta-SAEs** decompose large-SAE latents (e.g., an "Einstein" latent) into combinations of more-elementary latents ("scientist", "Germany", "famous person") — *non-atomicity*; (ii) **SAE stitching** reveals novel latents present in larger SAEs that smaller SAEs cannot recover — *incompleteness*. This contradicts the Bricken/Templeton "true features" framing which assumes a canonical decomposition exists. The 2026 community position is largely pragmatic ("SAEs are useful, basis-dependent tools") but the theoretical status is open. **Transcoders** (Dunefsky 2024) are the structural cousin: they reproduce the *function* of an MLP sublayer (input-invariant) rather than its *activations* (input-dependent), and the cross-layer transcoder variant (Lindsey 2025) is the architectural primitive of circuit tracing (§10). We document where transcoder-as-MLP-replacement provides additional analytical leverage and where it inherits the same canonicality concern.

- Anchors: `kb/notes/interpretability/sparse-autoencoders.md` §6 frontier; `kb/excerpts/{leask2025-sae-not-canonical,dunefsky2024-transcoders}.md` (both PDF-grounded).
- Contradiction `[CONTRADICTION]` block (anchor section): atomicity vs non-atomicity, completeness vs incompleteness; what evidence would close it (e.g., a basis-finding criterion that two independent SAE training runs converge on).
- Math to transcribe: transcoder loss $\mathcal{L}_\mathrm{TC}(\mathbf{x}) = \|\mathrm{MLP}(\mathbf{x}) - \mathrm{TC}(\mathbf{x})\|_2^2 + \lambda_1 \|\mathbf{z}_\mathrm{TC}(\mathbf{x})\|_1$; the input-invariance factorization (encoder/decoder bilinear product as the input-invariant term).

### §9 — Activation patching: the causal-intervention spine (6 pages)

The canonical *causal* intervention method. The three-run protocol — **clean run** caches $\{\mathbf{h}_i^{(l)}\}$ on $X_\mathrm{cl}$; **corrupted run** records $m_*$ on $X_*$; **patched run** substitutes one component $C$'s clean activation into the corrupted run, records $m_\mathrm{pt}$. Indirect Effect $\mathrm{IE}_C = m_\mathrm{pt} - m_*$. Average over input pairs to obtain AIE. Iterating $C$ over every (component, token, layer) cell produces the **causal trace heatmap** ROME made famous. Three methodological degrees of freedom (Zhang & Nanda 2023): (i) corruption method — **Gaussian noising (GN)** vs **symmetric token replacement (STR)**; (ii) metric — probability vs **logit difference** vs KL; (iii) granularity — full hidden state vs MLP vs attention layer vs single head vs subspace. Heimersheim 2024 worked example as the methodological pedagogy. Path patching (Wang 2022) as the partial-derivative analogue isolating direct effects. Attribution patching (Nanda 2023, FORUM-SIGNAL) as the linearized $\mathrm{IE}_C \approx \langle \nabla_\mathbf{h} m, \Delta\mathbf{h} \rangle$ first-order approximation.

- Anchors: `kb/notes/interpretability/activation-patching.md` §1-3; `kb/excerpts/{meng2022-rome,zhang2023-apatching,heimersheim2024,wang2022-ioi}.md`.
- Math to transcribe: IE definition (Eq. 1); the three-run protocol with explicit caching/recording; logit-difference metric formula; attribution-patching first-order Taylor.
- Methodology pitfall taxonomy (lifted from `zhang2023-apatching §2.1` and `activation-patching.md §2`): GN throws activations off-distribution and can mis-localize relative to STR — `[CONTRADICTION]`. Probability metric saturates near 0/1 and hides negative components. Coarse-granularity patching gets you "MLPs at layer 15 matter" but not "the 'Paris' direction in those MLPs matter".
- Recommendations table: STR by default; logit-difference by default; head-level granularity for circuit claims; ablation as the validation step.

### §10 — ROME and direct edits: where causal-edit interpretability connects to capability surgery (3 pages)

Meng 2022 ROME is the bridge between causal interpretability and weight-level model surgery. Causal tracing on GPT-2 XL localizes factual recall to **mid-layer MLPs at the last subject token** (AIE peaks at layer 15; MLP contribution AIE = 6.6% vs attention AIE = 1.6%). The **Localized Factual Association Hypothesis**: midlayer MLPs act as key–value associative memories indexed by subject representations. ROME validates this by **rank-one MLP weight edits** that insert new $(s, r, o)$ associations with 99.8% efficacy and 88% paraphrase robustness on zsRE. The follow-up critique (Hase 2023 *Does Localization Inform Editing*): ROME-located layers are not always the best layers for factual editing — locating $\neq$ editing-target. We document where ROME's claim survives STR re-validation (the headline locus is robust) and where the per-layer AIE peaks shift (the precise layer-of-edit recommendation is sensitive to corruption choice).

- Anchors: `kb/notes/interpretability/activation-patching.md` §3, §4; `kb/excerpts/meng2022-rome.md#sec-2`, `#sec-2-2`, `#sec-2-3`, `#sec-3-2`, `#sec-2-effects`.
- Math to transcribe: the rank-one MLP edit objective $\Delta W = \mathrm{argmin}_{\Delta W} \|\Delta W \mathbf{k}_*\|^2$ s.t. $(W + \Delta W) \mathbf{k}_* = \mathbf{v}_*$ (closed-form rank-one update from Meng §3.2).
- Cross-paper thread `[Paper 2 §SFT, §preference-RL]`: model editing as an alternative to fine-tuning for surgical knowledge updates; relationship to LoRA/QLoRA adapter merging.
- Frontier note: ROME on frontier-2026 models (Llama 3/4, DeepSeek-V3) — open question whether mid-layer-MLP localization persists at MLA / MoE architectures.

### §11 — Circuits: from IOI to attribution graphs on Haiku (6 pages)

The most-cited end-to-end circuit reverse-engineering: **Wang 2022 IOI** on GPT-2 small. The model implements indirect-object identification with **26 attention heads in 7 functional classes** (Duplicate Token, Previous Token, S-Inhibition, Name Mover, Negative Name Mover, Backup Name Mover, Induction Heads) — 1.1% of all (head, token-position) pairs. Three findings that complicated the field's expectations: (i) **backup heads** — ablating Name Movers doesn't kill the behavior; latent backups activate; circuits are *redundant*; (ii) **repurposed mechanisms** — induction heads (originally identified by Olsson 2022 for in-context learning) are reused here for duplicate-token detection; mechanism $\neq$ function; (iii) **negative components** — Negative Name Mover heads write *against* the correct answer, possibly to hedge cross-entropy loss. **Conmy 2023 ACDC** automates the discovery: iteratively path-patch every edge of the computational graph, prune below threshold, return the surviving subgraph as the circuit. **Lindsey 2025 attribution graphs** is the Anthropic 2025 synthesis: replace MLPs with **cross-layer transcoders**, compute the Jacobian-attribution graph from input embeddings to target unembed direction, prune, validate by ablation. Reported findings on Claude 3.5 Haiku (HTML source, quotes pending): forward planning in poetry; multi-step factual chains ("Texas" → "Austin" for "the capital of the state Dallas is in"); refusal-circuit features; cross-lingual feature overlap.

- Anchors: `kb/notes/interpretability/{mechanistic-interpretability,circuit-tracing}.md`; `kb/excerpts/{wang2022-ioi,conmy2023-acdc}.md` (PDF-grounded); Lindsey 2025 (HTML, FORUM-SIGNAL until Phase 2 transcription).
- Math/algorithm to transcribe: Wang's circuit-validity criteria (faithfulness/completeness/minimality) as Eq. block; the ACDC algorithm pseudocode (initialize graph; for each edge in topological order, patch, measure delta, prune below threshold); attribution-graph Jacobian factorization $\partial f_j^{(l_2, t_2)} / \partial f_i^{(l_1, t_1)}$ as bilinear product of decoder column and encoder row.
- Comparison table: activation-patching vs circuit-tracing × {granularity, causal-vs-correlational, cost per task, setup cost, output, validation, standardization-pitfalls} — lifted from `circuit-tracing.md` §5.
- Cost: circuit-tracing setup is ~20% of pretraining compute (transcoder training); per-task cost is 1 fwd + 1 bwd, vs $O(N_\mathrm{components})$ fwd-passes for full activation patching.

### §12 — Cross-method convergence and divergence (4 pages)

The thesis-anchor section. We list the cross-method evidence on canonical behaviors:

**Convergence cases:**
- **IOI circuit (GPT-2 small).** Activation patching (Wang 2022) and ACDC (Conmy 2023) recover overlapping circuit subgraphs; SAE-feature analysis on the same prompts identifies features matching the QK/OV roles of the discovered heads; tuned-lens prediction trajectories on IOI prompts are consistent with the circuit's mid-late layer commit. Three methods, one converging picture.
- **Factual recall localization (GPT-2 XL).** ROME causal tracing localizes mid-layer MLPs at last-subject token; SAE features on those MLPs (Templeton 2024 partial replication) include subject-entity features; tuned lens shows the predicted-object token rises in probability across exactly the localized layer band; rank-one ROME edits insert facts with 99.8% efficacy. Four methods, one converging picture.

**Divergence cases:**
- **Truth direction across datasets (LLaMA-2-13B).** Mass-mean probing (Marks–Tegmark 2023) finds truth directions that are antipodal in early layers between `cities` and `neg_cities`, then orthogonal, then aligned only in late layers; CCS finds a different "truth" direction that fails under negation; SAE-discovered "truthfulness" features (Templeton 2024) overlap partially but not fully with mass-mean directions. Methods disagree on whether there is a *single* truth direction at all.
- **SAE feature canonicality.** Bricken/Templeton "true features" framing vs Leask 2025 "non-atomic, incomplete" critique. Two independent SAE training runs at different $M$ may produce different feature inventories on the same activations.
- **Lens cross-model unreliability.** Logit lens works on GPT-2 but fails on BLOOM/GPT-Neo; tuned lens fixes it but adds parameters; whether DoLa contrastive decoding generalizes from LLaMA-7B/13B/33B to Llama 3/4 / Qwen 2.5/3 / DeepSeek-V3 has not been systematically reproduced.

- Anchors: `kb/index/contradictions.md` § interpretability; `kb/notes/interpretability/{lens-techniques,probing,sparse-autoencoders,activation-patching,circuit-tracing,mechanistic-interpretability}.md` frontier sections.
- Closing argument: when methods *should* agree (the underlying mechanism is the same), divergence is a falsification signal for at least one method's assumptions. When methods *answer different questions* (correlational vs causal; component vs feature), apparent divergence may be principled.

### §13 — Scaling interpretability and frontier-2026 snapshot (5 pages)

The compute / scale dimension. From ~125M GPT-2 small (the IOI substrate) to 27B Gemma 2 (Gemma Scope SAEs) to production Claude 3.5 Haiku (Lindsey 2025 attribution graphs). What scales:

- **SAEs scale.** Gao 2024 reports clean $L(C)$ and $L(N, K)$ scaling laws; 16M-latent SAE on GPT-4 is feasible. Gemma Scope releases >2000 SAEs across 2B/9B/27B at ~20% of GPT-3 pretraining.
- **Attribution graphs scale to production but expensively.** Anthropic's circuit-tracing on Haiku is the existence proof; comparable compute cost per model; not yet reproduced outside Anthropic.
- **Probes scale gracefully.** Linear probes are cheap relative to model forward passes; mass-mean is closed-form. Probe-derived steering directions are an inference-time cheap intervention.

What doesn't scale (yet):

- **Manual circuit hunts.** The IOI-style hand reverse-engineering does not scale past ~100M-parameter models with reasonable researcher-time. ACDC scales the discovery but produces larger, less interpretable circuits.
- **Full activation patching.** $O(N_\mathrm{components})$ forward passes per (clean, corrupted) pair becomes prohibitive on production-scale models; attribution patching linearizes but loses accuracy in non-linear regions.
- **Lens coverage of frontier-2026 architectures.** Tuned lenses for Llama 3/4, DeepSeek-V3 (with MLA + MoE), Gemma 3, Qwen 2.5/3 are not all published; whether the lens framework extends cleanly to MLA-compressed-KV / fine-grained-MoE / NSA-sparse-attention architectures is partially empirical.

What's running in production interp pipelines as of 2026: Anthropic circuit tracer (open-sourced 2025-06, integrated with Neuronpedia); OpenAI SAE evals (Gao 2024 lineage); DeepMind tuned-lens benchmarks across Gemma family. Snapshot table: which method, which lab, which models, which open release date.

- Anchors: `kb/index/timeline.md` 2024–2026 entries; `kb/notes/interpretability/{circuit-tracing,sparse-autoencoders}.md` frontier sections; `kb/excerpts/gemma-scope-2024.md#sec-1`.
- Output: a 1-page snapshot table of {method, paper-key, target-model, scale, open-release-status, 2026 reproduction status} that becomes the figure-of-record for the paper.
- Cross-paper thread `[Paper 1 §13 frontier-2026 snapshot]`: the architectural-convergence picture there is the substrate for the interpretability-coverage gap surfaced here.

### §14 — Contradictions live and open frontiers (3 pages)

Concentrated discussion of the `[CONTRADICTION]` markers. Five anchor contradictions:

1. **SAE canonical-feature claim.** Bricken/Templeton "true features" vs Leask 2025 "non-atomic + incomplete". What evidence would close it: a basis-finding criterion under which two independent SAE training runs converge on the same feature inventory, modulo a known equivalence (rotation, scaling).
2. **Lens cross-model unreliability.** Logit lens works on GPT-2, fails on BLOOM/GPT-Neo, fixed on those by tuned lens. Whether tuned-lens framework extends cleanly to MLA / MoE / NSA / sliding-attention architectures is empirical and not yet documented at frontier-2026 scale.
3. **Probe correlation-vs-causation.** Probe accuracy alone does not establish use; direction-ablation closes part of the gap; whether probe-derived steering generalizes from ID evaluation to OOD adversarial inputs is the load-bearing alignment question.
4. **Truth direction across datasets.** Marks–Tegmark §4 misalignment; Vennemeyer 2025 sycophancy multi-component finding generalizes the pattern. Whether "truth" is a single linear axis or a multi-axis subspace is unresolved.
5. **Circuit-tracing scalability past Anthropic-2025.** Reported Haiku findings have not been independently reproduced as of 2026; transcoder canonicality inherits the Leask 2025 critique; whether attribution-graph conclusions are stable across transcoder training seeds is open empirical.

For each: the strongest-form claim of each side, the evidence on the table, and the experiment that would resolve it. Close with the paper's thesis restated against this surface: method-pluralism is principled because the contradictions live at the boundaries between methods, and resolving them requires *cross-method* evidence — not better single-method tools.

- Anchors: `kb/index/contradictions.md` § interpretability (the 6 cataloged contradictions, with cross-references to method-note frontier sections).

## What this outline commits to

- **Math is verbatim from PDFs.** Every equation cited with a `kb/excerpts/<key>#<anchor>` dual-citation. Equations to transcribe include: residual update + unembed-from-layer-$\ell$ identity (§2); LogitLens / TunedLens / DoLa / tuned-lens distillation loss (§3); linear-probe loss + structural-probe distance + mass-mean direction (§4); orthogonal-projector ablation + steering rule (§5); SAE encoder/decoder + reconstruction+sparsity loss + TopK + JumpReLU + $L_0$-via-STE (§7); transcoder loss + input-invariance factorization (§8); IE definition + logit-difference metric + attribution-patching Taylor (§9); ROME rank-one closed-form edit (§10); Wang 2022 circuit criteria + ACDC pseudocode + attribution-graph Jacobian factorization (§11).
- **Tensor shapes everywhere relevant.** $B, S, D, H, d_h, V, L$ from Paper 1; plus probe / SAE notation $\mathbf{w}, \mathbf{B}, \mathbf{d}_y, \mathbf{f}, \mathbf{W}_\mathrm{enc}, \mathbf{W}_\mathrm{dec}, M, L_0$.
- **Contradictions are first-class.** Each `[CONTRADICTION]` block surfaced in §3, §6, §8, §9, §10, §11 → consolidated in §14, not buried in body.
- **Method-pluralism is the throughline.** Every section ends by naming which question that method *does not* answer, pointing the reader to the section that picks up the slack.
- **No analogy laundering.** `[INTUITION]` and `[ANALOGY]` markers preserved from KB notes; analogies always return to canonical math (lens-as-early-exit-decoding, probe-as-psychophysics, SAE-as-dictionary, transcoder-as-sparse-Taylor-expansion, patching-as-counterfactual-surgery, circuit-tracing-as-city-route-graph).
- **Forum-signal claims are flagged.** nostalgebraist 2020 (LessWrong logit lens), Nanda 2023 (attribution-patching blog), Lindsey 2025 (transformer-circuits.pub HTML for Haiku findings) all carry FORUM-SIGNAL or HTML-pending markers; never cited as the sole backing for a hard technical claim.
- **Cross-paper threads are explicit.** `[Paper 1 §residual-stream]`, `[Paper 2 §pre-training-output]`, `[Paper 3 §11 CoT-faithfulness]`, `[Paper 5 §alignment-detection]` flagged at the source rather than relegated to a final cross-reference appendix.

## Sections by writing-pass parallelism

- Pass A (independent foundations, easy parallelize): §3 (lens), §4 (probing), §7 (SAEs), §9 (activation patching). Each is a self-contained method exposition grounded in its own KB note.
- Pass B (synthesis, depends on Pass A landing): §5 (correlation–causation gap, threads §4 and §9), §6 (truth directions, threads §4 and §5), §8 (SAE canonicality, threads §7 with the transcoder bridge to §11), §10 (ROME, threads §9), §11 (circuits, threads §7 + §9 + transcoders from §8).
- Pass C (whole-paper synthesis, last): §1 introduction, §2 substrate-and-notation, §12 cross-method convergence/divergence, §13 scaling-and-snapshot, §14 contradictions.
- Pass D: §2 substrate-and-notation can technically run alongside Pass A but is best deferred to Pass C so that notation is fixed against the §3-§11 method exposition rather than ahead of it.

4 parallel section-subagents on Pass A; 5 sequenced on Pass B (§5 and §6 chain; §8, §10, §11 chain); 5 on Pass C after Pass A+B lands. Total: ~14 section-subagent invocations to assemble the full paper draft.
