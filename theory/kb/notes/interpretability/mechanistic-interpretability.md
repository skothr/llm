---
topic: interpretability/mechanistic-interpretability
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - wang2022-ioi
  - meng2022-rome
  - olsson2022-induction-heads     # Tier A but HTML-only at transformer-circuits.pub
secondary_sources:
  - templeton2024-scaling-monosemanticity
  - lindsey2025-circuit-tracing
  - leask2025-sae-not-canonical
related_topics:
  - interpretability/sparse-autoencoders
  - interpretability/activation-patching
  - interpretability/circuit-tracing
  - interpretability/probing
  - interpretability/lens-techniques
---

# Mechanistic interpretability

**Mechanistic interpretability** (MI) is the research program of
reverse-engineering neural networks into human-readable algorithms — at
the level of *individual circuits* (sub-graphs of components) and
*features* (directions in activation space), not just behavioral
metrics. Where probing tells you "the model represents X", MI asks
"by what algorithm, implemented in which weights and activations, does
the model use X to do Y". Activation patching, sparse autoencoders, and
circuit tracing are the three load-bearing methodologies.

## 1. Operational definition (working framework)

A **circuit** in MI usage is a subgraph $C$ of the model's computational
graph $M$ — nodes are forward-pass terms (residual stream, attention
heads, MLPs, embeddings) and edges are the residual / attention /
projection links between them — that is *responsible for* a specific
behavior, in the sense that the function $C(x)$ defined by knocking
out all nodes in $M \setminus C$ approximates $M(x)$ for inputs in the
behavior's distribution
`[wang2022-ioi §2; kb/excerpts/wang2022-ioi#sec-2-definition]`.

Three quantitative criteria for circuit validity, due to Wang et al.
2022 (IOI) and now standard
`[wang2022-ioi §4; kb/excerpts/wang2022-ioi#sec-4-criteria]`:

| Criterion | Statement | How to test |
|---|---|---|
| **Faithfulness** | $C$ performs the task as well as the whole model. | Forward-pass agreement / metric matching on task distribution. |
| **Completeness** | $C$ contains every node used to perform the task. | Knocking out $M \setminus C$ should not change behavior beyond noise; if it does, $C$ is missing nodes. |
| **Minimality** | $C$ does not contain task-irrelevant nodes. | Each node in $C$ should fail an "ablate-it-and-see-if-it-matters" test. |

Real circuits often pass faithfulness and minimality but fail the
strongest completeness tests
`[wang2022-ioi §4 quote; kb/excerpts/wang2022-ioi#sec-4-criteria]`.

A **feature** in MI usage is a direction $\mathbf{d} \in \mathbb{R}^n$
in some activation space such that the projection
$\mathbf{x}^\top \mathbf{d}$ (or equivalently, the activation of an SAE
latent with decoder column $\mathbf{d}$) corresponds to a
human-interpretable property of the input. Features are the *content*
flowing along the *channels* (circuits). The features-vs-circuits
distinction is fundamental: a circuit can route a feature from one
position to another without "computing" the feature
`[wang2022-ioi §2.1; kb/excerpts/wang2022-ioi#sec-2-definition]`.

## 2. Methodology — the three load-bearing techniques

### 2.1 Activation patching

The causal-intervention spine of MI: identify which components matter
by literally swapping their activations and measuring the output
change. Discovers circuits at component (head, MLP) granularity. See
`kb/notes/interpretability/activation-patching.md` for the full protocol;
the canonical paper is Meng et al. 2022 (ROME)
`[meng2022-rome §2; kb/excerpts/meng2022-rome#sec-2]`.

### 2.2 Sparse autoencoders / transcoders

Decompose activations into sparse linear combinations of learned
**features**, lifting the polysemanticity of the neuron basis. SAEs
solve the "which direction is the 'capital city' feature" question
that activation patching cannot answer on its own. See
`kb/notes/interpretability/sparse-autoencoders.md`. Templeton 2024
(Scaling Monosemanticity) and Lieberum 2024 (Gemma Scope) are the
production-scale references.

### 2.3 Circuit tracing (attribution graphs)

Anthropic's 2025 framework (Lindsey et al.) replaces MLPs with
**cross-layer transcoders** and uses the resulting feature graph plus
Jacobian attribution to build end-to-end attribution graphs on
production Claude 3.5 Haiku. This is the synthesis of (2.1) and (2.2):
features as nodes, attribution scores as edges, automated. See
`kb/notes/interpretability/circuit-tracing.md`.

## 3. Landmark findings (chronological)

### 3.1 The Mathematical Framework + Induction Heads (Elhage 2021, Olsson 2022)

The **Mathematical Framework for Transformer Circuits** (Elhage et al.,
Anthropic 2021) introduced the operational vocabulary the field still
uses: **residual stream** as a linear data bus that every layer reads
from and writes to; attention heads decomposed into **QK** (read
pattern) and **OV** (read content + write content) circuits;
**virtual attention heads** as compositions of attention across layers.

**In-context Learning and Induction Heads** (Olsson et al., Anthropic
2022) identifies a specific 2-attention-head pattern — a Previous Token
Head followed by an Induction Head that completes
`[A][B] ... [A] → [B]` — as the mechanism behind in-context learning.
Six independent lines of evidence (phase-change in loss curve aligned
with phase-change in induction-head formation; ablation of induction
heads removes ICL; etc.) make this one of the strongest single
circuit-level claims in the field. **Stub note: full excerpts pending —
HTML-only at transformer-circuits.pub, not retrievable in this
sandbox; Phase 2 follow-up should manually transcribe key passages.**

### 3.2 ROME / factual recall (Meng 2022)

`[meng2022-rome §2.3, §3; kb/excerpts/meng2022-rome#sec-2-3]`. Causal
tracing on GPT-2 XL localizes factual recall to mid-layer MLPs at the
last subject token (AIE peaks at layer 15, MLP contribution = 6.6%
vs. attention 1.6%). The **Localized Factual Association Hypothesis**:
midlayer MLPs act as key-value associative memories indexed by subject
representations. Validated by ROME (rank-one weight edit) which
inserts new $(s, r, o)$ associations with 99.8% efficacy, 88% paraphrase
robustness on zsRE
`[meng2022-rome §3.2; kb/excerpts/meng2022-rome#sec-3-2]`.

### 3.3 IOI circuit (Wang 2022)

The most-cited end-to-end circuit reverse-engineering. 26 GPT-2-small
attention heads in 7 functional classes implementing the algorithm
"identify all previous names → remove duplicates → output the
remaining name"
`[wang2022-ioi §3; kb/excerpts/wang2022-ioi#sec-3-circuit]`. Three
unexpected findings that complicated the field's expectations:

1. **Backup heads.** Ablating Name Mover heads doesn't kill the
   behavior; instead, latent "Backup Name Mover" heads activate and
   recover most performance. Circuits are *redundant*.
2. **Repurposed mechanisms.** Induction heads (a known structure from
   Olsson 2022) are used here for a different purpose (duplicate-token
   detection). Mechanism ≠ function.
3. **Negative components.** Negative Name Mover heads write *against*
   the correct answer, possibly to hedge cross-entropy loss
   `[wang2022-ioi §3 insights; kb/excerpts/wang2022-ioi#sec-3-insights]`.

### 3.4 Scaling Monosemanticity (Templeton 2024)

SAEs at production scale on Claude 3 Sonnet: a 34M-latent SAE
discovers abstract features (a "deception" feature, a "code-bug"
feature, a "Golden Gate Bridge" feature usable for steering). Empirical
demonstration that SAEs work on frontier models, not just toy
transformers. **Stub note: full excerpts pending — HTML-only at
transformer-circuits.pub.**

### 3.5 Circuit tracing on Haiku (Lindsey 2025)

Anthropic's 2025 attribution-graph framework reads end-to-end circuits
out of Claude 3.5 Haiku, finding mechanisms like:
- **Forward planning in poetry:** the model selects rhyme target
  features many tokens before producing the rhyme.
- **Multi-hop reasoning chains:** a "Texas" feature → "Austin" feature
  for "the capital of the state Dallas is in".
- **Refusal circuits:** safety-relevant features fire on harmful
  request classes.

`kb/notes/interpretability/circuit-tracing.md` covers the methodology.

## 4. The features-vs-circuits dichotomy

[INTUITION] An MI explanation has two layers:

1. **What the model represents** — its features. These are directions
   in activation space; SAEs find them.
2. **How those features flow through the model** — its circuits.
   Activation patching and circuit tracing find these.

A complete MI explanation of a behavior requires both. ROME's
"factual recall happens in mid-layer MLPs" is half an explanation: it
identifies the circuit (MLP layer 15) but not the feature (which
direction in the residual stream encodes "Paris"). Templeton 2024's
discovery of a "Golden Gate Bridge" feature is the other half: it
identifies a feature but not the circuit that generates it.

[ANALOGY] [SPECULATION-adjacent] If you reverse-engineer a CPU, you
need both the **wiring diagram** (which gates connect to which) and
the **encoding** (which voltage means logical "1"). MI's circuits are
the wiring diagram; MI's features are the encoding. The Mathematical
Framework's "residual stream" is the bus on the diagram; SAE latents
are the labels for the symbols on the bus.

## 5. Frontier and open questions (as of 2026-05)

- **Are SAE features canonical?** [CONTRADICTION] Leask et al. 2025
  argue not
  `[leask2025-sae-not-canonical §1; kb/excerpts/leask2025-sae-not-canonical#sec-1-canonical]`.
  If features are basis-dependent, the entire MI program needs a
  re-think on what "the" mechanism for a behavior even means.
  See `kb/notes/interpretability/sparse-autoencoders.md` §6.
- **Circuits at production scale.** Activation patching scales poorly
  (manual; one task at a time; sensitive to corruption choice). Circuit
  tracing scales but produces large (~1000-feature) attribution graphs
  that are hard to read in full. The community has not yet
  converged on a presentation format for production-scale circuits.
- **Generalization of circuits.** "Adaptive Circuit Behavior and
  Generalization in Mechanistic Interpretability" (ICLR 2024) shows
  the IOI circuit generalizes more flexibly than expected: variants
  of the IOI prompt activate slightly different head configurations,
  suggesting circuits are a soft-clustering of model behavior rather
  than rigid modules.
- **Mech-interp vs. alignment.** MI is increasingly load-bearing for
  alignment claims (e.g., "the model has a deception feature");
  whether discovered features generalize from in-distribution to
  out-of-distribution adversarial inputs is contested. The Anthropic
  alignment-faking work
  (`kb/notes/alignment/scheming-and-deceptive-alignment.md`) uses MI
  evidence; whether this is sufficient is an open methodological
  debate.
- **Theoretical foundations.** No theory yet predicts which circuits a
  given training run will produce. The Mathematical Framework + the
  superposition hypothesis explain *capacity* (why SAEs work in
  principle); they do not explain *mechanism* (why the IOI circuit
  takes the specific 7-class shape it does).

## 6. See also

- `kb/notes/interpretability/sparse-autoencoders.md` — the dominant
  feature-extraction method; necessary substrate for circuit tracing.
- `kb/notes/interpretability/activation-patching.md` — the causal
  intervention method; substrate for circuit discovery up to 2024.
- `kb/notes/interpretability/circuit-tracing.md` — the 2025 synthesis
  of features + activations + attribution.
- `kb/notes/interpretability/probing.md` — the correlational
  counterpart; probes find features, MI uses features.
- `kb/notes/interpretability/lens-techniques.md` — logit/tuned lens;
  layer-wise readout of the residual stream.
- `kb/notes/architecture/attention-mechanism.md` §4 — what individual
  heads can do; induction heads, copy heads, IOI mechanisms.
