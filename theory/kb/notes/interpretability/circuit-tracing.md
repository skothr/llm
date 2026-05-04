---
topic: interpretability/circuit-tracing
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - dunefsky2024-transcoders
  - lindsey2025-circuit-tracing  # HTML-only at transformer-circuits.pub
secondary_sources:
  - lindsey2025-biology          # HTML-only follow-up: findings on Haiku
  - templeton2024-scaling-monosemanticity
  - wang2022-ioi
related_topics:
  - interpretability/mechanistic-interpretability
  - interpretability/sparse-autoencoders
  - interpretability/activation-patching
---

# Circuit tracing (attribution graphs)

**Circuit tracing** is the 2025 Anthropic methodology for automated,
end-to-end mechanistic interpretability of production-scale language
models. It is the synthesis of three previously separate threads:
sparse-autoencoder feature decomposition, transcoder MLP replacement,
and Jacobian-based attribution. The output is an **attribution graph**:
a sparse subgraph of features (transcoder latents) connected by
edges weighted by their causal contribution, computed in one or two
forward passes — no per-task corruption design, no manual head
labeling. Operationally distinct from `activation-patching`, which
searches the activation graph one component at a time.

## 1. Operational definition

An **attribution graph** for a behavior on input $x$ has:

- **Nodes:** transcoder feature activations $\{f_i^{(l)}\}$ (one per
  feature, per layer, per token position) — the same inventory an SAE
  would produce, except the underlying decomposition is a transcoder
  (`kb/notes/interpretability/sparse-autoencoders.md` §3) rather than
  an SAE-on-activations.
- **Edges:** linear-attribution scores between feature activations
  across layers / positions, computed via the Jacobian of the
  cross-layer transcoder system. The edge $f_i^{(l_1)} \to f_j^{(l_2)}$
  is weighted by how much a perturbation of $f_i^{(l_1)}$ would change
  $f_j^{(l_2)}$, holding the rest of the graph fixed.
- **Sparsity:** because transcoder features fire on a small fraction
  of tokens (typical $L_0 \sim 50$–$200$ for a $\sim 100\text{k}$-feature
  transcoder), the graph is sparse by construction. A typical
  attribution graph for one prompt has $O(10^2$–$10^3)$ active
  features and a comparable number of edges after pruning.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $f_i^{(l, t)}$ | activation of transcoder feature $i$ at layer $l$, token $t$ |
| $\mathbf{W}_{\text{enc}}^{(l)}, \mathbf{W}_{\text{dec}}^{(l)}$ | encoder/decoder of the layer-$l$ transcoder |
| **Cross-layer transcoder** | a transcoder whose decoder writes into *all subsequent* layers' residual streams, not just the immediate output (the architectural innovation of Lindsey 2025) |
| **Attribution edge** | $\partial f_j^{(l_2, t_2)} / \partial f_i^{(l_1, t_1)}$ at fixed input — the linearized causal contribution |

## 2. Mechanism — replace, then trace

### 2.1 Replace MLPs with cross-layer transcoders

Train a transcoder for each layer's MLP — but the **cross-layer**
variant: the decoder writes into the residual stream of every
subsequent layer, not just the immediate next one. This makes a
feature's "skip-layer effects" explicit in the graph rather than
hidden inside the residual stream. The transcoder is trained to
match the original model's MLP output (faithfulness) plus a sparsity
penalty
`[dunefsky2024-transcoders §3.1 Eq.5; kb/excerpts/dunefsky2024-transcoders#sec-3-1]`,
extended in Lindsey 2025 to a multi-layer / cross-layer variant.

Once trained, the transcoders **replace** the MLPs at inference time.
The replaced model is a near-identical functional approximation of the
original (faithfulness loss is small), so any computation on the
replaced model is approximately a computation on the original.

### 2.2 Compute the Jacobian attribution graph

For a target behavior on input $x$ (e.g., "produce token $r$ at
position $T$"):

1. Forward-pass the replaced model on $x$. Record all transcoder
   feature activations $\{f_i^{(l, t)}\}$.
2. Choose a target node — typically the unembedding direction for
   the target token at position $T$.
3. Compute the linearized contribution of every active feature to
   the target via the Jacobian. Because the transcoders are linear
   reads/writes plus a sparse non-linearity, this Jacobian is sparse
   and analytically tractable: the attribution from $f_i^{(l_1, t_1)}$
   to $f_j^{(l_2, t_2)}$ factors into the (input-invariant) bilinear
   product of decoder column $\mathbf{d}_i^{(l_1)}$ with encoder row
   $\mathbf{e}_j^{(l_2)}$, modulated by attention weights for any
   intervening attention layers.
4. Prune edges below a threshold; iterate from the target node
   backward through the graph; stop at the embedding layer.

The result is a directed graph from input embeddings to output
unembedding, threaded through the transcoder feature space.

### 2.3 Validate by ablation

Activation patching is *not* obsoleted; it is now used to validate the
attribution graph. Pick a feature node, ablate it (set $f_i^{(l, t)} =
0$ in the replaced model), measure the effect on the target. The
attribution graph predicts which ablations should matter and by how
much; deviations flag spurious edges.

## 3. Reported applications (what the graphs reveal on Haiku)

Lindsey et al. 2025 ("On the Biology of a Large Language Model")
applies circuit tracing to Claude 3.5 Haiku. Headline findings (HTML
source; verbatim quotes pending Phase 2 follow-up — transferred from
Phase 1 landscape report):

- **Forward planning in poetry.** When the model writes a couplet, a
  feature representing the rhyme target activates *many tokens before*
  the rhyme word is produced, then drives the intervening words to
  set up the rhyme. This is operationally what humans call planning.
- **Multi-step factual chains.** "The capital of the state Dallas is
  in" → activates "Texas" features in middle layers, then "Austin"
  features later, then writes "Austin" to logits. The two-step chain
  is visible as a two-edge path in the attribution graph.
- **Safety / refusal circuits.** Specific feature classes (associated
  with harmful content categories) activate on harmful requests and
  drive refusal circuits. Cross-layer transcoders make these legible
  in a way SAE-only analyses did not.
- **Computation in non-English contexts.** Feature activations on a
  Spanish prompt show partial overlap with the corresponding English
  features but are not identical; the graph reveals where the model
  routes computation through a "language-agnostic" middle and where it
  routes through language-specific paths.

[FORUM-SIGNAL] These claims rely on Anthropic's HTML report at
`transformer-circuits.pub/2025/attribution-graphs/biology.html`; this
sandbox cannot fetch transformer-circuits.pub, so the verbatim
excerpts are not yet in the KB. **Phase 2 follow-up (manual transcription
or off-sandbox fetch) required.**

## 4. Variants and lineage

- **2022 (Mathematical Framework + Olsson induction heads)** —
  established the residual-stream + QK/OV-decomposition vocabulary
  that makes circuit tracing legible.
- **2022 (Wang IOI, Meng ROME)** — manual circuit reverse-engineering
  via activation patching; expensive, per-task, but established the
  faithfulness/completeness/minimality criteria
  `[wang2022-ioi §4; kb/excerpts/wang2022-ioi#sec-4-criteria]`.
- **2023 (Bricken Towards Monosemanticity)** — SAE-based feature
  decomposition; provides the first half (features) of what circuit
  tracing needs.
- **2024-06 (Dunefsky Transcoders, NeurIPS)** — input-invariant
  factorization of MLPs into sparse feature circuits via transcoders,
  with weight-based circuit analysis demonstrated up to 1.4B params
  `[dunefsky2024-transcoders §1; kb/excerpts/dunefsky2024-transcoders#sec-1-headline]`.
  This is the architectural primitive.
- **2025-03 (Lindsey Circuit Tracing, transformer-circuits.pub)** —
  cross-layer transcoders + Jacobian attribution + automated graph
  pruning, demonstrated on production Claude 3.5 Haiku.
- **2025-06 (Anthropic open release)** — circuit-tracing tooling
  open-sourced and integrated with Neuronpedia. Tier B / C signal
  from the Phase 1 sweep; not yet excerpted.

## 5. Comparison vs. activation patching

| Property | Activation patching | Circuit tracing |
|---|---|---|
| **Granularity** | components (heads, MLPs) | features (transcoder latents) |
| **Causal vs. correlational** | causal (intervene + measure) | causal-by-Jacobian (linearized intervention) |
| **Cost per task** | $O(N_\text{components})$ forward passes per (clean, corrupted) pair | 1 forward + 1 backward pass on the replaced model |
| **Setup cost** | none (use the original model) | train transcoders for every layer (one-time, ~20% of pretraining compute) |
| **Output** | localizations (which components matter) | attribution graphs (which features cause which features) |
| **Validation** | self-validating (the metric *is* the patching effect) | requires ablation to validate edges |
| **Standardization** | corruption method (GN vs. STR) and metric (probability vs. logit-diff) materially affect results `[zhang2023-apatching §1 findings]` | requires consistent transcoder training; sensitive to feature-splitting and SAE-canonicality concerns `[leask2025-sae-not-canonical]` |

## 6. Intuitions and analogies

[ANALOGY] If a model is a city, **activation patching** is "block off
this street and see what happens to traffic" — empirical, slow,
interpretable, but only one street at a time. **Circuit tracing** is
"give me the directed graph of every commuter route, with each route
labeled by how many people use it daily" — preprocessed (the
transcoders are the city's public-transit timetable), much cheaper to
query, but only as accurate as the timetable. The analogy returns to
canonical form: the patching effect is the IE
`[meng2022-rome §2 effects; kb/excerpts/meng2022-rome#sec-2-effects]`,
the attribution-graph edge is the Jacobian; both are first-order causal
contributions to a target metric.

[INTUITION] Cross-layer transcoders make **information flow through the
residual stream legible**. A vanilla transcoder factors a single MLP;
a cross-layer transcoder factors the full residual contribution of
that MLP to all downstream layers. This matters because in a real
LLM, mid-layer MLPs frequently "write" features that are not used
until 5–10 layers later (e.g., the rhyme-planning feature in poetry).
A per-layer transcoder + per-layer SAE would not connect these dots
without an explicit cross-layer attribution computation.

[INTUITION] Why this is "biology". Lindsey et al.'s framing
(`On the Biology of a Large Language Model`) is that circuit tracing
turns LLM mechanism into an **observable** rather than a
**hypothesis**: previously a researcher hypothesized "maybe the model
plans rhymes" and designed a patching experiment to test it; now the
researcher reads off the attribution graph, sees the feature
activating early, and *then* tests the hypothesis on harder cases.
The methodology shifts from confirmatory to exploratory.

## 7. Frontier and open questions (as of 2026-05)

- **Transcoder canonicality.** [CONTRADICTION] If SAEs are not
  canonical (Leask 2025), are transcoders? The ICLR 2025 critique
  applies in principle: a different transcoder training run may
  factor the same MLP into a different sparse basis. Lindsey et al.
  acknowledge this; their pragmatic position is that a *useful basis*
  is sufficient, not a *canonical* one. Whether circuit-tracing
  conclusions are stable across transcoder training seeds is an
  open empirical question.
- **Compute cost.** Training transcoders for every layer of a
  production model is roughly comparable to SAE training cost
  (~20% of pretraining compute on Gemma Scope; assume similar on
  Claude). This is a real adoption barrier for non-frontier labs.
- **Are attribution graphs faithful?** The Jacobian approximation is
  exact for linear systems; transcoders' sparse non-linearities make
  it locally linear. For computations that depend strongly on
  feature interactions outside the local linearization, attribution
  graphs may miss the relevant mechanism. Validation via ablation is
  necessary.
- **Coverage.** Anthropic's reported findings are on Haiku; whether
  the same techniques scale to larger models (Sonnet, Opus) and
  whether the discovered circuits generalize across model families
  has not been independently reproduced as of 2026.
- **Mech-interp vs. alignment.** Circuit tracing is increasingly
  cited as evidence in alignment claims (e.g., "we have visibility
  into the model's deception circuits"). The strength of these
  claims depends on circuit-tracing fidelity, which depends on
  transcoder canonicality, which is unresolved.

## 8. See also

- `kb/notes/interpretability/sparse-autoencoders.md` §3 — transcoders
  as the structural cousin of SAEs.
- `kb/notes/interpretability/activation-patching.md` — the predecessor
  methodology; circuit tracing extends but does not replace it.
- `kb/notes/interpretability/mechanistic-interpretability.md` — the
  research program that circuit tracing serves.
- `kb/notes/architecture/ffn.md` — the MLP sublayer being replaced
  by transcoders.
- `kb/notes/alignment/scheming-and-deceptive-alignment.md` — the
  alignment context where circuit-tracing evidence is increasingly
  cited.
