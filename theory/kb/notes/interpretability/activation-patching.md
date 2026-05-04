---
topic: interpretability/activation-patching
status: draft
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources:
  - meng2022-rome
  - zhang2023-apatching
  - wang2022-ioi
  - heimersheim2024-apatching-howto
secondary_sources:
  - conmy2023-acdc
  - vig2020              # Tier A but not yet downloaded
  - nanda2023-attribution-patching  # Tier B (blog) — discovery only
related_topics:
  - interpretability/mechanistic-interpretability
  - interpretability/circuit-tracing
  - interpretability/sparse-autoencoders
---

# Activation patching

**Activation patching** is the canonical *causal* intervention method
in mechanistic interpretability: it identifies which internal model
components matter for a behavior by literally swapping their activations
between forward passes and measuring the effect on the output. Unlike
probing or SAE feature activations (which are correlational), activation
patching tells you what the model *uses*, not just what it
*represents*. It is the method behind the ROME factual-recall
localization, the IOI-circuit discovery, and most modern circuit
analyses up through 2024.

## 1. Formal definition

The **patching effect** of a component $C$ at position-token-layer
$(i, l)$ on a target metric $m$, given a clean prompt $X_\text{cl}$
and a corrupted prompt $X_*$ that yield different model behaviors, is
defined by the three-run protocol
`[meng2022-rome §2; kb/excerpts/meng2022-rome#sec-2]`,
`[zhang2023-apatching §2.1; kb/excerpts/zhang2023-apatching#sec-2-1]`:

1. **Clean run.** Run the model on $X_\text{cl}$. Cache all internal
   activations $\{h_i^{(l)}\}_{i, l}$.
2. **Corrupted run.** Run the model on $X_*$. Record output metric
   $m_*$ (e.g., probability of the clean answer).
3. **Patched run.** Run the model on $X_*$ but at component $C$,
   replace its activation with the cached clean value $h_i^{(l)}$
   from step 1. Record output metric $m_\text{pt}$.

The **indirect effect** (IE) of component $C$ is

$$\text{IE}_C = m_\text{pt} - m_* \tag{1}$$

`[meng2022-rome §2 effects; kb/excerpts/meng2022-rome#sec-2-effects]`.
A large positive IE means restoring this component's clean value
recovers the behavior — i.e., $C$ is causally important. Average over a
sample of input pairs to obtain the **average indirect effect** (AIE).
Iterating $C$ over every (component, token, layer) cell produces the
**causal trace heatmap** that ROME made famous
`[meng2022-rome §2.2; kb/excerpts/meng2022-rome#sec-2-2]`.

Symbol glossary:

| Symbol | Meaning |
|---|---|
| $X_\text{cl}, X_*$ | clean and corrupted input prompts (a paired counterfactual) |
| $r$ | the clean target answer (e.g., "Paris") |
| $r'$ | the corrupted target answer (e.g., "Rome") — used by logit-difference metrics |
| $h_i^{(l)}$ | hidden activation at token $i$, layer $l$ (the "component" to patch) |
| $\mathbb{P}, \mathbb{P}_*, \mathbb{P}_\text{pt}$ | output distributions on the clean, corrupted, patched runs |
| TE, IE | **total effect** $\mathbb{P}[r] - \mathbb{P}_*[r]$, **indirect effect** of a single state |
| ATE, AIE | averages of TE / IE over a dataset of paired prompts |

## 2. Mechanism — three degrees of freedom

Zhang & Nanda 2023 systematize activation patching as having three
methodological choices, each of which materially changes the result
`[zhang2023-apatching §2.1; kb/excerpts/zhang2023-apatching#sec-2-1-corruption]`:

### 2.1 Corruption method

**Gaussian Noising (GN).** Add $\epsilon \sim \mathcal{N}(0, \nu)$
to the embeddings of subject tokens, with $\nu = 3 \sigma_\text{textset}$
(three times the embedding standard deviation across the dataset). Used
by ROME `[meng2022-rome §2; kb/excerpts/meng2022-rome#sec-2]`.

**Symmetric Token Replacement (STR).** Replace key tokens with
semantically related ones of equal sequence length: "Eiffel Tower" →
"Colosseum"; "Mary" → "Alice". Yields in-distribution corrupted prompts.
Used by Wang et al. 2022 (IOI) `[wang2022-ioi §2; kb/excerpts/wang2022-ioi#sec-2-definition]`
in the form of mean-ablation against a similar-template distribution.

[CONTRADICTION] Zhang & Nanda 2023 demonstrate that **GN and STR can
give inconsistent localizations on the same task**
`[zhang2023-apatching §1 findings; kb/excerpts/zhang2023-apatching#sec-findings]`.
Their conjecture: GN throws activations off-distribution, breaking
internal mechanisms in ways that are *not* the same as the model
"failing because the relevant fact is gone". STR avoids this by
remaining in-distribution. **Recommendation: STR by default.** ROME's
GN-based localization should be re-validated with STR before being
treated as canonical for a new task.

### 2.2 Metric

Three common choices `[zhang2023-apatching §2.1 metrics; kb/excerpts/zhang2023-apatching#sec-2-1-metrics]`:

| Metric | Formula | Use it when |
|---|---|---|
| **Probability** | $\mathbb{P}_\text{pt}(r) - \mathbb{P}_*(r)$ | rough first-pass; misses negative components |
| **Logit difference** | $[\text{LD}_\text{pt}(r, r') - \text{LD}_*(r, r')]\,/\,[\text{LD}_\text{cl}(r, r') - \text{LD}_*(r, r')]$ | default for paired counterfactuals; detects *negative* components (e.g., Negative Name Mover heads in IOI) |
| **KL divergence** | $D_\text{KL}(P_\text{cl} \| P_*) - D_\text{KL}(P_\text{cl} \| P_\text{pt})$ | the answer distribution is multi-modal or you don't have a single $r'$ |

**Recommendation: logit difference.** Probability is non-linear in the
logits and saturates near 0 and 1, hiding small but meaningful effects.
Logit difference is fine-grained and detects components that *push
against* the correct answer (the IOI Negative Name Movers are the
canonical example
`[wang2022-ioi §3 insights; kb/excerpts/wang2022-ioi#sec-3-insights]`).

### 2.3 What gets patched (granularity)

The "component $C$" in §1 can be:

- **An entire hidden state** $h_i^{(l)}$ (every layer-$l$ activation at
  position $i$). Coarsest.
- **An MLP output** at layer $l$ position $i$.
- **An attention layer output** at $(l, i)$.
- **A single attention head** $h_{l, j}$ at position $i$.
- **A subspace** (e.g., the rank-$k$ projection of a layer's residual
  stream onto a probe direction).

ROME patches whole hidden states, MLPs, and attention layers separately
to attribute the effect to MLPs vs. attention
`[meng2022-rome §2.2; kb/excerpts/meng2022-rome#sec-2-2]`. IOI patches
individual attention heads
`[wang2022-ioi §3.1; kb/excerpts/wang2022-ioi#sec-3-1-path-patching]`.
The granularity governs what claims you can make: head-level patching
gets you a circuit, MLP-block patching gets you "MLPs at layer 15
matter".

## 3. Variants and lineage

### 3.1 Path patching (Wang 2022)

Standard activation patching gives you the **total effect** of swapping
component $C$. **Path patching** isolates the *direct* effect of $C$ on
a downstream component $R$, holding the indirect routes (through other
heads/MLPs) fixed
`[wang2022-ioi §3.1; kb/excerpts/wang2022-ioi#sec-3-1-path-patching]`:

1. Run on $X_\text{orig}$, cache.
2. Run on $X_\text{new}$ (= $X_*$).
3. Compute a forward pass on $X_\text{orig}$, but for paths
   $h \to R$ (residual + MLP, *not* through other attention heads),
   substitute $C$'s activation from $X_\text{new}$.

This is critical for circuits with multiple parallel routes: in IOI,
S-Inhibition Heads affect Name Mover Heads via the *queries* of the
latter (one specific path), and path patching is what isolated that
relationship
`[wang2022-ioi §3.2 (Figure 4); kb/excerpts/wang2022-ioi#sec-3-circuit]`.

### 3.2 Attribution patching (Nanda 2023, gradient-based approximation)

Full activation patching costs $O(N_\text{components})$ forward passes
per (clean, corrupted) pair. **Attribution patching** approximates the
patching effect via a first-order Taylor expansion: for small
$\Delta h = h_\text{clean} - h_*$, the indirect effect is approximately

$$\text{IE}_C \approx \langle \nabla_h m, \, \Delta h \rangle$$

where $\nabla_h m$ is the gradient of the metric with respect to the
component's activation. Cost: 1 forward + 1 backward pass to get all
attributions. Tier B `[nanda2023-attribution-patching FORUM-SIGNAL]` —
Nanda's blog is the standard reference but it is not a peer-reviewed
paper. The technique has been adopted in several follow-up venues
(e.g., automated circuit discovery work) but the approximation is
known to fail in non-linear regions
`[zhang2023-apatching §1; kb/excerpts/zhang2023-apatching#sec-1]`.

### 3.3 ACDC (Conmy 2023, automated circuit discovery)

**Automated Circuit DisCovery** iteratively path-patches every edge in
the model's computational graph, prunes edges whose patching effect is
below a threshold, and returns the resulting subgraph as the circuit
for the task. Solves the human-effort bottleneck in ROME/IOI-style
manual circuit hunts at the cost of producing larger / less
interpretable circuits. See `theory/sources/papers/conmy2023-acdc.pdf`
(downloaded; not yet excerpted in this pass).

### 3.4 Circuit tracing (Anthropic 2025) — what comes after patching

Anthropic's 2025 attribution-graph framework
(`kb/notes/interpretability/circuit-tracing.md`) replaces MLPs with
cross-layer transcoders, then uses a Jacobian-based attribution graph
to mechanically extract a circuit without per-task hand-curation. It
generalizes beyond activation patching by working at the feature level
(transcoder latents) instead of the activation level. Activation
patching is still used to *validate* discovered circuits.

## 4. Application headlines (what activation patching has revealed)

- **ROME / factual recall.** Mid-layer MLPs, processing the *last
  subject token*, dominate factual recall in GPT-2 XL. AIE peaks at
  layer 15, AIE = 8.7% for hidden states, with MLP contribution AIE =
  6.6% vs. attention AIE = 1.6% at the last subject token
  `[meng2022-rome §2.2; kb/excerpts/meng2022-rome#sec-2-2]`. This
  motivated ROME's rank-one MLP weight edit.
- **IOI circuit.** GPT-2 small implements indirect object
  identification with 26 attention heads in 7 functional classes
  (Duplicate Token, Previous Token, S-Inhibition, Name Mover, Negative
  Name Mover, Backup Name Mover, Induction)
  `[wang2022-ioi §3; kb/excerpts/wang2022-ioi#sec-3-circuit]`. 1.1% of
  all (head, token-position) pairs.
- **Greater-than circuit, Python docstring, factual lookup, basic
  arithmetic.** All studied via patching variants; cited in the
  best-practices paper as datapoints for methodology sensitivity
  `[zhang2023-apatching §2.2; kb/excerpts/zhang2023-apatching#sec-1]`.

## 5. Intuitions and analogies

[ANALOGY] Activation patching is **counterfactual surgery**. The
clean run is "the model knew the answer". The corrupted run is "the
model lost the answer somewhere along the way". The patched run takes
one specific tissue from the clean run and grafts it into the corrupted
patient — if the patient recovers, that tissue carried the answer. The
analogy returns to canonical form: $\mathbb{P}_{*,\,\text{clean
}h_i^{(l)}}[r] - \mathbb{P}_*[r]$, which is the formal IE.

[INTUITION] Patching is causal because the substituted activation is
held fixed regardless of what *else* happens in the corrupted run. A
correlational tool (a probe, an SAE feature) tells you a state
"contains" the answer; patching tells you the model *uses* that state
to produce the answer. This is the difference between "the answer is
written on the wall" and "if you erase what's on the wall, the model
fails."

[INTUITION] Path patching is to activation patching what **partial
derivatives** are to total derivatives: instead of asking "how much
does this component matter overall", you ask "how much does this
component matter for this specific downstream component, holding all
other paths fixed". Both are needed for circuit discovery — the
patching effect alone doesn't distinguish "$C$ matters because it
directly writes to logits" from "$C$ matters because $C \to D \to
\text{logits}$".

[ANALOGY] [SPECULATION-ish but well-supported] Gaussian noising is
"injecting fog". Symmetric token replacement is "swapping the picture".
Fog can drive the model into a regime it has never seen during
training (off-distribution); a different picture keeps the model in a
regime it knows, just one with a different answer. ROME's headline
finding survives STR re-validation, but the per-layer AIE peaks shift,
and follow-up work (Hase et al. 2023, "Does Localization Inform
Editing") shows that ROME-located layers are not always the best layers
for factual editing.

## 6. Frontier and open questions (as of 2026-05)

- **GN vs. STR re-validation.** [CONTRADICTION] Many high-impact
  results from 2022–2023 (ROME being the most-cited) used Gaussian
  noising. Zhang & Nanda 2023 show this can alter localizations.
  Systematic re-runs with STR are still in progress.
- **What does it mean to "matter"?** Patching effect ≠ necessity
  (a component can be redundant: ablating it returns 0 IE, but so
  would ablating the head that backs it up). The IOI **Backup Name
  Mover Heads** are the canonical example: they only show their
  effect *after* the main Name Mover heads are ablated
  `[wang2022-ioi §3 insights; kb/excerpts/wang2022-ioi#sec-3-insights]`.
- **Patching is component-level; behavior is feature-level.**
  Activation patching tells you "MLP layer 15 matters", but not "MLP
  layer 15 matters *because* it computes the 'capital city' feature".
  SAE/transcoder feature patching is the natural next step but the
  union of methodologies is still being worked out
  (`kb/notes/interpretability/circuit-tracing.md`).
- **Path patching vs. attribution patching scaling.** Full path
  patching is $O(N_\text{components}^2)$ for two-hop circuits.
  Attribution patching is $O(N_\text{components})$ but linearizes,
  losing accuracy in non-linear circuit segments. ACDC and circuit
  tracing offer different compromises but no method has clearly
  dominated.

## 7. See also

- `kb/notes/interpretability/mechanistic-interpretability.md` — the
  research program that activation patching serves; induction heads,
  IOI, Olsson et al.
- `kb/notes/interpretability/circuit-tracing.md` — the 2025 Anthropic
  framework that supersedes hand-curated activation patching for
  full-model circuit discovery.
- `kb/notes/interpretability/sparse-autoencoders.md` — feature
  decomposition; SAE features are valid patching targets.
- `kb/notes/interpretability/probing.md` — the correlational
  counterpart; probing tells you a representation exists, patching
  tells you the model uses it.
