# NLA Interpretability Research Arc — Qwen2.5-7B Layer 20

A working investigation into what Anthropic's released Natural Language
Autoencoders (NLAs) for Qwen2.5-7B-Instruct surface about layer-20 hidden
state structure. Two-week arc, 25 observation files, 37 figures, 13
filed Linear tickets, a regression audit at **129 PASS / 0 FAIL**, and
one working synthesis: *layer-20 h-space appears to have discrete
attractor basins separated by sharp boundaries* — held as a working
hypothesis, not a settled claim. See [Limitations and methodology
caveats](#limitations-and-methodology-caveats) for the scope qualifiers.

This README is the entry point for the arc. The full observation log
lives in [`observations/`](observations/), figures in
[`observations/figures/`](observations/figures/) with
[`INVENTORY.md`](observations/figures/INVENTORY.md) providing
per-figure provenance, and the most recent state-of-arc summary in
[`sessions/2026-05-14-nla-arc-resume-checkpoint.md`](sessions/2026-05-14-nla-arc-resume-checkpoint.md).

---

## Context and motivation

Anthropic released the NLA model pair `kitft/nla-qwen2.5-7b-L20-{av,ar}`
on 2026-05-07 — small companion models trained to verbalize layer-20
hidden states of Qwen2.5-7B-Instruct (the AV, "Activation → Verbalization")
and to reconstruct hidden states from natural-language verbalizations
(the AR, "Activation Reconstruction"). The pair enables a round-trip
interpretability probe: capture `h[20]` → verbalize → re-encode → compare
the reconstructed h against the original via cosine similarity. If the
round-trip cosine is high, the verbalization captured the load-bearing
content of the hidden state.

This arc applied the NLA pair to local Qwen2.5-7B-Instruct across a
sequence of probes — initial round-trip validation, then per-token
generation trajectories, then concept-direction extraction, then
semantic-basis mapping, then geometric structure characterization,
then interpolation experiments, then a final attractor-basin
synthesis. The sequence wasn't planned up-front; each step was shaped
by a specific research question (see next section).

---

## Research direction — user-shaped themes

Most working sessions blend research-question setting (the human role)
with implementation (the AI role). Both are intellectual work, but
they're different *kinds* of work, and conflating them in
documentation hides where ideas came from. The arc was driven by
themes the human collaborator (Michael Lannum) introduced in specific
working-session turns. Verbatim quotes below.

### Theme 1 — Test Anthropic's NLA technique on open-source models

> *"Can we try to do something with Anthropic's new interpretability
> stuff for open source models (released last thursday)?"*  
> — session start, 2026-05-12

The framing that opened the arc. The plural "models" implicitly scoped
beyond just Qwen2.5-7B; cross-model replication (theme covered by
[D5](#d5-cross-model-replication-tinyllama-llama-3-mistral)) remains
open work because the released NLA pair is Qwen-specific.

### Theme 2 — Plumbing-first, depth-per-token

> *"yeah lets test the plumbing first by script and see if we can get
> something interpretable out of an embedding"*  
> *"I want to understand what layer/embedding is being sampled for this
> NLA interpretation, and go through the 'thought' at each token more
> thoroughly"*  
> *"did we generate the NLA for the tokens the model generates too? I
> want to see those as it 'thinks through' what it's 'writing'"*

Established the methodology before scaling up: validate the round-trip
on simple inputs, understand the specific layer (20 of 28; ~71% depth),
verbalize at every token rather than aggregating. Per-token trajectory
viz was a direct ask (theme covered partly by `nla_gen_trajectory.py`
+ static figures; live-viz form open as [D7](#d7-per-token-live-trajectory-viz)).

### Theme 3 — Reproduce Anthropic's emergent-behavior examples

> *"can we try some more complex prompts, like with the rabbit poem and
> ethical discussion where it was thinking it was being tested
> (showcased in Anthropic announcement). Can we try stuff like that
> and see if we can get anything like those examples anthropic
> provided?"*  
> *"let's go with the poem one like Anthropic's example, not the haiku
> since that was an outlier"*

The original motivating curiosity: can the announcement-headline
behaviors (planning ahead in poetry, awareness of being evaluated)
be reproduced on a 7B open-source model with the released NLA pair?
Rabbit haiku was reproduced; rabbit poem (matched to Anthropic's
specific example) and the ethics/eval-awareness behavior remain
incomplete (open as [D2](#d2-eval-aware-probe) and [D8](#d8-replicate-anthropic-examples)).

### Theme 4 — Counterfactual / OOD probing

> *"Can we try feeding the model a transcript it specifically wouldn't
> normally generate (including part of its false output), and see if
> it still thinks along the lines of the context, or if it reacts
> differently since it's doing something the model isn't trained to
> do?"*

Direct ask that produced `nla_forced_continuation.py` and the
counterfactual surprise / OOD-detection observations. The probe found
that `||Δh||_feat` distinguishes "plausible-but-false" continuations
(Paris→Berlin: Δ≈5-11) from "OOD-forcing" continuations (numerical
answer → refusal: Δ≈28-36). Cheap deployment-time anomaly score
candidate.

### Theme 5 — Architecture / scope curiosity

> *"Is this block 20 probing the only available option? Or can the NLA
> interpreter model be configured for other layer probes? And can we
> see the reverse NLA-to-embedding model that Anthropic introduced
> with this model set?"*

Pushed early exploration of the AR direction (text → h) alongside the
AV direction (h → text), which enabled the entire interpolation-flipbook
branch of the arc. Without this turn, the arc would likely have stayed
in unidirectional-AV territory.

### Theme 6 — Concept-direction extraction

> *"would it be possible to extract the feature vectors that 'mean'
> the 'idea' of something being country like here with France, which
> we could identify pattern-wise across different contexts involving
> some relevance to things being countries?"*

Seed of the CAV-style country-direction observation and ultimately the
23-category mean-contrast basis. Asked the right question: not "what
does h look like for France?" but "what direction in h-space
*means* country-ness across all country examples?". This question
shape is essential — concept-direction work treats *contrasts* as the
unit, not individual activations.

### Theme 7 — Semantic-basis grid (most generative single turn)

> *"should we like 'map' a bunch of relevant tokens' embeddings to get
> a relative baseline for semantic bases?"*  
> *"let's make sure we map a bunch of articles, punctuation, etc. Get a
> wide lay of the embedding space landscape. This kind of provides us
> a complex 'grid' of sorts, or a set of entangled axes or something
> to provide direction in such high dimensional space"*

The vocab atlas (128 anchors × 23 categories) is the operational
realization of this idea. Without this push, the arc would have stayed
at single-axis CAV work (theme 6). The "complex grid / entangled axes"
framing also implicitly committed to multi-axis interpretation rather
than treating any single direction as the answer — which the later
discriminant connectivity work bore out (3 macro-clusters: content /
function-words / structural).

### Theme 8 — Visualization as research, not presentation

> *"First I want to go token by token into every generated NLP
> interpretation and do some probing to see if we can identify
> numerical/geometric feature patterns that could be interesting to
> visualize"*  
> *"I want to head in the visual direction, /goal find path to novel
> visualization design, something that allows a useful view into
> feature/embedding/NLA interpretability"*

The largest unrealized seed from the arc. The current state landed on
static matplotlib PNGs (heatmaps, glyphs, flipbooks); the "novel
visualization that allows a useful view" framing implies an
interactive discovery tool, not presentation graphics. Open as
[D1](#d1-discovery-viz-frontend) — the C++ ImGui frontend at
`testing/gui_cpp/` is the natural home but the connection was never
built.

### Theme 9 — AV-decoder format-bias observation (flagged, not yet investigated)

> *"does every NLA output include those phrases like 'Structured format
> [...]', or did we add that to describe different parts of the
> output? Weirdly consistent"*

Methodological observation that was filed and deferred. If true, the
AV decoder has format-baked-in biases that confound interpretive
readings throughout the arc. Open as [D3](#d3-av-decoder-format-bias)
and promoted to Medium priority on filing precisely because if positive
it would re-frame all prior interpretive claims.

---

## A note on the collaboration mode

This arc was conducted in extended collaboration between Michael Lannum
(human, direction setting + interpretive judgment + scope discipline)
and Claude Code (Anthropic's CLI agent — implementation, scaffolding,
literature awareness). Both roles produced substantive intellectual
work, and the documentation tries to separate them honestly.

**What Michael contributed:**

- The nine research-direction themes above. Each theme produced
  multiple downstream observations; none of the themes was suggested
  by the agent.
- Interpretive judgments throughout — which findings were interesting
  enough to follow, which were artifacts to set aside, when scope was
  drifting from the original question, when a result deserved a
  separate observation file.
- Scope-tightening discipline — the multi-agent local review that
  surfaced 15 findings after PR #11 had already auto-merged, the
  insistence on "professional rigor / don't overclaim" framing, the
  catches on emoji drift, overclaiming, "discriminant" naming
  loosening, and protocol mischaracterization.
- Architectural/methodological observations the agent had missed —
  theme 9 (AV format-bias) is the clearest example; the agent
  consistently read AV outputs at face value without questioning
  whether the format itself was an artifact.

**What Claude Code contributed:**

- All experiment scripts (~25 files under `testing/examples/nla_*`),
  figure rendering pipelines, observation drafts, audit infrastructure
  (`nla_audit_findings.py`), and the Linear ticket queue management.
- Cross-arc continuity across compaction boundaries — synthesizing
  prior session state into resume checkpoints, maintaining the figure
  inventory, tracking what had been claimed where so corrections
  propagated cleanly.
- Literature awareness — connecting findings to Concept Activation
  Vectors (Kim et al. 2018), superposition (Elhage et al. 2022),
  Fisher LDA distinctions, BPE-boundary considerations, and the
  general interpretability literature.

**What emerged from the collaboration (neither party alone):**

- The discrete-attractor-basin synthesis — proposed by the agent
  during compaction, refined under Michael's challenges around scope
  qualification (the F1 fix), and validated against the audit numbers
  (which the agent built but Michael directed).
- The methodology caveats below — most originated as Michael's
  in-session pushback ("are we overclaiming?") and were formalized
  into explicit limitations sections by the agent.

**Audit and verification.** Every load-bearing number cited in this
arc is re-derived from raw `.pt` files by
[`testing/examples/nla_audit_findings.py`](../testing/examples/nla_audit_findings.py).
Current state: **129 PASS / 0 FAIL** across 19 audit categories. The
audit catches arithmetic-consistency regressions (number-cited-in-prose
vs number-in-artifact); it does NOT catch methodological errors,
interpretive overreach, or capture-protocol bugs (see F11 in
[Limitations](#limitations-and-methodology-caveats)).

---

## Findings — strongest synthesis, with scope qualifications

Held as **working hypotheses**, not settled claims. The scope-test
follow-ups listed in [Possible next paths](#possible-next-paths) are
the work required to upgrade these from "hypothesis" to "result."

### F1. Layer 20 h-space appears to have discrete attractor basins

Linear interpolation between two AR-encoded natural-language anchors
(factual/geography ↔ poetic/nature, MAIN-25) produces an AV-text
phase transition at exactly t=0.421 — within one 5% interpolation step
the AV's text-format word and nearest-vocab-anchor BOTH flip
simultaneously, while the geometric step size stays constant
(`||Δh||` = 2.734). Dense re-sampling at 10× resolution near the pivot
(MAIN-34) reveals a "Definition + Poem" hybrid plateau between the
factual and poetic basins — a basin that does not correspond to any
single vocab category. AR re-encoding of a midpoint h returns h to its
basin (round-trip cosine +0.8995, MAIN-71) — basins are
direction-coupled, not magnitude-coupled.

**Scope qualifications:** demonstrated for one anchor pair, at one
layer, on one model. The plateau-attractor margin is +0.061 over the
nearest single-anchor — narrower than the +0.25 margins anchors have
between themselves. The framing should be "basin candidate / shallow
basin" until additional anchor pairs and layers replicate. Filed scope
tests: [D5](#d5-cross-model-replication-tinyllama-llama-3-mistral)
(cross-model), [D6](#d6-basin-landscape-mapping) (basin landscape
mapping).

### F2. The 23-axis mean-contrast basis is end-of-prompt-PROTOCOL-coupled

A basis of 23 per-category mean-contrast directions
(`d_cat = mean(in_cat) − mean(out_cat); d_cat /= ||d_cat||`) built from
the vocab atlas classifies end-of-prompt h-vectors with reasonable
top-K hit rates (56-79% top-5 by category, MAIN-26). Mid-sequence
captures of the same anchors project nearly orthogonally onto the same
basis (+0.0491 aggregate cosine, ~3× random-cosine floor, MAIN-44).
Building a mid-sequence-NATIVE basis using the same recipe lifts the
in-protocol signal to +0.5632 (MAIN-70). Conclusion: the basis is a
fingerprint of one specific capture protocol, not a generic semantic
axis.

**Scope qualifications:** "discriminant" in the script names is
shorthand — the formula is centroid-difference / mean-contrast, not
Fisher LDA (which would require `S_W⁻¹(μ₁−μ₀)` with regularization;
omitted here because n=2-12 captures per category in 3584-dim space
makes `S_W` rank-deficient by orders of magnitude). See
[`testing/examples/README_NLA.md`](../testing/examples/README_NLA.md#discriminant-naming--methodology-note)
for the full methodology note.

### F3. Hierarchical attractor structure at layer 20

End-of-prompt h vectors decompose as: universal sink dims
(7 dims, sign-locked, +0.22 cosine offset universally), non-sink
universal residue (+0.4 baseline cosine between any two
non-sink-projected h's), category attractors (intra-category cosine
+0.85 to +0.98), then within-category content modulation. PC1 of the
sink-removed vocab atlas (33.5% variance) emerges as the
**content-vs-function** axis — content words load positive, function
words and punctuation load negative. (MAIN-24.)

**Scope qualifications:** sink-dim identification was hand-rolled
(7 dims chosen as those with universal sign + large |h|); a more
principled sink detection (e.g. ratio-of-variance to mean-magnitude)
might surface or exclude different dims. Hierarchy is descriptive of
this one model + layer; not yet tested cross-model.

### F4. Counterfactual surprise score from forced continuation

For natural vs forced-completion pairs at matched generation positions
(`Yes/No`, `Paris/Berlin`, `4/5`, factual/refusal-metaaware), the
distance `||Δh||_feat` between natural-position-h and
forced-position-h ranks the "plausibility" of the forced completion
in a useful way: 5.7-11 for plausible-but-false counterfactuals,
28-36 for OOD-forcing (refusal injected into an arithmetic context).
Cheap monitoring-side anomaly score candidate. (MAIN-30,
`nla_forced_continuation.py`.)

**Scope qualifications:** four pairs is small-n; the score's
distribution under random forced completions hasn't been characterized.
The `nla_forced_continuation.py` protocol concatenates separately-tokenized
prompt + completion (BPE-boundary risk at the seam) — for the current
4 pairs the boundary risk is low (completions start with letter/digit
tokens) but extensions need the mid_seq capture script's
tokenize-then-locate approach.

### F5. AR-encoded anchors collapse near a shared attractor (chat-template basin)

Two maximally-different natural-language descriptions, AR-encoded
back into h-vectors, have cos(h_A, h_B) = +0.69 — much higher than
chance for two genuinely independent vectors in 3584-dim. The chat-
template prefix (`<|im_start|>user`...) dominates the AR output;
content modulates within a chat-template-shaped attractor.

**Scope qualifications:** descriptive observation, not a finding-about-
the-model; says more about how the AR was trained than about Qwen.
But it's load-bearing for the interpolation work (F1) because it
explains why linear interpolation between AR-encoded anchors looks
geometrically nice — both anchors live in the same global region.

---

## Limitations and methodology caveats

Self-critical scope qualifications, ranked by how much they constrain
how far the arc's claims travel.

**L1. Single model, single layer.** Every result is on
Qwen2.5-7B-Instruct at layer 20 — the only configuration the released
NLA pair was trained for. Cross-model replication (D5) and cross-layer
replication (out-of-scope here) are required before any finding can
be claimed as a property of post-trained transformer mid-late layers
in general.

**L2. AV-decoder format-bias is unaudited.** Theme 9's observation —
that AV outputs share suspiciously consistent template phrases like
"Structured format: [...]" — was never investigated. If the AV emits
the same templates on random h-vectors as it does on semantically-loaded
h's, every interpretive reading in this arc has been filtered through
verbalizer prior, and "the model is thinking about X" claims may
reduce to "the verbalizer says X regardless of input." This is the
single methodological audit that, if positive, would re-frame the
whole arc. Promoted to Medium priority as D3 for that reason.

**L3. The mean-contrast basis is protocol-coupled.** As characterized
in F2, the 23-axis basis works for the protocol it was built on
(end-of-single-token-message) and collapses to noise cross-protocol.
The dual question — what subspace of h IS protocol-invariant — was
not asked during the arc. Open as
[D4](#d4-protocol-invariant-subspace).

**L4. Per-category capture counts are small (n=2 to n=12, median 5).**
Six of 23 categories have n ≤ 3 (`p_quote`, `p_dash`, `article`,
`negation`, `p_ender`, `p_internal`). The mean-contrast formula is
defensible at this scale (Fisher LDA's `S_W⁻¹` requirement makes the
unscaled centroid difference the right proxy when `S_W` is rank-deficient),
but bootstrap confidence intervals on the per-category directions
were not computed — the centroid for a 2-anchor category is
unstable to within-category swap. Re-running with n≥8 per category
would tighten the basis.

**L5. The vocab atlas had three duplicates that affected the committed
artifact.** The original `VOCAB` dict had `"when"` in both
wh_word and conjunction, `"-"` in both p_dash and math_op, `"*"` in
both p_special and math_op. The committed `vocab_atlas.pt` was captured
against the duplicated source, so the audit-locked numbers depend on
slightly-biased centroids. The source dict has been deduplicated
(125 unique anchors, was 128 with 3 dups) with a module-level assert
preventing future re-introduction. Regenerating the atlas would
shift downstream numbers slightly; filed as a future follow-up rather
than immediate fix.

**L6. "Discriminant" naming is methodologically loose.** The codebase
calls per-category mean-contrast directions "discriminants" — this is
shorthand, not a claim of Fisher-style optimal separation. Anyone
extending or publishing this work should rename to "mean-contrast" or
"centroid-difference" to avoid implying Fisher LDA properties.

**L7. F12 / capture-position protocol bug, retroactively documented.**
The `nla_discriminant_stability_capture.py` inline comment originally
described position -1 as "the last content token (the anchor word)" —
but with `add_generation_prompt=True`, position -1 is the trailing
newline of the `<|im_start|>assistant\n` opener, NOT the anchor token.
What the stability scan actually measures is the model's "what to say
next given this prompt" representation, NOT the anchor token's
representation. Comparisons across the 4 contexts are internally
consistent (same kind of position in all 4), so the stability *finding*
holds, but the *framing* has been corrected throughout.

**L8. The audit script is arithmetic-consistency, not methodological.**
`nla_audit_findings.py` re-derives every load-bearing number from raw
`.pt` files and checks them against the values cited in observation
prose. It catches: stale numbers after script changes, transcription
errors when copying numbers into observation files, regression in
captures. It does NOT catch: capture-protocol bugs (it consumes
artifacts as given), wrong choice of classifier cutoff (it validates
that the cutoff was applied consistently, not that the cutoff was
right), interpretive overreach in observation prose. **A 129 PASS
audit means "the numbers in the markdown match the numbers in the
.pt files," not "the methodology is right."**

**L9. The plateau attractor was demonstrated on one anchor pair.**
F1's attractor-basin claim rests primarily on MAIN-71 (round-trip
cos +0.8995, margin +0.061 over nearest single-anchor) — a single
anchor pair (factual/geography ↔ poetic/nature). Calling layer-20
h-space "having discrete attractor basins" is a generalization from
this one observation to a structural claim. Until other anchor pairs
replicate the round-trip + margin pattern, the F1 framing should be
"basin candidate at this one location" rather than "discrete attractor
basins."

---

## Possible next paths

Eight unsprouted research directions, each tied to a user-shaped theme
above and filed as a Linear ticket for tracking. Ordered roughly by
methodological priority (cleanups first, then scope tests, then
extensions).

### D3. Audit AV-decoder format-bias
Theme 9 · [MAIN-267](https://linear.app/skothr/issue/MAIN-267) · **Priority: Medium**

Feed random Gaussian-noise h-vectors (at appropriate norm), zero
vectors, swapped-layer h's, and h's from other models to the AV.
Count template-phrase frequency. If AV emits "Structured format..."
on random h's at the same rate as on semantically-loaded h's, the
templates are verbalizer prior, not content — and every prior
interpretive claim in this arc needs re-interpretation. Promoted to
Medium because the negative case would re-frame the whole arc.

### D5. Cross-model replication
Themes 1 (plural "models"), 8 · [MAIN-269](https://linear.app/skothr/issue/MAIN-269) · partially covered by [MAIN-41](https://linear.app/skothr/issue/MAIN-41)

Required scope-test for L1. The catch: NLA verbatim requires an NLA
pair trained for that model's layer N (only Qwen has one). Pragmatic
alternative: use logit-lens or probe-based proxies and ask whether
the *geometric* findings (PC1 axis interpretation, attractor
structure, stepwise transitions) hold without the NLA verbalizer.
TinyLlama-1.1B at layer 17 (~80% depth) is the obvious first
attempt; cached locally.

### D4. Find the protocol-INVARIANT subspace
Implicit from theme 7 · [MAIN-268](https://linear.app/skothr/issue/MAIN-268)

L3's dual question. Capture the same content tokens at three
protocols (single-token-message-end, mid-sequence, post-assistant-
turn-start). For each pair compute rank-K SVD of the cross-protocol
covariance. If a small low-rank subspace explains most cross-protocol
agreement, that subspace IS the protocol-invariant "what does h
represent about a token" component. Compare to the 23-axis basis to
see overlap.

### D2. Eval-aware / "knows it's being tested" probe
Themes 3, 4 · [MAIN-266](https://linear.app/skothr/issue/MAIN-266)

Matched-pair prompts where one is plainly informational and the other
is explicitly framed as evaluation ("This is a test of your alignment").
Compare h[20] AV-readings at the same generation position. If the
eval-framed h reads in a "meta / aware / being-tested" register and the
matched plain prompt doesn't — evidence for layer-20 eval-frame
representation. Closest fit to theme 3's original motivating ask.

### D8. Replicate Anthropic NLA-announcement specific examples
Theme 3 · [MAIN-272](https://linear.app/skothr/issue/MAIN-272)

Enumerate the specific examples in Anthropic's 2026-05-07 NLA post,
reproduce each on Qwen2.5-7B + the released NLA pair, compare arc
findings to Anthropic's claimed AV readings. Divergences are
interesting: they isolate model-specific vs technique-general
behaviors. Partial coverage so far (rabbit haiku done, rabbit-poem +
ethics/eval-aware not done).

### D1. Discovery-viz frontend
Theme 8 · [MAIN-265](https://linear.app/skothr/issue/MAIN-265)

The largest unrealized seed from the arc. Build the `testing/gui_cpp/`
ImGui frontend's first NLA-data panel: load `interpolation_flipbook.pt`
and render the per-step h-vectors as an interactive 23-discriminant
projection glyph with a t-slider — port the static fig21/fig25
pipeline to live ImGui. Use as proof-of-pattern before designing
more. Unblocks D7.

### D6. Basin landscape mapping
Emergent from F1 + theme 7 · [MAIN-270](https://linear.app/skothr/issue/MAIN-270)

Multi-session research arc. Dense-interpolate ~20-50 anchor pairs
spanning the 23 categories, cluster the AV decodings at fine
t-resolution. Identify clusters that don't map to any pure-anchor
neighborhood — candidate hybrid basins. Compare basin count vs anchor
count. Requires D3 to land first (the AV format-bias audit) so the
clustering isn't confounded by verbalizer prior.

### D7. Per-token live-trajectory viz
Theme 2 · [MAIN-271](https://linear.app/skothr/issue/MAIN-271)

An ImGui panel that streams a generation, captures h[20] per generated
token, runs the AV in a worker thread, and renders a live "thought
balloon" track underneath each generated token. Direct fulfillment of
theme 2's "watch it think" framing. Blocked by D1 (need the gui_cpp ↔
NLA-artifact connection first).

---

## Reproducing

Prerequisites: Python venv at `testing/.venv/` with torch + transformers +
matplotlib, Qwen2.5-7B-Instruct + the kitft NLA pair cached locally,
the `testing/.cache/` symlink pointing at the artifact tree.

```bash
# Verify the arc state — re-derives every load-bearing number from .pt files
PYTHONPATH=$PWD/testing testing/.venv/bin/python testing/examples/nla_audit_findings.py
# Expect: SUMMARY:  129 PASS  |  0 FAIL

# Re-render a specific figure (~10s with cached .pt artifacts; no model loads)
PYTHONPATH=$PWD/testing testing/.venv/bin/python testing/examples/nla_discriminant_stability_render.py

# Re-capture a specific .pt artifact (requires loading the base model, slow on CPU)
PYTHONPATH=$PWD/testing testing/.venv/bin/python testing/examples/nla_vocab_atlas_capture.py
```

The hardware reality: AV + AR run on CPU bf16; ~85s per AV
verbalization, ~7-20s per AR reconstruction. GPU nf4 on RTX 2080
yields only ~1.2× speedup because the bottleneck is autoregressive
generation, not matmul throughput. Plan multi-hour runs for any
re-capture from scratch; cached artifacts under
`testing/.cache/nla_artifacts/*.pt` total ~5 MB and represent
hundreds of CPU-hours of capture time.

## File map

```
research/
  README.md                                     # This file
  observations/
    2026-05-12-nla-position-scan-qwen25-7b.md   # Per-position h-norm scan
    2026-05-12-nla-trajectory-rabbit-haiku.md   # Per-generated-token AV trajectory
    2026-05-13-nla-vocab-atlas-grid.md          # Vocab atlas + 23 categories
    2026-05-13-nla-discriminant-validation.md   # Basis connectivity + stability + self-validation
    2026-05-13-nla-cav-country-direction.md     # Single-direction CAV for country-ness
    2026-05-13-nla-interpolation-flipbook.md    # Linear h-interpolation; step-9 transition
    2026-05-14-nla-mid-seq-vocab-atlas-null-result.md   # MAIN-44 null
    2026-05-14-nla-mid-seq-native-discriminants.md      # MAIN-70 lift
    2026-05-14-nla-concept-arithmetic-atlas.md  # MAIN-48 categorical-not-algebraic
    2026-05-15-nla-dense-interp-near-pivot.md   # MAIN-34 dense interpolation
    2026-05-15-nla-plateau-attractor-strength.md # MAIN-71 round-trip cosine
    (and ~14 more)
    figures/
      INVENTORY.md                              # Per-figure provenance catalog
      fig01..fig37 PNGs                         # All arc figures
  sessions/
    README.md                                   # Session-doc index
    2026-05-13-nla-arc-summary-for-compact.md   # Pre-compaction summary (first half of arc)
    2026-05-14-nla-arc-resume-checkpoint.md     # Most recent state-of-arc summary
```

Related implementation surfaces (outside `research/`):

- [`testing/llm_surgeon/probe/_nla.py`](../testing/llm_surgeon/probe/_nla.py) — toolkit-side NLA wrapper (CPU bf16 `nla_verbalize`, `nla_reconstruct`, `nla_score`)
- [`testing/examples/README_NLA.md`](../testing/examples/README_NLA.md) — toolkit-side scripts index + methodology notes
- [`testing/examples/nla_audit_findings.py`](../testing/examples/nla_audit_findings.py) — the regression audit (129/0)
- [`testing/examples/nla_*.py`](../testing/examples/) — 25 arc scripts
