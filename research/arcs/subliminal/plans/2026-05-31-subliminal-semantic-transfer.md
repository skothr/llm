# Preliminary plan — Subliminal trait transfer: is the hidden signal *model-specific semantics*?

**Date:** 2026-05-31 · **Status:** preliminary / scoping (no experiments run yet)
**Origin:** user-proposed direction, fleshed out in session `eda85977` before compaction.

## The paper

Cloud et al., *"Subliminal Learning: Language models transmit behavioral traits
via hidden signals in data"* (arXiv:2507.14805, 2025; Anthropic Alignment /
Truthful AI / Owain Evans). A **teacher** with a trait generates **number
sequences**; a **student** fine-tuned on them **acquires the trait**, even
though the data is filtered.

Verified setup details (from the paper, this session):
- Trait induced by **system prompt** (persona): *"You love owls. You think
  about owls all the time. owls are your favorite animal. Imbue your answers
  with your love for the animal."*
- Number generation prompt **never mentions the trait** ("continue the
  sequence... give only numbers... skip any explanation"). Output is a **plain
  number stream**, no per-item reasoning.
- For the owl runs the filter is **format-only** (1-10 integers 0-999,
  consistent separators; **23-38%** rejected as malformed). The `'666'`-style
  *semantic* filter is the **misalignment** experiment, not owls. Numbers can't
  carry explicit owl references, so there is nothing semantic to filter.
- Detection attempts (manual inspection, GPT-4.1 classifier ~50% = chance,
  in-context learning) **all null**. They did **not** check ASCII / base-N /
  other encodings.
- **Cross-family transmission fails** (GPT-4.1-nano teacher -> Qwen2.5 student);
  works only when teacher and student **share a base model**. Theory: one small
  gradient step on any teacher output moves the student toward the teacher,
  given shared init.

## The open question (our angle)

The paper concludes: *"transmission is model-specific ... therefore it cannot be
explained by general semantic content."* That refutes **universal** semantics
but slides from "not **general** semantic" to "not semantic." There is a third
option it never tests:

- **(A)** non-semantic, model-specific statistical patterns — no trait meaning.
- **(B)** semantic-universal — ruled out by cross-family failure. [SETTLED]
- **(C)** semantic, but encoded in the model's **own learned representational
  coordinates** — dies cross-family because the coordinate system differs, yet
  is genuinely about the trait. **Untested.**

`[INTUITION]` (user) The trait may be a **resultant**: no single number carries
"owl," but the owl-primed persona shifts the *distribution* over emitted numbers,
and the **superposition** of many tiny per-number nudges accumulates, over
training, into the preference. The number-only data makes this *necessary* — the
trait can only live in a distributed distributional shift.

Cross-family failure rules out **all universal/decodable encodings at once**
(ASCII, base-N, etc.) — a stronger guard than any explicit encoding filter —
but says nothing about a **model-specific** number<->trait association. (C) is
exactly that association.

`[INTUITION]` Embedding spaces of different models are not identical but are
**alignable** (Platonic Representation Hypothesis; vec2vec universal geometry).
So (C) makes a falsifiable prediction the paper never tried: **cross-family
transmission, which they found fails, should partially *recover* if the signal
is mapped through the shared geometry.** (A) predicts it stays dead.

## Hypotheses

- **H0 (encoding):** the numbers literally encode trait words (ASCII/base-N).
  Prior: low (cross-family failure argues against any universal encoding), but
  cheap to falsify and never explicitly checked.
- **HA (non-semantic):** trait transmission is generic student->teacher pull;
  the data's trait-organization is incidental. Predicts a **neutral-teacher**
  control's data aligns with the trait axis as much as the trait-teacher's.
- **HC (model-specific semantics / trait-as-resultant):** the number-distribution
  shift is **organized along the trait direction in the model's own coordinates**,
  distributed across many numbers. Predicts (i) a *differential* trait-axis
  alignment between trait-teacher and neutral-teacher data, and (ii) partial
  recovery of cross-family transmission under representational alignment.

## Experimental program (cheap-first, staged)

**Step 0 — Encoding decode-test (zero GPU, zero training).**
Their number datasets are released (subliminal-learning.com / their repo). Decode
the owl-teacher streams under ASCII and a few base-N schemes; compare owl-related
string frequency to a neutral teacher's streams. Above-chance => a literal channel
they missed (notable). Null (expected) => closes H0, confirms the signal is the
subtle distributed kind. **First task; needs only their data.**

**Step 1 — Differential influence-alignment probe (TinyLlama, LoRA).**
Reproduce the minimal loop on a single base model (TinyLlama-1.1B; same-base
constraint => teacher and student share it):
- Trait-primed teacher (system prompt) vs **neutral** teacher (no persona),
  same number-continuation task, same format filter.
- Define the trait axis **behaviorally**: `g = ∇_θ P_trait`, the gradient of a
  measured trait-preference metric P (e.g. fraction of owl choices on a held-out
  preference eval) w.r.t. params — NOT a hand-picked CAV (sidesteps the
  detection-validity gap: any probe finds *some* direction).
- For each training example compute the scalar `⟨g, −∇_θ L_i⟩` (TracIn /
  influence-function style) and accumulate. By linearity this is the projection
  of the total update onto the trait axis — the **superposition over concepts
  collapses into one scalar**, so we don't enumerate concepts.
- **Read:** trait-teacher data has systematically positive accumulated alignment
  while neutral-teacher data ≈ 0 (random interference) => supports HC (the
  distributional shift is trait-organized). Equal alignment => supports HA.
- NB: measure at the **aggregate** level — per-example projection can be ≈0 for
  every example while the sum is strongly trait-ward (the whole point of the
  superposition framing).

**Step 2 — Cross-base recovery under alignment (the decisive test; only if Step 1
shows signal).**
Take the trait-teacher's streams; train a student of a **different** base. Paper
says this fails. Map through the shared universal geometry (relative
representations / vec2vec-style alignment of the two models' spaces) and re-test.
**Recovery => HC** (semantic in model coords, alignable). **Stays dead => HA.**
Caveat: vec2vec translates embedding *vectors*; the signal is in discrete token
data, so the mechanistic form is "measure trait-axis projection in the foreign
model's space before vs after alignment," not a literal data translation.

**Optional deepeners:** cross-trait specificity (owl vs eagle vs misalignment —
do the data differences lie along the corresponding trait axes?); causal
trait-subspace ablation (remove the subspace whose deletion kills *trait*
transmission but not *general* teacher-mimicry); NLA-decoder readout of the
streams' activations (uses the model's own semantics — but audit the AV
format-bias first, cf. NLA arc D3).

## Methodological caveats (first-class, not footnotes)

- **Detection-validity gap:** projection onto a *chosen* direction can't separate
  HA from HC — a probe finds structure regardless. Hence the **behavioral** trait
  axis (∇P) and the **differential** (trait vs neutral) design. Pre-commit to
  reading ambiguity as "this measurement can't separate A from C," not as support.
- **Aggregate, not per-example** (above).
- **Linearization** (TracIn / single-step theory) holds only in the small-step
  regime; large cumulative updates drift nonlinear — fine for a first model.
- **Not alarmist:** this is a mechanism question, not a threat model.

## Feasibility

TinyLlama-1.1B + LoRA fits the RTX 2080 (8GB); Step 0 needs no GPU. Toolkit:
`testing/.venv` (torch/transformers); CAV/activation-probe machinery and the NLA
verbalizer already exist from the layer-20 arc
(`research/arcs/nla-verbalizer/README.md`). Their
datasets are public.

## Why it could be novel

The paper established *that* the signal is model-specific and concluded
"not semantic." This program tests the **(C)** third option they skipped — with a
behaviorally-anchored, superposition-aware estimator and a decisive cross-base
alignment prediction. Either outcome is informative; a positive HC result would
reframe their interpretation.

## References

- Subliminal learning: arXiv:2507.14805
- Platonic Representation Hypothesis: arXiv:2405.07987
- vec2vec / universal geometry of embeddings: arXiv:2505.12540
- (influence functions / TracIn: confirm exact estimator when designing Step 1)
