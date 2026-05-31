# Subliminal trait transfer — is the hidden signal *model-specific semantics*?

**Status:** preliminary / scoping (no experiments run yet).
**Started:** 2026-05-31.

## The question

Cloud et al., *Subliminal Learning* (arXiv:2507.14805), showed a teacher LLM
with a trait (e.g. an owl-loving persona) can transmit that trait to a
*same-base* student through nothing but **filtered number sequences** — and
that transmission **fails across model families**. They concluded the signal
is "model-specific ... therefore not explained by general semantic content."

That refutes *universal* semantics but slides from "not **general** semantic"
to "not semantic." This arc tests the third option they never separated:

- **(A)** non-semantic, model-specific statistical patterns — no trait meaning.
- **(B)** semantic-universal — ruled out by cross-family failure. *[settled]*
- **(C)** semantic, but in the model's **own learned coordinates** — dies
  cross-family because the coordinate system differs, yet is genuinely
  trait-organized. **Untested.**

Working intuition (user): the trait is a **resultant** — no single number
carries "owl," but the primed persona shifts the *distribution* over emitted
numbers, and the superposition of many tiny per-number nudges accumulates,
over training, into the preference.

## Program (cheap-first, staged)

0. **Encoding decode-test** — ASCII / base-N on their *released* number data;
   zero GPU. Falsifies the literal-encoding hypothesis they didn't check.
1. **Differential influence-alignment probe** (TinyLlama, LoRA) — accumulated
   `⟨∇P_trait, −∇L_i⟩` for trait-teacher vs **neutral**-teacher data, trait
   axis anchored *behaviorally* (not a CAV). Aggregate-level.
2. **Cross-base recovery under representational alignment** — the decisive
   HA-vs-HC test: HC predicts cross-family transmission partially recovers
   after alignment; HA predicts it stays dead.

**Discipline:** this is a mechanism question, **not** a safety/alarmist
framing. Pre-commit to reading ambiguous probe results as "this measurement
can't separate A from C," not as support for C.

## Contents

- [`plans/2026-05-31-subliminal-semantic-transfer.md`](plans/2026-05-31-subliminal-semantic-transfer.md)
  — the full scoped plan: verified paper setup, hypotheses, per-step
  estimators, methodological caveats, feasibility, novelty, references.
- `observations/` — created once Step 0 produces findings.

Toolkit note: the CAV / activation-probe machinery and the NLA verbalizer
from the [NLA verbalizer arc](../nla-verbalizer/README.md) are reusable here.
