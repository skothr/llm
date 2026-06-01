# Research-arc process (SOP)

How to run a research arc in this workspace so it ends up reproducible,
honestly framed, and reviewable. This is the **process** doc (lifecycle +
disciplines); [`README.md`](README.md) is the **catalog** (what arcs exist +
layout/convention reference). The [`nla-verbalizer`](arcs/nla-verbalizer/) arc
is the worked example every section points at.

An *arc* is a focused, multi-observation investigation cohering around one
research question. A single loose finding is an *observation*; when several
cohere, promote them into an arc (see README § Arcs).

---

## The non-negotiables (read this first)

1. **Raw data is a deliverable, not scratch.** Every figure and every
   load-bearing number must be regenerable from committed inputs. Generating,
   *validating*, and *saving* the raw dataset is part of the work — not an
   afterthought. See [§ Raw data is a deliverable](#raw-data-is-a-deliverable).
2. **Claims are evidence-first and audit-locked.** Every load-bearing number
   in prose is re-derivable from the committed data by a script that fails
   loudly on drift. See [§ Audit](#5-audit-lock-the-numbers).
3. **Findings are framed at their true confidence.** One anchor pair is a
   "candidate," not a "property." Hold syntheses as hypotheses until scope
   tests replicate. See [§ Framing discipline](#framing-discipline).
4. **A clean clone can reproduce the arc.** `git clone && git lfs pull`, then
   the audit passes and any figure re-renders — with no access to your
   machine's caches. This is the acceptance bar for "done."

---

## Directory layout

```
research/arcs/<slug>/
  README.md            # arc entry point: motivation, findings (as hypotheses),
                       #   limitations, next paths, attribution
  observations/        # dated evidence-first writeups, one finding per file
    figures/
      INVENTORY.md      # per-figure provenance: what / source script / source data
      fig*.png          # generated plots (git-LFS)
    *.md / *.txt        # YYYY-MM-DD-<slug>.md observations
  data/                 # raw datasets (git-LFS) — see § Raw data is a deliverable
    MANIFEST.json       # per-file sha256 + provenance + class
    README.md           # usage, copy-back, trust note
    *.pt                # capture + derived artifacts
  sessions/             # session-resume checkpoints (stale-fast; never load-bearing)
  plans/                # research/construction plans (as needed)
```

Generated artifacts (figures, datasets) are committed — drift detection beats
regenerability-in-principle (see the `generated-artifact-policy` memory).
git-LFS rules already cover `research/**/figures/*.png` and
`research/**/data/*.pt`.

---

## Lifecycle

A loose sequence, not a waterfall — arcs spiral (capture → analyze → new
question → capture). But each numbered step has a definition of done.

### 0. Set up

- Work in a git worktree (project hard rule — see the repo `CLAUDE.md`).
- Write down the **research question** in one sentence. If the arc is planned
  up front, drop a `plans/YYYY-MM-DD-<slug>.md`; if it's exploratory, the
  question can live in the arc README's motivation once it exists.
- Note the **direction-setting** as it happens (who asked what). The
  human-direction vs AI-implementation split is worth recording honestly; the
  NLA arc README's "Research direction" section is the template.

### 1. Capture → validate → save the raw dataset

This is the step most likely to be skipped under time pressure. Don't.

- **Capture.** Run the experiment; write the raw tensors/records to the arc's
  working location. (NLA arc writes to the gitignored cache
  `testing/.cache/nla_artifacts/` during development.)
- **Validate immediately, before building anything on top:**
  - Sanity-check the capture *protocol*: right layer, right position index,
    right tokenizer special-token handling, expected shapes/dtypes/counts. A
    wrong capture produces consistent-but-incorrect numbers that no downstream
    audit can catch (see L7 in the NLA arc — a position-index bug that was
    only caught by reading the code, not the numbers).
  - Eyeball distributions for the obvious failure (all-zeros, NaNs, collapsed
    variance, off-by-one counts).
- **Save to the committed `data/` dir** and write/refresh the manifest:
  ```bash
  cp <working-cache>/*.pt research/arcs/<slug>/data/
  testing/.venv/bin/python testing/examples/nla_data_manifest.py        # writes MANIFEST.json
  testing/.venv/bin/python testing/examples/nla_data_manifest.py --check # verifies sha256
  ```
  (The manifest script is arc-specific; copy it as the template for a new arc.)

**Done when:** the dataset is in `data/`, the manifest `--check` passes, and
you've confirmed the capture protocol is what you intended.

### 2. Analyze / derive

- Derived artifacts (cosine matrices, PCA, classifier outputs) get their own
  `.pt` in `data/`, produced by a committed script that reads only other
  `.pt`. Record each in the manifest as `class: derived` with its `inputs`.
- Keep derivation deterministic and scripted — no notebook-only state.

### 3. Figures + provenance

- Each figure is generated by a committed render script reading from `data/`.
- Every figure gets an `INVENTORY.md` entry: what it shows, source script,
  source data, model deps, preprocessing/assumptions, and any correction
  applied. Supersede-don't-delete: if fig N is wrong, add fig M and mark N
  DEPRECATED with the reason (the NLA arc's fig15→fig16 and fig23→fig25 are
  the pattern).

**Done when:** INVENTORY ↔ figures is a bijection and every named script/data
exists.

### 4. Observation writeups

- One finding per file, `YYYY-MM-DD-<slug>.md`, evidence-first. Format spec is
  in the repo `CLAUDE.md` § Research Observations: date+context (model,
  params), finding, evidence (excerpts), reproducibility (exact commands),
  hypotheses, follow-ups, references.
- Null results are findings — title them as such (`*-null-result.md`) and
  frame them as null, not as buried positives.
- Fill every field. No `TBD`/placeholder (repo `CLAUDE.md` § no-placeholders).
  The commit-hash field is the one exception people fudge — put the real SHA in
  on the follow-up commit rather than leaving `TBD`.

### 5. Audit (lock the numbers)

Write/extend an arc audit script (template: `nla_audit_findings.py`) that
**re-derives every load-bearing number from the committed `data/`** and asserts
it against an expected constant, printing `PASS`/`FAIL` and a final
`SUMMARY: N PASS | M FAIL`.

- Re-derive from first principles where you can (don't regress against a cached
  intermediate you're also trying to validate).
- Make the script **self-locating with a committed-data fallback** so it runs
  from a clean clone (the NLA audit prefers the gitignored cache, falls back to
  `data/`).
- Audit the **headline result**, not just the easy structural counts. The NLA
  audit originally locked the geometry but not the round-trip faithfulness
  cosines that the whole arc rests on — a reviewer caught it; AUDIT 20-21 close
  it. When you add a figure or a claim, add its audit line in the same change.
- For qualitative decode claims, assert the **content** (a substring of the
  decoded text), not just non-emptiness (NLA AUDIT 17 asserts the decoded
  identities London/Spain/China).

**Be honest about what the audit does NOT catch:** it verifies arithmetic
consistency *given the captures* — not capture-protocol bugs, not interpretive
overreach, not whether a threshold was the right choice, and not that the prose
was transcribed faithfully into the script's expected constants (those are
maintained by hand). State this in the arc README. "N PASS" means "the numbers
agree," never "the methodology is right."

**Done when:** the audit passes from a clean clone and every load-bearing
number in the observations has a corresponding assertion.

### 6. Arc README synthesis

- **Findings as hypotheses** with explicit scope qualifications (see Framing).
- **Limitations** section, ranked by how far they constrain the claims.
- **Possible next paths**, each tied to a question and (if tracked) a ticket.
- **Attribution**: separate human direction-setting from AI implementation
  honestly.
- Cross-link: README → observations → figures/INVENTORY → data/MANIFEST.

### 7. PR

- Push the branch, open a PR (one arc = one scope-bounded diff). `git lfs pull`
  works for reviewers. Run `/ultrareview <PR#>` if warranted. Merge per the
  repo's manual-merge SOP.

---

## Raw data is a deliverable

The discipline the rest of this doc leans on, stated once, in full.

**Why.** Scripts + figures without the data they were generated from are not
reproducible and not verifiable. A reviewer can't check a figure against its
source; a future session can't re-run the audit; "trust me, it computed +0.87"
is not evidence. The figures and the audit are *downstream* of the data — ship
the data.

**What to commit.** All artifacts a figure or audit consumes:

- **Capture-roots** — produced by a run that loads a model (expensive,
  CPU-hours). Always commit; they are irreplaceable without re-running the
  experiment.
- **Derived** — produced cheaply from other `.pt` by a committed script.
  Commit these too: the marginal MB buys bit-exact figure + audit
  reproduction with zero model load. (Only consider shipping roots-only +
  a regeneration script if the derived set is genuinely large; if you do,
  `log`/document what was dropped — silent truncation reads as completeness.)

**Where.** `research/arcs/<slug>/data/`, git-LFS-tracked via the existing
`research/**/data/*.pt` rule. Keep your working/scratch captures in a
gitignored cache; the committed `data/` dir is the canonical copy.

**Wiring (so the data is *usable*, not just stored).** Scripts should resolve
inputs **cache-first, committed-copy-fallback**, and write outputs only to the
cache. The NLA arc centralizes this in `testing/examples/_nla_artifacts.py`
(`read_artifact`/`find_artifact` for loads, `write_artifact` for saves). That
one indirection is what lets the *same* script a developer runs locally
(writing fresh captures to the gitignored cache) also re-render figures and
replay the audit on a clean clone (reading the committed copy) — no manual copy
step, no clone-vs-local branching. Without it, committed data is inert: the
scripts still point at an empty cache.

**Manifest.** A `data/MANIFEST.json` (template generator:
`nla_data_manifest.py`) records per file: `filename`, `sha256`, `size_bytes`,
`class` (capture-root | derived), `producing_script`, `producing_command`,
`inputs` (upstream `.pt`), `requires_model` (none | base | +av/+ar/…),
`consumers` (figures / downstream artifacts / audit). The generator's `--check`
mode re-verifies every sha256 — run it in the audit step as a drift detector.

**Validate before you save.** "Save" includes confirming the data is *correct*
(protocol sanity, shapes, no NaNs/collapse) and *locked* (audit re-derives the
load-bearing numbers; manifest pins the bytes). A committed wrong dataset is
worse than none.

**Trust note.** `torch.load(..., weights_only=False)` executes pickle on load.
Fine for locally-generated tensor dumps; never normalize it for third-party
data. The manifest's sha256 lets a consumer verify integrity before loading.

---

## Framing discipline

The review's central rigor finding: late, careful scope-qualifications in the
arc README hadn't propagated back into the individual observation files, so the
most quotable sentences read as settled where the synthesis read as hypothesis.

- **State claims at their evidence level.** One anchor pair / one layer / one
  model → "candidate," "appears to," "for this configuration." Generalizing to
  a property of the model (or of transformers) requires the cross-condition
  scope test — name it as a follow-up rather than asserting the general claim.
- **Tag, don't launder.** `[INTUITION]` / `[ANALOGY]` / `[SPECULATION]` /
  `[CONTRADICTION]` per the repo `CLAUDE.md`. A hypothesis block (explicit
  "H1: …, to test: …") is the functional equivalent and keeps the speculation
  out of the findings prose.
- **Cite load-bearing external claims** (a paper or `theory/kb/` note).
  Inline arXiv/URL citation is the right register for `research/` (these are
  observations, not KB notes).
- **Propagate corrections everywhere.** When a number/location is refined
  (e.g. NLA's t=0.421 → plateau relocation), back-reference the refinement from
  every file that states the old value. Dated observations are snapshots, but a
  one-line "later refined, see X" pointer keeps a reader from taking a
  superseded number as current.
- **No emojis** (use `★ → • ─` or bold); **no placeholders** in any committed
  doc. (Repo `CLAUDE.md` + user memory.)

---

## Sessions are not findings

`sessions/` files are operational checkpoints (worktree path, branch tip,
"what's next") that go stale within hours. They are never load-bearing for a
claim. Don't rewrite old session snapshots to match current state — newer files
supersede older ones, and the README/INVENTORY/audit carry the durable record.

---

## New-arc checklist

```
[ ] worktree created; research question written in one sentence
[ ] capture run; protocol validated (layer/position/shapes/counts sane)
[ ] raw data saved to arcs/<slug>/data/ ; MANIFEST.json written; --check passes
[ ] derived artifacts scripted + in data/ + in manifest (class: derived)
[ ] figures generated by committed scripts; INVENTORY.md bijection complete
[ ] observations written (evidence-first, fields filled, nulls labeled as null)
[ ] audit script re-derives every load-bearing number incl. the headline;
        passes from a clean clone; "what it can't catch" stated in README
[ ] arc README: findings-as-hypotheses + limitations + next-paths + attribution
[ ] clean-clone test: git lfs pull → audit PASS → a figure re-renders
[ ] PR opened (one arc = one scope-bounded diff)
```
