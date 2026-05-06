# Glossary linking — design spec

**Date:** 2026-05-06
**Status:** approved (brainstorming complete; awaiting writing-plans handoff)
**Scope:** the 5-paper LaTeX series at `theory/series/` and the canonical glossary at `theory/kb/glossary.md`.

## 1. Motivation

The series carries 551 technical terms across ~397 pages of body text.
A reader entering a paper at random — particularly on an unfamiliar
acronym — has no convenient way to recover a term's definition without
scrolling/searching the glossary. The goal is to attach a clickable
hyperlink AND a hover tooltip to every body-text occurrence of every
glossary term, so the reader gets the definition *in place* without
losing reading position.

Priority ordering (per author, 2026-05-06):
1. Click-link to a glossary entry — universal across PDF viewers.
2. Hover tooltip with the short definition — premium where supported.
3. Citations to source papers — already work via `hyperref`.

## 2. Source of truth

`theory/kb/glossary.md` is canonical. It is hand-edited; nothing else
overwrites it. Format established by the existing file:

```
- **TermName** — Concise definition. Optional second sentence with
  context. `[paper-key §X]`
```

Variants the parser must handle:
- Plain: `- **RoPE** — Positional encoding ...`
- Parenthetical alias: `- **T5 (Text-to-Text Transfer Transformer)** — ...`
- Inline math in def: `- **Token Embedding** — ... ($W_e \in \mathbb{R}^{V \times d_{\text{model}}}$) ...`
- KB-cite anchor: trailing `` `[paper-key §X]` ``

The glossary's section structure (`## Models and Architectures`, etc.)
is preserved as `source_section` metadata but not used for term-resolution.

## 3. Architecture

### 3.1 Data flow

```
                 theory/kb/glossary.md  (canonical, hand-edited)
                            │
                            ▼
                  gen_glossary.py
                            │
                            ▼
                 glossary-terms.json
                 (term, key, aliases,
                  short-def, full-def,
                  case-strict, kb-cite)
                            │
                            ▼
              mark_glossary_terms.py
                  ┌─────────┬─────────┬─────────┐
                  ▼         ▼         ▼         ▼
       paper-N/        paper-N/         paper-N/         glossary-mentions.json
       sections/*.tex  glossary-        glossary-        (debug: which keys
       (in-place       section.tex      tooltips.tex     appeared in which
        rewrite,       (auto, per-      (auto, per-      paper)
        \glsterm{}{}-  paper, scoped    paper, short-
        wrapping)      to terms used    def macro table
                       in body)          for \pdftooltip)
```

Both scripts mirror the existing `gen_bibliography.py` design and live
at `theory/series/`. Generated artifacts (`glossary-terms.json` and
each `glossary-section.tex`) are **committed to git**, matching the
treatment of `references.bib`. The principle: drift between source
and artifact would be a real bug; gitignoring the artifacts would let
that drift go undetected.

### 3.2 Component contract

#### `gen_glossary.py`

- **Input:** `theory/kb/glossary.md`.
- **Output:** `theory/series/glossary-terms.json`.
- **Per-term record fields:** `key`, `primary_form`, `aliases[]`,
  `short_def` (first sentence, ≤140 chars; tooltip text),
  `full_def` (full markdown body; rendered into the per-paper glossary
  section), `case_strict` (bool from heuristic), `source_section`
  (markdown section), `kb_cite` (extracted backtick reference, may be
  null).
- **Key derivation:** lowercase + slugify the `primary_form`. Collisions
  get a numeric suffix (`-2`, `-3`).
- **Case-strict heuristic:** `True` if `primary_form` contains any
  uppercase letter past position 0 (covers `RoPE`, `GQA`, `FlashAttention`,
  `MoE`); `False` otherwise (covers `Tokenization`, `Embedding`).
  Per-term override allowed via an optional `case:` directive in the
  glossary source.
- **Alias extraction:** parenthetical contents in
  `**X (Y)**` are added as aliases when `Y` matches an acronym shape or
  starts with a capital letter. Conservative — false positives in the
  alias list cost more than missing aliases (which the user can add
  manually to glossary.md).
- **Failure modes:** duplicate key after slugification → exit 1 with
  both source-line numbers; malformed entry → exit 1 with line number.
- **Dependencies:** Python 3 stdlib only.

#### `mark_glossary_terms.py`

- **Input:** `theory/series/glossary-terms.json`,
  `theory/series/paper-{1..5}/sections/*.tex`,
  `theory/series/paper-{1..5}/glossary-section.tex` (when re-running for
  transitive hover inside glossary defs).
- **Output:** in-place rewrite of section files; emit/rewrite
  `paper-N/glossary-section.tex` per paper (scoped to terms whose body
  occurrences appeared in that paper); emit `paper-N/glossary-tooltips.tex`
  (per-paper short-def macro table for `\pdftooltip` lookup); emit
  `glossary-mentions.json` (debug artifact: which terms appeared in
  which paper).
- **Algorithm:**
  1. Build canonical regex from `glossary-terms.json`: alternation of
     all terms + aliases, sorted longest-first, with `\b...\b` anchors,
     case rule applied per term.
  2. Tokenize each input `.tex` file into "body" and "skip" regions.
     Skip set:
     - Math envs: `$...$`, `\(...\)`, `\[...\]`, `\begin{equation|align|gather|...}...\end{...}`.
     - Reference/cite/anchor commands' brace args: `\label{}`, `\ref{}`,
       `\cref{}`, `\eqref{}`, `\cite{}`, `\citep{}`, `\citet{}`,
       `\citeauthor{}`, `\input{}`, `\bibliography{}`, `\bibitem{}`.
     - Section-shaped commands' brace args: `\section{}`, `\subsection{}`,
       `\subsubsection{}`, `\paragraph{}`. (Skipped because a glossary
       hyperlink inside a section title fights with the section's own
       hypertarget anchor; clicking the TOC entry would jump to the
       glossary instead of the section.)
     - Already wrapped: `\glsterm{}{}` and `\nogls{}` brace args.
     - Comments: anything after `%` on a line.
     - **Captions are NOT skipped.** Wrapped on equal terms with body
       text. (User priority: random-access readability beats caption
       cleanliness.)
  3. Run regex substitution on body regions only. Replacement format:
     `\glsterm{<key>}{<matched-text>}` — surface form preserved
     verbatim.
  4. Track every replacement as `(key, paper_id, surface_form)`; record
     which terms were referenced in each paper.
  5. Write back the section file via atomic temp-file rename. Never
     partial-write.
  6. After all sections processed for a paper, render the per-paper
     `glossary-section.tex` containing only the terms with at least one
     wrap in that paper.
  7. Run the regex pass on the freshly-emitted `glossary-section.tex`
     itself, one level only — enables transitive hover inside glossary
     definitions (hovering "RoPE" inside the iRoPE definition surfaces
     RoPE's tooltip). Re-runs see the existing `\glsterm{}{}` and skip.
- **CLI flags:**
  - `--dry-run`: print intended changes; write nothing.
  - `--conflicts`: report likely false positives (terms matching in
    surprising contexts), false negatives (glossary terms with zero
    matches anywhere), regex over-matches (>100 hits in one paper),
    and stale wrapped occurrences (key no longer in glossary-terms.json).
  - `--paper N`: limit operations to paper N (fast iteration).
- **Failure modes:** unbalanced LaTeX braces or unclosed math envs in
  any input file → exit 1 with file+line; never write any output.
  Stale `\glsterm{key}{...}` whose key no longer exists in glossary →
  warn but leave as-is.
- **Dependencies:** Python 3 stdlib only.

#### `\glsterm{key}{surface}` macro — `theory/series/preamble.tex`

```latex
\usepackage{pdfcomment}  % requires TeXLive 2018+; project uses TL2022
\newcommand{\glsterm}[2]{%
  \pdftooltip{\hyperlink{glossary:#1}{#2}}{\@gls@def@#1}%
}
\newcommand{\nogls}[1]{#1}
```

The `\@gls@def@<key>{...}` table of short definitions is `\input`'d
from `glossary-tooltips.tex` (one entry per known glossary term;
generated alongside `glossary-section.tex`).

`\pdftooltip` requires `pdfcomment`. TeXLive 2018+ ships it; the
project's existing TeXLive 2022 install is confirmed compatible. No
new system dependency beyond a one-line `\usepackage`.

`\hyperlink` resolves via `hyperref`, already loaded.

`\nogls{}` is a no-op renderer whose sole purpose is to mark a
region the auto-wrap script must skip — the author's escape hatch
for false-positive sites.

#### Per-paper glossary section — `paper-N/glossary-section.tex`

Auto-generated. One file per paper. `\input`-ed from
`paper-N/main.tex` immediately before `\bibliography`. Structure:

```latex
% paper-N/glossary-section.tex — AUTO-GENERATED by mark_glossary_terms.py
% DO NOT EDIT BY HAND. Source of truth: theory/kb/glossary.md
%
\section*{Glossary}
\addcontentsline{toc}{section}{Glossary}
\label{sec:glossary}

\begin{description}
\item[\hypertarget{glossary:rope}{RoPE} (Rotary Position Embedding)]
Positional encoding that rotates query and key vectors by
position-dependent angles, ... \citep[\S 3.4]{su2021}.
\end{description}
```

The `\hypertarget{glossary:rope}{RoPE}` is the anchor each body
`\hyperlink{glossary:rope}` resolves to. Standard `hyperref` pattern.
Each paper's glossary section is scoped to terms used in that paper's
body — paper-1's glossary doesn't list paper-4-only terms.

### 3.3 Tooltip definitions table — `paper-N/glossary-tooltips.tex`

Generated per paper alongside `glossary-section.tex`. Defines per-term
short-definition macros consumed by `\pdftooltip`:

```latex
% paper-N/glossary-tooltips.tex — AUTO-GENERATED by mark_glossary_terms.py
\def\@gls@def@rope{Positional encoding that rotates query and key
vectors by position-dependent angles, making attention dot products
depend on relative position.}
\def\@gls@def@gqa{Interpolation between MHA and MQA: query heads are
partitioned into groups, each group shares one $W^K, W^V$.}
% ... only entries for terms actually used in this paper's body
```

`\input`'d from each paper's `main.tex` (not from `preamble.tex`).
Per-paper scoping keeps each PDF's bundle small and means the tooltip
table is co-located with the glossary section it serves.

### 3.4 Build pipeline integration

Author workflow when glossary.md or body text changes:

```bash
python3 theory/series/gen_glossary.py
python3 theory/series/mark_glossary_terms.py
# then standard:
cd theory/series/paper-N
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Both Python scripts are idempotent — repeated runs produce the same
output. Documented in `theory/series/README.md`'s Build section.

## 4. Edge-case decisions

| Case | Decision | Rationale |
|---|---|---|
| `iRoPE` containing `RoPE` substring | No false match | `\b` anchor doesn't fire between word chars; `\bRoPE\b` won't match inside `iRoPE`. |
| `FA-3` containing `FA` substring | Longest-match wins | Regex alternation is sorted longest-first, so `FA-3` matches before `FA` is tried. |
| Sentence-medial lowercase form of mixed-case term | No match | Acronyms with internal uppercase (`RoPE`, `GQA`) are case-strict. |
| Sentence-medial lowercase form of single-cap term | Match | `Tokenization`/`tokenization` both link (case-insensitive heuristic). |
| Plurals (`RoPEs`, `tokens`) | No match unless aliased | Author adds explicit `aliases:` entry in glossary.md when wanted. |
| Possessives (`RoPE's`) | No match | Apostrophe-s creates new word; not auto-linked. |
| Glossary term inside math (`$\text{RoPE}$`) | Skipped | Math envs are skip regions. |
| Glossary term inside `\label{}`, `\citep{}` | Skipped | Reference/cite command args are skip regions. |
| Glossary term inside `\section{}` title | Skipped | Section anchor / glossary anchor would conflict. |
| Glossary term inside `\caption{}` | Wrapped | Captions are reading surfaces; random-access value beats visual cleanliness. |
| Glossary term inside another glossary entry's def | Wrapped | Transitive hover. The mark script runs over `glossary-section.tex` itself. |
| Manual `\glsterm{rope}{rotary embeddings}` | Honored as-is | Already-wrapped sites are skip regions. |
| Term appears only in skip contexts in paper-1 | Not in paper-1 glossary | Per-paper scoping is by *body* mentions only. |
| Single-letter term (`Q`, `K`, `V`) | Not auto-linked | 2-char minimum on regex; would catastrophically false-positive. Glossary section still lists them; body is plain. |
| Cross-paper duplication | Each paper gets its own glossary entry | `\hyperlink` only resolves intra-document; per-paper independent build. Slightly redundant but reading-correct. |

## 5. Testing strategy

**Tier 1 — pytest unit tests** at `theory/series/tests/test_glossary.py`:
- All 3 entry-shape parses (plain, parenthetical-alias, math-in-def).
- Skip-region tokenization (math envs, command args, `\nogls`,
  already-wrapped).
- Regex collision behaviors (FA vs FA-3, RoPE vs iRoPE).
- Plural / possessive non-matching.
- Idempotency: run-twice produces same output as run-once.
- Atomic-write recovery: process killed mid-write leaves source intact.

Approximate ~30-50 tests, ~250 LOC. New test directory; this is the
first test harness for the theory workspace.

**Tier 2 — `--conflicts` dry-run report** on the real glossary +
papers. Pre-merge gate. Reports:
- Glossary terms with zero body matches anywhere → likely typo or
  missing alias.
- Terms with >100 body matches → likely over-broad regex; review.
- Multi-term overlap at one position → resolved longest-first; report
  for verification.
- Stale wrapped occurrences (key no longer in glossary).

**Tier 3 — full LaTeX build verification.** After running mark, rebuild
all 5 papers from clean. Check:
- Page count delta is bounded (per-paper glossary section adds 1-3 pp;
  total +5-10 pp).
- Zero new `Reference … undefined` warnings (every `\hyperlink`
  resolves to a `\hypertarget`).
- Zero `\glsterm`-related TeX errors.

**Tier 4 — visual spot-check.** One paper, sampled body region,
rendered in:
- A tooltip-supporting viewer (Acrobat, Foxit, or evince): hover
  surfaces tooltip; click jumps to glossary section.
- A non-tooltip viewer (Chrome PDFium, macOS Preview): tooltip is
  silent; click still jumps correctly.

## 6. Rollout — five sequential commits on a feature branch

1. **`feat(theory): glossary scaffolding`** — preamble macros +
   `pdfcomment` package. Verifies build still clean before any body
   wraps land.
2. **`feat(theory): glossary parser`** — `gen_glossary.py`, parser
   tests, generated `glossary-terms.json` committed.
3. **`feat(theory): body-text glossary wrapper`** —
   `mark_glossary_terms.py` with `--dry-run` and `--conflicts` flags;
   wrapper / tokenizer tests. `--conflicts` clean.
4. **`feat(theory): wrap body terms across all 5 papers`** — apply
   mark in non-dry mode, commit rewritten `paper-N/sections/*.tex`,
   `paper-N/glossary-section.tex`, `paper-N/glossary-tooltips.tex`,
   updated `paper-N/main.tex`. Rebuild all 5 papers; verify Tier 3
   checks pass; visual spot-check.
5. **`docs(theory): build pipeline + glossary feature`** — README
   updates documenting the two-script pre-build invocation and the
   `\glsterm` / `\nogls` author surface.

Bisect-friendly. Each commit produces a clean buildable state. Single
PR or sequential master commits — author's call.

## 7. Risk register

| Risk | Mitigation |
|---|---|
| `pdfcomment` ↔ `hyperref` interaction (both manage PDF annotations) | Load `pdfcomment` after `hyperref` (current preamble order does this). First build run reveals any conflict. Risk is well-understood; `pdfcomment` documents this ordering requirement. |
| TeXLive version drift on author machine | TeXLive 2018+ ships compatible `pdfcomment`. Project's TL2022 install confirmed compatible. Document minimum version in README. |
| Author writes a custom surface form like `\glsterm{rope}{rotary positional embeddings}` and is surprised by behavior | README explicitly notes the surface arg is purely cosmetic; the key arg is what links. |
| Mark-script performance on the full tree | ~70 section files × few KB each; sub-second per file. Total <10s. Not a real concern. |
| `--conflicts` report identifies overly-permissive aliases | Iteration: tighten regex / aliases until clean before merging step 4. |
| Page-count growth eats reading budget | Per-paper glossary scoped to body-referenced terms only. Expected +1-3 pp per paper, +5-10 pp total — acceptable for the navigation gain. |

## 8. Out of scope

- **Reverse flow:** body text never updates `glossary.md`. Canonical
  direction only.
- **Semantic-similarity matching:** `tokeniz**er**` is a different
  lexical form from `tokenization`; not auto-linked unless explicitly
  aliased.
- **CI gating:** the scripts run on demand by the author.
  Promotable to a pre-commit hook later if drift becomes a problem.
- **Section/paragraph titles** as wrap targets: skipped permanently
  (link-conflict reason), not deferred.
- **Back-references** in glossary entries (`RoPE used in §3, §5,
  §11`): would require additional `hyperref` machinery; deferred to a
  future iteration if the author requests it.
- **xr-hyper integration** for cross-paper `\cref`: orthogonal to this
  spec (Track D in the deferred backlog). Composes nicely later but
  doesn't depend on this work.

## 9. Open questions for the implementation phase

These are not blockers for design approval; they are decisions the
writing-plans phase will surface and the implementation phase will
make. Author's preference noted alongside each.

1. **Conflict-scan report format.** Default human-readable terminal
   output; expose `--conflicts --json` for machine parse. Decided in
   writing-plans.
2. **Test-fixture style for the LaTeX tokenizer.** Inline strings for
   the unit tests where the fixture is small; small `.tex` fixture
   files for full-document edge-case scenarios. Decided in writing-plans.
3. **Tier-table documentation surface.** No additional mention in
   `theory/sources/README.md`. The glossary feature is a
   workspace-internal navigation aid, not a source tier.
