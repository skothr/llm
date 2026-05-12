# theory/series/ — LaTeX paper series (Shape C)

Five thesis papers covering modern LLM theory and engineering, drawn from
the now-stable KB at `theory/kb/`. Decision logged in
`theory/plans/2026-05-04-latex-series-brainstorm.md` (Shape C, recommended)
and `SHAPE-C-decision.md` here.

The previous single-LaTeX-doc deliverable (`llm-core-architecture`) is
archived under `theory/archive/2026-05-03-pre-expansion/`. Its `.tex` may
be partially reused.

## Layout

```
series/
├── README.md                       # this file
├── SHAPE-C-decision.md             # 5-paper structure + cross-cuts
├── preamble.tex                    # shared LaTeX setup (math, theorems, custom env)
├── references.bib                  # auto-generated from kb/index/papers.json
├── paper-1/                        # The modern Transformer is a small set of choices
│   ├── outline.md
│   ├── implementation-plan.md
│   ├── main.tex
│   └── sections/
├── paper-2/                        # Training is a multi-stage pipeline
│   ├── outline.md
│   ├── ...
├── paper-3/                        # Reasoning is compute, search, and verification
├── paper-4/                        # The internal computation can be partially read
└── paper-5/                        # What we measure and what slips through
```

## Build

Each `paper-N/main.tex` builds to `paper-N/main.pdf` via the standard
LaTeX `pdflatex + bibtex + pdflatex × 2` cycle (no `latexmk` required).

Before the first build (and any time `theory/kb/glossary.md` or body
sources change), regenerate the glossary artifacts from project root:

```bash
testing/.venv/bin/python theory/series/gen_glossary.py
testing/.venv/bin/python theory/series/mark_glossary_terms.py
```

Both scripts are idempotent. `gen_glossary.py` reads
`theory/kb/glossary.md` (the canonical source of truth) and writes
`theory/series/glossary-terms.json`. `mark_glossary_terms.py` reads
that JSON, wraps body-text glossary-term occurrences in
`\glsterm{key}{surface}` across each `paper-N/sections/*.tex`, and
emits per-paper `glossary-section.tex` (the rendered glossary) and
`glossary-tooltips.tex` (the short-def macro table). All generated
files are committed — drift between source and artifact would be a
real bug.

Then build any paper:

```bash
cd theory/series/paper-1
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

PDFs ship in-repo under each paper directory; intermediate
`*.aux/.log/.bbl/.blg/.out/.toc/.upa/.upb` files are git-ignored.

## Glossary feature

Every body-text occurrence of a glossary term (sourced from
`theory/kb/glossary.md`) is wrapped in a `\glsterm{key}{surface}`
macro that attaches:

- a clickable `\hyperlink` jumping to the per-paper Glossary section
  near the back (universal across PDF viewers — Acrobat, Chrome
  PDFium, Firefox, Preview, evince);
- a `\pdftooltip` whose hover text is the term's short definition
  (rendered in Acrobat / Foxit / evince; silent in browser-embedded
  viewers — degrades gracefully).

Author surface:

- **Most of the time, write nothing special** —
  `mark_glossary_terms.py` auto-detects body-text occurrences and
  wraps them in place.
- **Manual `\glsterm{key}{surface}`** when you want to wrap a
  non-canonical phrasing (e.g. `\glsterm{rope}{rotary positional
  embeddings}` instead of just `RoPE`). Already-wrapped sites are
  skipped by the auto-detection script.
- **`\nogls{...}`** is an escape hatch for false positives — the
  argument renders verbatim, but the auto-detection script will
  not wrap inside it. Use sparingly when the regex picks up an
  occurrence that shouldn't be linked.

Skip set (these sites are never auto-wrapped): inline + display
math, `\begin{lstlisting|verbatim|minted|alltt}` environments,
`\label{}` / `\ref{}` / `\cite*{}` brace args (including the
optional `[\S X]` arg of natbib `\citep[X]{}` / `\citet[X]{}`),
section / paragraph titles (`\section{}`, `\paragraph{}`, …),
`\caption{}` and `\item[]` label args (moving arguments —
`\pdftooltip` cannot appear there), `\hypertarget{}` / `\hyperref{}`
anchor args, comments, and already `\glsterm`- or `\nogls`-wrapped
regions.

`theory/kb/glossary.md` is the source of truth. Edit it; regenerate
`glossary-terms.json` with `gen_glossary.py`; re-run
`mark_glossary_terms.py`. Conflict-scan mode
(`mark_glossary_terms.py --conflicts`) reports glossary terms that
match nowhere in the corpus (likely typos / missing aliases) and
terms that over-match (likely too-generic regex).

## Built PDFs

| Paper | Title | PDF | Pages |
|------:|-------|-----|------:|
| 1 | The modern Transformer is a small set of choices | [`paper-1/main.pdf`](paper-1/main.pdf) | 79 |
| 2 | Training is a multi-stage pipeline | [`paper-2/main.pdf`](paper-2/main.pdf) | 96 |
| 3 | Reasoning is compute, search, and verification | [`paper-3/main.pdf`](paper-3/main.pdf) | 74 |
| 4 | The internal computation can be partially read | [`paper-4/main.pdf`](paper-4/main.pdf) | 69 |
| 5 | What we measure and what slips through | [`paper-5/main.pdf`](paper-5/main.pdf) | 83 |
|   | **Total** |   | **401** |

Build state as of 2026-05-11 (post glossary-feature build): every
paper produces a clean PDF; per-paper glossary sections are present.
Pre-existing source bugs in papers 2/3/4 emit non-fatal LaTeX errors
during build (`\footnotemark` inside `\caption{}`,
`\textit{``\$42$.$00$\$''}` mixed-mode dollar, `\to` inside
`\texttt{}`) — LaTeX recovers and produces a PDF, but these warrant
a separate cleanup pass. paper-4 also carries the known
`Citation 'l' undefined` warning (a stray-key artifact whose source
defies grep — renders as a single `[?]` in the bibliography). Cross-
paper references are inline-text rather than `\cref` since each
paper builds independently; xr-hyper would lift this if ever desired.

## Status (2026-05-05)

| Paper | Title | Outline | LaTeX sections | Pages drafted |
|------:|-------|:-------:|:--------------:|---------------:|
| 1 | The modern Transformer is a small set of choices | done | 14/14 | ~70 of 80 |
| 2 | Training is a multi-stage pipeline | done | 14/14 | ~88 of 100 |
| 3 | Reasoning is compute, search, and verification | done | 14/14 | ~83 of 80 |
| 4 | The internal computation can be partially read | done | 14/14 | ~69 of 70 |
| 5 | What we measure and what slips through | done | 14/14 | ~80 of 70 |
|   |                                                  |        | **70/70** | **~390 of 400** |

All section files are written. Remaining work is final-build:
regenerate `references.bib`, write each `paper-N/main.tex`,
resolve cross-paper `\cref` labels, run `latexmk -pdf` per paper.

After all 70 sections land: bibliography sync, cross-ref resolution, build, polish.
