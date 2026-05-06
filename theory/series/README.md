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
LaTeX `pdflatex + bibtex + pdflatex × 2` cycle (no `latexmk` required):

```bash
cd theory/series/paper-1
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

PDFs ship in-repo under each paper directory; intermediate
`*.aux/.log/.bbl/.blg/.out/.toc` files are git-ignored.

## Built PDFs

| Paper | Title | PDF | Pages |
|------:|-------|-----|------:|
| 1 | The modern Transformer is a small set of choices | [`paper-1/main.pdf`](paper-1/main.pdf) | 73 |
| 2 | Training is a multi-stage pipeline | [`paper-2/main.pdf`](paper-2/main.pdf) | 98 |
| 3 | Reasoning is compute, search, and verification | [`paper-3/main.pdf`](paper-3/main.pdf) | 77 |
| 4 | The internal computation can be partially read | [`paper-4/main.pdf`](paper-4/main.pdf) | 72 |
| 5 | What we measure and what slips through | [`paper-5/main.pdf`](paper-5/main.pdf) | 77 |
|   | **Total** |   | **397** |

Build state as of 2026-05-06 (post-MINOR-sweep): paper-3 builds
clean; paper-4 carries the known `Citation 'l' undefined` warning
(a stray-key artifact whose source defies grep — it does not prevent
the PDF from building, and renders as a single `[?]` in the
bibliography on page 60); paper-1/2/5 carry small pre-existing
warnings (a few cross-paper `\cref` targets, a Unicode μ in one bib
title, a font-shape note) that do not block the build. Cross-paper
references throughout the series are inline-text rather than `\cref`
since each paper builds independently; xr-hyper would lift this if
ever desired.

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
