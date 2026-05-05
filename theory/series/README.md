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

Once `paper-N/main.tex` exists, build with:

```bash
cd theory/series/paper-1 && latexmk -pdf main.tex
```

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
