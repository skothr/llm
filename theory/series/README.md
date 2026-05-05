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

## Status

| Paper | Title | Outline | Plan | LaTeX |
|------:|-------|:-------:|:----:|:-----:|
| 1 | The modern Transformer is a small set of choices | pending | pending | pending |
| 2 | Training is a multi-stage pipeline | pending | pending | pending |
| 3 | Reasoning is compute, search, and verification | pending | pending | pending |
| 4 | The internal computation can be partially read | pending | pending | pending |
| 5 | What we measure and what slips through | pending | pending | pending |

(Status table updated as papers land.)
