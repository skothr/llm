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
| 1 | The modern Transformer is a small set of choices | done | 10/14 | ~50 of 80 |
| 2 | Training is a multi-stage pipeline | done | 7/14 | ~42 of 100 |
| 3 | Reasoning is compute, search, and verification | done | 6/14 | ~35 of 80 |
| 4 | The internal computation can be partially read | done | 4/14 | ~25 of 70 |
| 5 | What we measure and what slips through | done | 7/14 | ~43 of 70 |
|   |                                                  |        | **34/70** | **~195 of 400** |

### Sections completed (34)

- Paper 1: §3 §4 §5 §6 §7 §8 §9 §10 §11 §12
- Paper 2: §3 §6 §7 §8 §10 §11 §12
- Paper 3: §2 §3 §5 §9 §11 §12
- Paper 4: §3 §4 §7 §11
- Paper 5: §3 §4 §5 §6 §9 §10 §12

### Sections remaining (36)

- Paper 1: §1 §2 §13 §14 (4 sections)
- Paper 2: §1 §2 §4 §5 §9 §13 §14 (7 sections)
- Paper 3: §1 §4 §6 §7 §8 §10 §13 §14 (8 sections)
- Paper 4: §1 §2 §5 §6 §8 §9 §10 §12 §13 §14 (10 sections)
- Paper 5: §1 §2 §7 §8 §11 §13 §14 (7 sections)

After all 70 sections land: bibliography sync, cross-ref resolution, build, polish.
