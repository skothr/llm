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
| 1 | The modern Transformer is a small set of choices | done | 12/14 | ~60 of 80 |
| 2 | Training is a multi-stage pipeline | done | 11/14 | ~73 of 100 |
| 3 | Reasoning is compute, search, and verification | done | 12/14 | ~74 of 80 |
| 4 | The internal computation can be partially read | done | 11/14 | ~56 of 70 |
| 5 | What we measure and what slips through | done | 12/14 | ~70 of 70 |
|   |                                                  |        | **58/70** | **~333 of 400** |

### Sections completed (58)

- Paper 1: §1 §2 §3 §4 §5 §6 §7 §8 §9 §10 §11 §12
- Paper 2: §1 §3 §4 §5 §6 §7 §8 §9 §10 §11 §12
- Paper 3: §1 §2 §3 §4 §5 §6 §7 §8 §9 §10 §11 §12
- Paper 4: §1 §2 §3 §4 §5 §6 §7 §8 §9 §10 §11
- Paper 5: §1 §3 §4 §5 §6 §7 §8 §9 §10 §11 §12 §13

### Sections remaining (12) — Pass C wave 9 + 10

- Paper 1: §13 §14 (2 sections)
- Paper 2: §2 §13 §14 (3 sections)
- Paper 3: §13 §14 (2 sections)
- Paper 4: §12 §13 §14 (3 sections)
- Paper 5: §2 §14 (2 sections)

After all 70 sections land: bibliography sync, cross-ref resolution, build, polish.
