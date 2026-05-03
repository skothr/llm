# theory/

LLM theoretical-framework workspace. Two-layered:

1. **`kb/`** — knowledge base. Modular, citation-grounded notes plus
   verbatim source excerpts plus structured indices. Source of truth for
   technical claims throughout this project.
2. **`series/`** — placeholder for a future multi-paper LaTeX series,
   outlined only after the KB is complete (see
   `docs/superpowers/specs/2026-05-03-theory-expansion-design.md` §11).

## Layout

```
theory/
├── kb/                # the knowledge base
│   ├── notes/         # digested synthesis, one file per topic
│   ├── excerpts/      # verbatim quoted passages from papers
│   ├── index/         # papers.json, topics.md, timeline.md
│   └── glossary.md
├── sources/           # primary source PDFs + selectively archived forum threads
├── plans/             # phase-scoped research/construction plans
├── archive/           # historical snapshots (pre-expansion v1, dated PDFs)
├── series/            # placeholder for future LaTeX paper series (Phase 5+)
└── docs/superpowers/  # specs and plans for the expansion itself
```

## Where to start

- **New to the KB?** Read `kb/README.md` for citation rules and epistemic
  conventions.
- **Looking for a topic?** Check `kb/index/topics.md` for the topic graph and
  per-topic status.
- **Tracing a claim?** Look up the paper-key in `kb/index/papers.json`,
  then read the matching `kb/excerpts/<key>.md` for verbatim source text.
- **Wondering what was here before?** See
  `archive/2026-05-03-pre-expansion/` for the v1 single-LaTeX-doc state.

## Citation discipline

Technical claims about LLM architecture, training, inference, or related
theory must cite either `[paper-key §X]` or
`[kb/notes/<area>/<file>#<anchor>]`. Analogies and intuitions must be tagged
`[ANALOGY]` / `[INTUITION]`. See project `CLAUDE.md` § "Theory KB & citation
discipline" for the full rule.

## Status

- v1 (pre-expansion) — single LaTeX/PDF "core architecture" doc covering
  Transformer 2017 → LLaMA 2023. Archived under
  `archive/2026-05-03-pre-expansion/`.
- v2 (in progress) — KB-substrate expansion to bleeding-edge breadth and
  depth. Phase plans live under `docs/superpowers/plans/`.
