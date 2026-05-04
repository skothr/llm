# KB build — final review (Phases 0-4)

_2026-05-04. Snapshot of the theory/ KB-substrate after Phases 0-4 of the
2026-05-03 expansion plan. Phases 5 (LaTeX series) and 2.5 (PDF deepening)
remain deferred — both have entry points captured below._

## Inventory

| Slice | Count | Notes |
|------|------:|------|
| Papers in `kb/index/papers.json` | 164 | 63 with `excerpts_file`, 155 with `notes_referenced_by` |
| Topic leaves | 54 | 9 areas; 38 draft + 16 stub |
| Excerpt files (`kb/excerpts/*.md`) | 63 | mostly abstract-level (see Phase 2.5 gap) |
| Synthesis notes (`kb/notes/*/*.md`) | 54 | one per leaf |
| Cross-area contradictions surfaced | 73 | 38 notes; reasoning + architecture lead |
| Glossary entries | ~425 | pre-existing + pilot + 9 area fragments |
| Timeline entries | pre-2017 → 2026-H1 | 41 unique cited keys, all resolve in `papers.json` |

## Phase outcomes

- **Phase 0** — directory restructure + CLAUDE.md citation discipline.
  Single-LaTeX archived at `theory/archive/2026-05-03-pre-expansion/`.
- **Phase 1** — landscape sweep across all 9 areas; 53-leaf v1 topic
  graph; corrected several stale priors (DeepSeek-V3 cost, FineWeb size,
  Muon FLOPs, R1 AIME numbers). Findings at
  `theory/plans/2026-05-03-landscape-sweep-findings.md`.
- **Phase 2** — pilot (`architecture/attention-mechanism.md`) + 9 area
  subagents in parallel; 54 notes (38 draft / 16 stub) and 58 new
  excerpts. Idempotent `merge.py` + `merge_glossary.py` consolidated
  partitioned outputs.
- **Phase 3** — chronological `kb/index/timeline.md`; every cited key
  resolves to `papers.json`.
- **Phase 4** — citation hygiene sweep (9 author-misattributions or
  topic-year placeholders renamed; 12 papers added: 9 verified + 3
  foundational fills); cross-area `kb/index/contradictions.md` lens.
- Lint (`theory/plans/_phase2/lint.py`) reports **0 errors / 0
  warnings**.

## Quality bar status

- **Citation discipline (CLAUDE.md rule):** every load-bearing technical
  claim either has a `[paper-key §X]` citation or carries an
  `[INTUITION]` / `[ANALOGY]` / `[CONTRADICTION]` / `[SPECULATION]` tag.
  Tags total: INTUITION=80, ANALOGY=43, CONTRADICTION=73, SPECULATION=11,
  FORUM-SIGNAL=10.
- **Feynman-bar layout** — pilot (`attention-mechanism.md`) is the
  template; subagent notes vary in adherence. The 38 draft notes hit
  formal-def → mechanism → variants → tagged-intuition → frontier
  consistently; depth varies.
- **Verifiable provenance** — pilot achieves the
  `[kb/excerpts/<key>#anchor]` dual-citation pattern (the spec ideal).
  Only 2 of 415 inline citations across the broader KB use this dual
  form — the rest cite paper-keys but not anchors. Captured as the
  Phase 2.5 follow-up.

## Known limitations (deferred work)

1. **Excerpt depth** — most excerpts are abstract-level rather than
   §/eq.-anchored. Sandbox blocked PDF fetches for several agents
   (alignment, post-training, evaluation, reasoning), so they extracted
   from arXiv abstracts. Phase 2.5 is the targeted re-pull-and-deepen
   pass.
2. **16 stubs** — placeholder note files exist for all topic leaves but
   16 (mostly inference and architecture sub-topics) still need first
   pass content.
3. **LaTeX series outline (Phase 5)** — explicitly deferred until the KB
   stabilizes. Spec at
   `theory/docs/superpowers/specs/2026-05-03-theory-expansion-design.md`.
4. **No external graph/render** — 73 contradictions are extracted into a
   single index file but not yet visualized. Could become a graph viz
   in the C++ GUI long-term, but not Phase-4 scope.
5. **Citation regex** — `lint.py`'s `PAPER_CITE_RE` only catches keys
   followed by `§`. Keys cited as `[key]` or `[key, arXiv ...]` bypass
   the lint. Phase 4's manual sweep was the first-pass correction; a
   stronger regex would be a useful tooling improvement.

## Trust the KB for what?

- ✅ **Topic landscape and lineage** — every note has primary-source
  citations and a frontier section; safe scaffolding for further
  research and paper writing.
- ✅ **Chronological progression** — `timeline.md` is fully linked.
- ✅ **Cross-area open questions** — `contradictions.md` is a curated
  surface, not interpretive synthesis; safe to read and cite.
- ⚠️ **§/eq.-level claims** — verify against the original PDF before
  propagating into LaTeX or experiments. The KB digests, the paper is
  canonical (this is the standing CLAUDE.md rule).
- ⚠️ **Author attributions on 2025-26 papers** — the reasoning subagent
  hallucinated 7 author prefixes that Phase 4 corrected; spot-check any
  unusually-named keys (no author prefix, or mismatched arXiv ID) before
  citing externally.

## Recommended next session

Pick whichever serves the next research goal:

- **(A) Phase 2.5 — excerpt deepening.** Re-pull missing PDFs to
  `theory/sources/papers/`, deepen abstract-level excerpts to
  §/eq.-anchored, backfill `[kb/excerpts/<key>#anchor]` citations
  across draft notes. Highest leverage for citation provenance.
- **(B) 16 stub fills.** Targeted draft passes on the stub notes;
  smaller in scope than the original Phase 2 fan-out and easy to
  parallelize via subagents.
- **(C) Phase 5 — LaTeX series outline.** Brainstorm paper-set shape
  using the now-stable KB as substrate.
