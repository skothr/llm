# Restart prompt — paste this after `/clear`

This is the prompt to feed Claude after `/clear` if you want to resume the
theory/series work where the prior session left off (commits up to
`4d6c354` post-review fix-up, plus the kb-build-review plan update).

---

## Prompt to paste

```
Resume the theory/series LaTeX paper work where the prior session left off.

Read first to ground yourself:
1. theory/series/README.md — current build status table (397pp across 5 PDFs).
2. theory/plans/2026-05-04-kb-build-review.md — full multi-phase history through
   the post-review fix-up wave at the end.
3. theory/reviews/00-summary.md — the 6-reviewer audit triage.
4. git log --oneline -25 to see the 12+ commits from the prior session.

State at handoff (commit 4d6c354 + kb-build-review plan update):

* All 70 sections written; all 5 PDFs build clean from fresh tree
  (`pdflatex + bibtex + pdflatex × 2` per paper, no latexmk needed).
  Page counts: 73 / 98 / 77 / 72 / 77 = 397 pages total.
* 6-reviewer audit complete; reports at theory/reviews/01-…06-….md.
* The single BLOCKER + 8 of 11 MAJORs from review fixed in commit 4d6c354.
* Bibliography is 206 entries; gen_bibliography.py converts comma-separated
  authors to ` and ` separators and escapes `&` in titles.
* Cross-paper references are inline-text (not \cref), since each paper
  builds independently.
* Paper-4 has one residual `Citation 'l' undefined` warning — untraceable
  via grep; doesn't block PDF; renders as [?] in the bibliography.

What's deferred to this session:

A) MINOR sweep (~30 min, single batched fix). Source: theory/reviews/00-summary.md
   "MINOR" section. The biggest items:
   - Paper 4 notation drift from canonical \BSH / \BHSdh shape macros
     (Paper 1 sets the convention; Paper 4 drifted to lowercase $h$).
   - `\begin{speculation}` environment defined in preamble but never used;
     ~12 [SPECULATION] markers in KB notes never landed as LaTeX callouts.
     Tag-coverage gap to close.
   - forumsignal undercount: only 9 across 70 sections; nostalgebraist 2020
     and Nanda 2023 attribution-patching should be tagged in body sections,
     not just the §14 contradictions.
   - Snell θ* subscript drift between paper-3 §4 intro and Eq. 1.
   - PUCT formula labeled "UCT" in paper-3 §6 (rStar-Math uses PUCT with prior;
     pure UCT does not).
   - Paper 3 §13 o-series RLVR cell should soften from positive assertion
     to "claimed in the system card; not externally reproduced."

B) M5 re-check (optional). The math-correctness reviewer flagged that
   paper-3 §7's GRPO equation cites `deepseek-r1` for the token-level form
   that's actually from `shao2024`. The fix-up wave determined the
   equation honestly dual-cites both. Re-read once more to be sure.

C) Phase 2.5 (optional polish, ~1-2 sessions). ~26 KB excerpts at
   `theory/kb/excerpts/<key>.md` are still abstract-only rather than
   §/Eq.-anchored. Local PDFs are on disk for most. A re-deepening pass
   drops `\deepencite` density in the LaTeX series.

D) xr-hyper for cross-paper \cref (~1 hour). Currently the ~10
   inline-text "Paper~2 §RLHF" mentions could become clickable PDF links
   via the xr-hyper package. Edit each paper-N/main.tex to load xr-hyper
   and \externaldocument the others; then convert the inline-text refs.

E) The `Citation 'l' undefined` mystery in paper-4 (low priority).
   Renders as [?] in the bibliography on page 60 of paper-4/main.pdf.
   Suspected stray bib-key fragment somewhere; my grep failed to find it.

Recommended order: (A) is the highest-leverage cleanup; (D) makes the
PDFs feel more polished without source rewrites; (C) is the deepest
content-quality investment.

Operational rules I should keep:
- ≤6 parallel opus subagents per dispatch wave (orchestrator crashes
  with 10-12).
- Each section file is atomic via the leaf agent's single Write call;
  re-dispatching with the same prompt is content-equal if a leaf is
  killed mid-flight.
- Append-only on theory/kb/index/papers.json (don't sort).
- Don't touch CLAUDE.md, learn/, pyproject.toml, testing/gui_cpp/.
- For substantive changes, edit theory/series/paper-N/sections/*.tex,
  rebuild that paper, keep theory/series/README.md page-count table
  in sync.

Ready when you are — pick a track and let's go.
```

---

## What's NOT in this prompt (lives in repo or memory)

- The full review wave — sits at `theory/reviews/00-…06-…md`.
- Operational lessons from the prior session — sits in
  `~/.claude/projects/-home-ai-ai-projects-llm/memory/project_theory_kb.md`.
- Phase-by-phase build history — sits in `theory/plans/2026-05-04-kb-build-review.md`.
- The frontier-currency reviewer's report (which carries the Nature DeepSeek-R1
  citation, the Mistral Large 3 finding, and the OpenAI/Apollo deliberative-alignment
  paper details) — sits at `theory/reviews/05-frontier-currency.md`.

The fresh Claude session reading the prompt above will Read each of these and
have full context for continuing.
