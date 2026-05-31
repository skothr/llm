# Research

Experimental LLM-interpretability research on local open-source models
(TinyLlama, OpenLLaMA 3B, Qwen2.5-7B; inference via llama.cpp / HF
transformers). Work is organized into **arcs** — focused, multi-observation
investigations that each cohere around a single research question — plus a
landing bin for one-off findings and an archive for retired material.

## Arcs

| Arc | Status | Question | Entry point |
|---|---|---|---|
| **NLA verbalizer — Qwen2.5-7B layer 20** | Paused; synthesis written | What do Anthropic's released NLA verbalizer/reconstructor models surface about layer-20 hidden-state geometry? | [`arcs/nla-verbalizer/README.md`](arcs/nla-verbalizer/README.md) |

**Flagship: the NLA verbalizer arc.** A two-week investigation applying
Anthropic's released NLA model pair (`kitft/nla-qwen2.5-7b-L20-{av,ar}`) to
local Qwen2.5-7B-Instruct — two dozen dated observation files, 36 figures, a
129-PASS / 0-FAIL regression audit, and one working synthesis (*layer-20
h-space appears to have discrete attractor basins separated by sharp
boundaries*, held as a hypothesis, not a settled claim). Start there.

## Layout

```
research/
  README.md              ← this file (abstract index)
  arcs/<slug>/           ← one directory per investigation
    README.md            ← arc entry point: motivation, findings, caveats
    observations/        ← dated evidence-first writeups
      figures/           ← generated plots + INVENTORY.md provenance
    sessions/            ← session-resumption checkpoints (stale-fast)
    plans/               ← research / construction plans
  observations/          ← one-off findings not (yet) part of an arc
  archive/               ← retired / pre-arc material, kept for archaeology
```

A finding starts life in `observations/` (or, if it's already arc-scoped, in
the arc's `observations/`). When several loose observations cohere into an
investigation, promote them: `mkdir arcs/<slug>/`, move the files in, and
write the arc `README.md` that ties them together.

## Conventions

**Observations** — dated markdown, `YYYY-MM-DD-<slug>.md`, one finding per
file. Each carries: date + context (model, params), the finding, evidence
(output/transcript excerpts), reproducibility (exact commands), hypotheses,
follow-ups, references. The canonical format spec lives in the repo
`CLAUDE.md` under *# Research Observations*. No index file — scan by
filename. Evidence-first: numbers should be reproducible or audit-locked.

**Sessions** — LLM session checkpoints, *not* research findings. They capture
the operational state of a Claude Code session at a compaction or hand-off
boundary (worktree path, branch tip, audit-pass count, "what to do next"
pointers). Same `YYYY-MM-DD-<slug>.md` naming, with the slug carrying one of
`arc-summary` / `arc-resume` / `checkpoint` / `for-compact`. These go **stale
within hours or days** — read them as a snapshot at write-time, not as
current guidance; newer files supersede older ones. Nothing in `sessions/` is
load-bearing for a research claim, so a stale or deleted session file never
affects the correctness of an observation or figure.

**Plans** — research/construction plans (what to investigate, in what order,
with what caveats). Dated `YYYY-MM-DD-<slug>.md`. A plan in `Status:
preliminary` has run no experiments yet.

**Figures** — generated plots under an arc's `observations/figures/`, with an
`INVENTORY.md` giving per-figure provenance (which script + commit produced
each one). PNGs are tracked via **git-LFS** — run `git lfs install` before
working the repo, or figures show as phantom modifications. See the
`project-repo-lfs-rewrite` memory.

**Citations.** Load-bearing claims about LLM architecture / training /
interpretability cite a source — a paper key or a `theory/kb/` note — per the
discipline in the repo `CLAUDE.md` (*# Theory KB & citation discipline*).
Analogies and intuitions are tagged (`[INTUITION]`, `[ANALOGY]`,
`[SPECULATION]`) so they're never laundered as formal claims.
