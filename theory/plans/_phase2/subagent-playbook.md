# Phase 2 area-subagent playbook

You are an area subagent for the LLM theory KB at `/home/ai/ai-projects/llm/theory/`. Your job: produce KB excerpts + synthesis notes for **all topics in your assigned area**, replicating the quality of the pilot note `theory/kb/notes/architecture/attention-mechanism.md`.

You MUST read the pilot note and at least 2 of the pilot excerpts before writing anything — they are your concrete quality bar.

## What "your area" means

Your assignment names an area like `architecture/`, `training/`, `inference/`, etc. The leaf topics in your area are listed in `theory/kb/index/topics.md` (read it). For each leaf topic, you produce:

1. A note file at `theory/kb/notes/<area>/<topic>.md`.
2. Excerpt files at `theory/kb/excerpts/<paper-key>.md` for the **2–5 primary papers** the note cites most heavily.
3. Per-area additions JSON at `theory/kb/index/_phase2-additions/<area>.json` (see schema below) — the orchestrator will merge this into `papers.json`.

## Priority and scope budget

You will not finish every topic at full draft quality. Triage:

- **DRAFT (full Feynman-bar note):** 3–5 highest-leverage topics in your area. Pick by current relevance to frontier LLMs. The Phase 1 landscape report for your area (under `theory/plans/landscape-sweep/`) is your guide.
- **STUB (skeleton):** the remaining topics. Stub format below.

If you run out of budget, **prefer fewer high-quality drafts over many shallow notes**. The orchestrator will spawn follow-up agents to fill stubs.

## Note structure (Feynman bar) — match the pilot

Every DRAFT note follows the structure of `theory/kb/notes/architecture/attention-mechanism.md`:

1. **Frontmatter** — `topic`, `status: draft`, `last_updated: 2026-05-04`, `primary_sources` (paper-keys), `related_topics`.
2. **Formal definition** — math + variables defined underneath; cite paper-key §X Eq.Y. Equations transcribed in LaTeX.
3. **Mechanism** — step-by-step computation with tensor shapes where applicable. For non-architectural topics (training, eval, etc.), the equivalent is "the protocol / procedure / experimental setup."
4. **Variants and lineage** — cited list, ideally a comparison table when multiple variants exist (cf. the pilot's KV-sharing axis table).
5. **`[INTUITION]` / `[ANALOGY]` paragraphs** — every analogy must return to canonical symbolic form. Tag every one. Never assert an analogy as a fact.
6. **Frontier and open questions** — what's contested, what's recent, what's unresolved. Use `[CONTRADICTION]` where sources disagree.
7. **See also** — links to `kb/notes/<area>/<topic>.md` for adjacent topics.

Citation rule: **every load-bearing claim cites a paper-key §X Eq.Y AND a `kb/excerpts/<paper-key>#<anchor>` link** to the verbatim quote. Two-citation pattern. The pilot is the example.

Tags you may use:
- `[ANALOGY]` — must return to canonical symbolic form
- `[INTUITION]` — informal explanation
- `[CONTRADICTION]` — sources disagree, name them
- `[FORUM-SIGNAL]` — Tier C, used for discovery only, never as sole citation
- `[SPECULATION]` — your own extrapolation; clearly hedged

## Stub format (for lower-priority topics)

```markdown
---
topic: <area>/<topic>
status: stub
last_updated: 2026-05-04
maintainer: theory-kb
primary_sources_planned:
  - <paper-key>
  - <paper-key>
---

# <Topic title>

**Status:** stub. Drafted from Phase 1 landscape sweep; needs full Phase 2 treatment.

## What this topic covers

<2–3 sentence synopsis from Phase 1 sweep, naming the key papers and the
core mechanism/idea>

## Primary sources to read (in order)

1. `<paper-key>` — `<title>` (`<arxiv-id>`) — <one-line "what this paper is for">
2. ...

## Key claims to ground (Phase 2 todo)

- <claim 1>
- <claim 2>

## Related notes

- `kb/notes/<area>/<topic>.md`
```

## Excerpt files

For each primary paper that you cite heavily (rule of thumb: cited 3+ times in your draft notes), create `theory/kb/excerpts/<paper-key>.md` with verbatim quoted passages anchored to PDF section/equation numbers. Templates:

- See `theory/kb/excerpts/vaswani2017.md` (foundational paper)
- See `theory/kb/excerpts/dao2022.md` (technical paper with theorems/algorithms)
- See `theory/kb/excerpts/deepseek-v2.md` (recent tech report with many equations)

Anchor convention: `#sec-X-Y-Z` for section X.Y.Z; `#abstract` for abstract; `#fig-N` for figures. Use stable kebab-case anchors so citations like `[kb/excerpts/dao2022#sec-3-1]` resolve.

## Acquiring primary PDFs

Most papers in your area are NOT yet downloaded. Workflow:

1. Identify candidate papers from your area's Phase 1 landscape report.
2. For each paper you'll cite heavily, check whether it's in `theory/kb/index/papers.json` (read the file; search by key).
3. If the PDF is not present in `theory/sources/papers/`, download it from arxiv:
   ```
   curl -sL -o theory/sources/papers/<paper-key>_<short-slug>.pdf "https://arxiv.org/pdf/<arxiv-id>"
   ```
4. Read only the load-bearing sections of the PDF (with Read tool, `pages` param). Do not try to read entire papers; pick sections that ground the claims you'll make.
5. For tier B/C sources (vendor blogs, named researchers, Reddit), do NOT download — cite the URL and label `[FORUM-SIGNAL]` or treat as discovery-only. Tier B/C never solely back a hard claim.

## Per-area additions JSON

The orchestrator will merge your additions to `papers.json`. Write to `theory/kb/index/_phase2-additions/<area>.json` with this exact schema:

```json
{
  "area": "<area>",
  "new_papers": [
    {
      "key": "<paper-key>",
      "title": "...",
      "authors": "...",
      "year": 2024,
      "venue": "...",
      "url": "https://arxiv.org/abs/...",
      "local_file": "sources/papers/<paper-key>_<slug>.pdf",
      "excerpts_file": "kb/excerpts/<paper-key>.md",
      "notes_referenced_by": ["<area>/<topic>", ...],
      "topics": ["<topic>", ...],
      "category": "foundational | landmark | recent | tech-report | superseded",
      "summary": "1–2 sentence summary"
    }
  ],
  "existing_paper_updates": [
    {"key": "<existing-paper-key>", "set_excerpts_file": "kb/excerpts/<key>.md", "add_notes_referenced_by": ["<area>/<topic>"]}
  ]
}
```

Do NOT edit `theory/kb/index/papers.json` directly — orchestrator merges to avoid conflicts.

## Topics.md status updates

Do NOT edit `theory/kb/index/topics.md`. The orchestrator will update statuses based on your final report.

## Glossary updates

Do NOT edit `theory/kb/glossary.md` directly (parallel subagents will conflict). Instead, write your glossary additions to a fragment file at:

```
theory/kb/index/_phase2-additions/<area>-glossary.md
```

Format the fragment as one or more `## Section heading` blocks, each followed by `- **Term** — definition. citation.` lines. The orchestrator will merge fragments into the master glossary, resolving section overlaps.

Each new term gets a 1–3 sentence definition + a paper-key citation. Use the pilot's glossary additions (in `theory/kb/glossary.md`, search for "Attention variants and KV-cache compression" and "Attention implementation (hardware)") as the format model.

## Tool discipline (mandatory — these are the user's standing rules)

- Use `Read` for file contents (NOT `Bash(cat / head / tail)`).
- Use `Edit` / `Write` for file changes (NOT `Bash(sed / awk / echo >)`).
- For broad codebase questions, use the Explore subagent. **You may not spawn further subagents from inside a subagent** — do it yourself or skip.
- For git ops in other directories: `git -C <path> ...` (NOT `cd <path> && git ...`).
- Avoid unnecessary command chaining; one focused command per Bash call is preferred.
- For curl/arxiv downloads: 1 command per paper is fine; do not loop.

## Final report (what you return)

Return a brief summary (~250 words max) with these sections:

1. **Topics covered:** list `<topic>: status (draft|stub)`
2. **Excerpts written:** list of files
3. **Notes written:** list of files
4. **New papers added:** count + list of keys
5. **Glossary additions:** count + section name(s)
6. **Open questions / followups:** anything you couldn't resolve
7. **Stale-prior corrections:** any case where the Phase 1 sweep or v1 taxonomy needs updating based on what you found

The orchestrator will use this report to update topics.md statuses and merge `_phase2-additions/<area>.json` into `papers.json`.
