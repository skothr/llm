# theory/sources/

Primary sources for the KB.

## Layout

- `papers/` — PDF copies of canonical papers, named `{paper-key}_{slug}.pdf`.
  Registered in `kb/index/papers.json`. Source of truth.
- `forums/` — selectively archived blog/forum/thread snapshots. Used as
  discovery-path provenance, **never** as the sole backing for a hard claim.

## Source tiers

| Tier | Sources | Can solely back hard claims? | Storage |
|------|---------|----------------------------|---------|
| **A — Canonical** | arxiv preprints, peer-reviewed venues, official tech reports / model cards (Llama, Qwen, DeepSeek, GPT-4 system card, Claude/Gemini model cards), reference github repos | **Yes** | `papers/<key>.pdf` + `kb/excerpts/<key>.md` |
| **B — High-signal commentary** | Vendor research blogs (Anthropic / OpenAI / DeepMind / HF), respected lab blogs (EleutherAI / Databricks-MosaicML / Together / Anyscale), individual respected researchers (Dettmers, Rush, Alammar, Lilian Weng) | **Only if** the underlying tier-A source is also cited | Optional snapshot to `forums/` if uniquely informative |
| **C — Community signal** | r/LocalLLaMA, r/MachineLearning, HuggingFace community, X/Twitter (named researchers preferred), HN comment threads | **No** — discovery only | Selective: only when a thread materially shaped a synthesis. Tagged `[FORUM-SIGNAL]` for context, not authority |

## Forum/blog archival format

When a tier-B or tier-C source materially shapes a KB note, snapshot it
under `forums/`:

```markdown
---
url: https://reddit.com/r/LocalLLaMA/comments/...
captured: YYYY-MM-DD
informed-notes: [<area>/<topic>, ...]
tier: B | C
---

# [Thread/post title]

[author, date]

> [verbatim, blockquoted excerpt]

**Why captured:** Surfaced X claim that I then verified against [paper-key].
The thread is *not* the citation; the paper is. This is here for provenance
of the discovery path.
```

## Adding a new paper

1. Download PDF to `papers/<paper-key>_<slug>.pdf` (slug is filesystem-safe
   short title).
2. Add an entry to `kb/index/papers.json` with all schema fields. `topics`
   should align with leaves in `kb/index/topics.md`.
3. If the paper is going to be cited from a note, write
   `kb/excerpts/<paper-key>.md` with the relevant verbatim passages and
   set `excerpts_file` in `papers.json`.
