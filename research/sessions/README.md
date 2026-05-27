# Session-resumption metadata

Files in this directory are **LLM session checkpoints**, not research
findings. They capture the operational state of a Claude Code session at
compaction or hand-off boundaries — worktree path, branch tip, audit-pass
count, ticket inbox state, "what to do next" pointers.

Distinct from `research/observations/` (evidence-first writeups citing
audit-locked numbers): nothing here is load-bearing for a research claim.
A session file going stale or being deleted has no effect on the
correctness of any observation or figure.

## Naming

Same `YYYY-MM-DD-<slug>.md` convention as observations, but the slug
includes one of: `arc-summary`, `arc-resume`, `checkpoint`, `for-compact`.
Filenames make the kind explicit.

## Lifetime

These files become stale within hours or days of the session that wrote
them — commit hashes shift, ticket states change, the "next move" gets
done. Read them as a *snapshot at write-time*, not as up-to-date guidance.
Newer files supersede older ones; old session files are kept for
archaeology rather than active use.
