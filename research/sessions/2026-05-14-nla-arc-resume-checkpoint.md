# NLA Research Arc — Resume Checkpoint (2026-05-15)

**Branch:** `session/nla-research`
**Worktree:** `/home/ai/ai-projects/llm/.claude/worktrees/nla-research`
**Tip commit (pre-PR-rebase):** `8b79cd4` — rebased onto current origin/master before PR. Post-rebase hashes have shifted; this checkpoint's hash table below references the pre-rebase chain on backup branch `session/nla-research-backup`. Diff vs origin/master after rebase: 94 files, +11976 / -1.
**Working tree:** clean (only gitignored `testing/.cache` symlink visible)
**Audit state:** `nla_audit_findings.py` passes 129/0 (extended through MAIN-71)

## Strongest synthesis from the arc

**Layer-20 h-space appears to have discrete attractor basins separated by sharp boundaries** — *demonstrated for one anchor pair (factual/geography ↔ poetic/nature) and not yet replicated across other content domains; treat as a working hypothesis rather than a settled property of layer 20.*
Basins (in the cases probed) include both named vocab categories AND hybrid combinations not
in the original atlas (e.g. the "Definition + Poem" plateau between
factual and poetic anchors). AR re-encoding returns h to the basin's
directional region — basins are direction-coupled, not magnitude-coupled.
Linear interpolation between basin-residing h's traverses one or more
intermediate basins with sharp boundaries between them. This explains
MAIN-48 (arithmetic = basins not offsets), MAIN-44/70 (protocol-specific
basin subsets), MAIN-25 (sharp boundaries), MAIN-34 (basins have non-zero
volume), MAIN-71 (basin attractor strength empirically demonstrated *at this one plateau, with margin +0.061 over the nearest anchor — narrower than the +0.25 margins anchors have to each other; "basin" framing is supported but the basin is shallow*).

**Open scope-test follow-ups before generalizing**: replicate dense interpolation on at least 2-3 additional anchor pairs (e.g. code↔nature, math↔emotion, factual↔refusal); bootstrap the plateau round-trip against multiple in-zone t values to establish a noise baseline for the +0.061 margin; check whether the sharp-boundary behavior survives at finer resolutions than Δt=0.0025.

## Research direction — user-shaped themes (added 2026-05-30)

The arc was driven by nine research-direction themes the human collaborator
(Michael Lannum) introduced in specific working-session turns. The canonical
list with verbatim transcript quotes lives in
[`../README.md#research-direction-user-shaped-themes`](../README.md#research-direction--user-shaped-themes);
short reference for resume-time orientation:

1. **Test Anthropic's NLA technique on open-source models** — session-start framing; "models" plural implicitly scoped beyond Qwen alone.
2. **Plumbing-first, depth-per-token** — validate round-trip on simple inputs; verbalize at every token, not just aggregates.
3. **Reproduce Anthropic's emergent-behavior examples** — rabbit poem, ethics/eval-aware behavior.
4. **Counterfactual / OOD probing** — feed text the model wouldn't naturally generate.
5. **Architecture / scope curiosity** — AR direction (text→h) alongside AV.
6. **Concept-direction extraction** — feature vectors that *mean* an abstract idea (country-ness).
7. **Semantic-basis grid** — wide lay of the embedding-space landscape; complex grid / entangled axes.
8. **Visualization as research, not presentation** — novel viz of feature/embedding/NLA interpretability.
9. **AV-decoder format-bias observation** — flagged ("Structured format..." weirdly consistent), not yet investigated.

The arc's findings (F1-F5 in `../README.md`) emerged at the intersection of
these direction themes and the agent-side implementation work. See
[`../README.md#a-note-on-the-collaboration-mode`](../README.md#a-note-on-the-collaboration-mode)
for the explicit human/AI role separation.

## Session-of-2026-05-14/15 arc summary

5 tickets closed (MAIN-44, 70, 48, 34, 71), 5 commits, 7 new figures
(fig31-37), 5 observation files, audit 93 → 129 PASS:

| commit | ticket | finding |
|---|---|---|
| `3c9acec` | MAIN-44 | mid-seq vocab atlas null result; refines MAIN-26 |
| `6125cb2` | MAIN-70 | basis is protocol-coupled by construction; per-category axis stability mapped |
| `a9a2dd7` | MAIN-48 | arithmetic = categorical not algebraic (specific-identity analogies fail) |
| `7a10001` | MAIN-34 | dense interp reveals "Definition + Poem" hybrid plateau (3-region geometry) |
| `8b79cd4` | MAIN-71 | plateau IS a true attractor basin (round-trip cos +0.90) |

Supersedes the earlier resume doc `2026-05-13-nla-arc-summary-for-compact.md` (written at commit `c49a1d9`, before Path B, vocab atlas, discriminant validation, MAIN-46, MAIN-47, and figure cleanup landed).

## Quick orientation

1. `cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research`
2. Confirm in worktree: `git rev-parse --abbrev-ref HEAD` should say `session/nla-research`
3. Read [`figures/INVENTORY.md`](../observations/figures/INVENTORY.md) for the figure catalog with provenance per figure
4. Run audit to verify state: `PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python testing/examples/nla_audit_findings.py` — expect 129/0 (was 93/0 at the time this checkpoint was written; suite was extended through AUDIT 19 in the final session-of-2026-05-15 work)

## What landed since the prior resume

14 new commits across these themes:

### Geometric structure deep dive (commits `f17a47f`, `d4960ac`)
- 167-capture inventory of geometric features (norms, kurtosis, top-k components, sparsity)
- Identified the 7 sink dims (universal, sign-locked, +0.22 cosine offset)
- 5-character dim classification (sink / polarized / feature / rare-burst / background)
- Found "kurtosis predicts AR-cos floor" relationship (slope 0.063)
- **Sink-removed atlas correction**: removing sinks does NOT rotate PCA layout (variance fraction 16.5%→15.3%). What sinks DO is add a constant +0.22 offset to every pairwise cosine.

### Glyph primitives
- **dim-indexed signature glyph** (fig10, fig11) — 8 rays = 8 feature dims, length = `|h[dim]|`, color = sign
- **anchor-projection glyph centroid version** (fig23, fig24) — **broken**: all 23 axes active simultaneously because centroids are 0.85+ mutually correlated
- **discriminant-direction glyph** (fig25, fig26) — fixed: use `mean(cat) − mean(non-cat)` instead of centroids; mean cross-axis cosine drops from +0.85 to +0.006. **This is the right primitive.**

### Path B — interpolation flipbook (commit `c2d4525`)
- AR-encoded two NL anchors → linearly interpolated h-vectors at 20 steps → AV-decoded each
- **The biggest finding of the arc**: linear h-space interpolation produces **stepwise semantic transitions** at exactly t=0.421. Geometric step size constant (`||Δh||` = 2.734); AV-text format word and nearest-vocab-anchor BOTH flip simultaneously within one 5% step.
- Path A → Path B established the viz pipeline that the rest of the arc builds on.

### Vocab atlas (commit `a9454f1`)
- 128 anchors × 23 categories captured at end-of-single-token-user-message
- **Hierarchical attractor structure** revealed: sink + non-sink residue + category attractor + content modulation
- PC1 of vocab-only (sink-removed) = **content-vs-function axis** (33.5% variance)
- **fig21 independently confirmed the t=0.421 pivot** via anchor projection — capitals→nature transition at exactly the same step the AV text flipped

### Discriminant validation (commits `d1524a9`, `1151808`)
- fig27 connectivity reveals 3 macro-clusters: content / function-words / structural (punctuation+numbers)
- fig29 self-validation: top-K hit rates 56-79% per category
- fig28 stability scan (8 anchors × 4 prefix-length contexts): **scope-clarifying finding** — `happy → emotion` projection is only +0.083 ± 0.061. The discriminants do prompt-TOPIC detection, NOT token-presence detection.

### Audit (commits `6049812`, `ee58e62`)
- `nla_audit_findings.py` re-derives every load-bearing number from raw `.pt` files
- 65 → 93 PASS checks after MAIN-46 extension (later extended to 129 by AUDITs 15-19; current canonical count)
- Caught and corrected: fig15 position-drift confound (`||Δh||_feat` 35.55→28.06), `||h_A||` mislabel in Path B observation (was 51.94, that's `||h_A − h_B||`; truth `||h_A||=65.73`)

### Hierarchical re-discrimination (commit `189a054`)
- **Null result**: MAIN-47's hypothesis (hierarchical scheme lifts country top-1 toward 70%) rejected — only 1 capture flipped, accuracy 34%→38%
- Diagnostic revealed the original 34% bundled three failure modes; only one is fixable by basis improvement. Filed MAIN-68 for the labeling-fidelity audit follow-up.

### Figure quality cleanup (commit `2135289`)
- Bumped DPI to 180 uniformly across all 13 render scripts
- Fixed title-vs-first-row overlap on the 5 flipbook scripts (TITLE_RESERVE pattern)
- Wrote `research/observations/figures/INVENTORY.md` cataloguing all 29 figures with full provenance (the inventory now covers 36, after fig31–37 were added in the final session)

## The 6 most important findings (ranked)

1. **Stepwise semantic transitions near t≈0.42** (fig17, fig21) — phase-transition-like discontinuity; the coarse 20-step grid flagged t=0.421, dense re-sampling (MAIN-34) relocated the sharp flip to t≈0.4475 with t=0.421 inside a hybrid plateau
2. **23-discriminant basis classifies prompt-TOPIC, not token-presence** (fig28) — scope-clarifying for all viz primitives in the arc
3. **Hierarchical attractor structure** at layer 20 — sinks (+0.22), non-sink universal residue (+0.4), category attractors (+0.85-0.98), within-category content
4. **`||Δh||_feat` ranks counterfactual surprise** (fig16) — 5.7-11 for plausible swaps vs 28-36 for OOD forcing; potential deployment-time anomaly score
5. **Content-vs-function is the dominant PC1** in vocab-only sink-removed PCA (33.5% variance) — emerges naturally
6. **AR-encoded NL anchors live near a shared attractor** — cos(h_A, h_B) = +0.69 for two maximally-different anchors

## Linear inbox state

11 NLA tickets filed in this session (project:llm). Current state:

| ID | Kind | Status | Title |
|---|---|---|---|
| MAIN-24 | research | Backlog | Hierarchical attractor structure |
| MAIN-25 | research | Backlog | Stepwise semantic transitions at t=0.421 |
| MAIN-26 | research | Backlog | Discriminant basis is prompt-topic, not token-presence |
| MAIN-30 | research | Backlog | ‖Δh‖_feat ranks counterfactual surprise |
| MAIN-34 | idea | Backlog | Dense interpolation near t=0.421 pivot |
| MAIN-38 | idea | Backlog | Sink-dim knockout |
| MAIN-41 | idea | Backlog | Replicate vocab atlas on TinyLlama |
| MAIN-44 | idea | Backlog | Mid-sequence-anchored vocab atlas |
| MAIN-46 | follow-up | **Done** (commit `ee58e62`) | Audit script extension |
| MAIN-47 | follow-up | **Done** (commit `189a054`, null result) | Hierarchical re-discrimination |
| MAIN-48 | idea | Backlog | Concept arithmetic atlas |
| MAIN-68 | follow-up | Backlog | Rebuild country_test with strict prompts |

All carry `ai-filed` + `needs-triage` + `project:llm` + a kind label, per the linear skill conventions. Bodies are free-form prose ≤15 lines with session-UUID footer.

## Artifact data on disk (gitignored, do NOT delete)

`testing/.cache/nla_artifacts/` — ~hundreds of CPU-hours of captured (h, av_text, h_pred) tuples:

```
aggregate_faithfulness.pt   113 captures, 8 prompts, AV+AR roundtrip
country_concept_vector.pt   29 captures (8 country + 8 non + 13 test) + CAV direction
discriminant_stability.pt   32 captures (8 anchors × 4 prefix-length contexts)
forced_continuation.pt      10 captures, 4 forced-continuation pairs
geometric_features.pt       167 feature rows (norms, kurtosis, sparsity, etc.)
interpolation_flipbook.pt   20 interpolation steps + 2 AR-encoded anchors
pairwise_and_hotdims.pt     (167, 3584) H matrix + dim classifier output
rabbit_haiku_gen_trajectory.pt   15 captures, single haiku generation
sink_removed_atlas.pt       cached sink-removed analysis state
vocab_atlas.pt              128 anchors × 23 categories
```

The `testing/.cache` directory is a symlink to the main checkout's cache (`/home/ai/ai-projects/llm/testing/.cache`) so all sessions share the model + artifact cache.

## Suggested next moves (ordered)

**Status (2026-05-30 update):** the original MAIN-44/48/34/71 queue closed
during the arc; the PR landed as #11; PR #66 layered methodology fixes from
a post-merge multi-agent review (F1-F15). The canonical next-paths list
now lives in [`../README.md#possible-next-paths`](../README.md#possible-next-paths)
as **D1-D8**, with each direction tied to a user-shaped theme and filed as
a Linear ticket (MAIN-265 through MAIN-272). Brief priority guide for
resume:

1. **D3 — Audit AV-decoder format-bias** ([MAIN-267](https://linear.app/skothr/issue/MAIN-267), **Medium**). If positive, re-frames the whole arc. Methodological cleanup; do first.
2. **D5 — Cross-model replication** ([MAIN-269](https://linear.app/skothr/issue/MAIN-269)) + **D4 — protocol-invariant subspace** ([MAIN-268](https://linear.app/skothr/issue/MAIN-268)). Scope-test pair: confirms or rejects layer-20-as-attractor-structure as a model-general property.
3. **D6 — Basin landscape mapping** ([MAIN-270](https://linear.app/skothr/issue/MAIN-270)). Inverts the arc: basin-first instead of vocab-first. Multi-session.
4. **D1 — Discovery-viz frontend** ([MAIN-265](https://linear.app/skothr/issue/MAIN-265)). Largest unrealized direction theme; connects gui_cpp to NLA arc. Unblocks D7.
5. **D2 — Eval-aware probe** ([MAIN-266](https://linear.app/skothr/issue/MAIN-266)) + **D8 — replicate Anthropic examples** ([MAIN-272](https://linear.app/skothr/issue/MAIN-272)). Original-motivating-question track.

The remaining open follow-up tickets from the arc proper —
[MAIN-38](https://linear.app/skothr/issue/MAIN-38) (sink-dim knockout),
[MAIN-41](https://linear.app/skothr/issue/MAIN-41) (vocab atlas on TinyLlama,
partially superseded by D5), [MAIN-68](https://linear.app/skothr/issue/MAIN-68)
(country_test rebuild) — remain in Backlog and are independently picked up.

## How to resume after compact

```bash
cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research
git log --oneline -5      # confirm tip is 2135289
git status                # confirm clean (only testing/.cache showing)

# Run audit to verify state hasn't drifted
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_audit_findings.py        # expect 129 PASS / 0 FAIL

# Read the figure catalog
$EDITOR research/observations/figures/INVENTORY.md
```

Then pick the next direction from the D1-D8 list in the 2026-05-30 update above (start with D3 — audit AV-decoder format-bias) and follow the linear skill's working flow: transition to In Progress with a starting comment, do the work, close with resolution comment + commit reference.
