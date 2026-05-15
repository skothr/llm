# NLA Research Arc — Resume Checkpoint (2026-05-14)

**Branch:** `session/nla-research`
**Worktree:** `/home/ai/ai-projects/llm/.claude/worktrees/nla-research`
**Tip commit:** `2135289` (27 commits ahead of master)
**Working tree:** clean (only gitignored `testing/.cache` symlink visible)
**Audit state:** `nla_audit_findings.py` passes 93/0

Supersedes the earlier resume doc `2026-05-13-nla-arc-summary-for-compact.md` (written at commit `c49a1d9`, before Path B, vocab atlas, discriminant validation, MAIN-46, MAIN-47, and figure cleanup landed).

## Quick orientation

1. `cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research`
2. Confirm in worktree: `git rev-parse --abbrev-ref HEAD` should say `session/nla-research`
3. Read [`figures/INVENTORY.md`](figures/INVENTORY.md) for the figure catalog with provenance per figure
4. Run audit to verify state: `PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python testing/examples/nla_audit_findings.py` — expect 93/0

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
- 65 → 93 PASS checks after MAIN-46 extension
- Caught and corrected: fig15 position-drift confound (`||Δh||_feat` 35.55→28.06), `||h_A||` mislabel in Path B observation (was 51.94, that's `||h_A − h_B||`; truth `||h_A||=65.73`)

### Hierarchical re-discrimination (commit `189a054`)
- **Null result**: MAIN-47's hypothesis (hierarchical scheme lifts country top-1 toward 70%) rejected — only 1 capture flipped, accuracy 34%→38%
- Diagnostic revealed the original 34% bundled three failure modes; only one is fixable by basis improvement. Filed MAIN-68 for the labeling-fidelity audit follow-up.

### Figure quality cleanup (commit `2135289`)
- Bumped DPI to 180 uniformly across all 13 render scripts
- Fixed title-vs-first-row overlap on the 5 flipbook scripts (TITLE_RESERVE pattern)
- Wrote `research/observations/figures/INVENTORY.md` cataloguing all 29 figures with full provenance

## The 6 most important findings (ranked)

1. **Stepwise semantic transitions at t=0.421** (fig17, fig21) — phase-transition-like discontinuity confirmed by two independent measurements
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

After compact, the recommended sequence:

1. **MAIN-44** — mid-sequence-anchored vocab atlas. Tests the central scope-clarifying finding (MAIN-26): does token-presence detection work at non-end-of-prompt positions? ~30 min, base model. **Highest-info-yield open ticket.**
2. **MAIN-48** — concept arithmetic atlas (`France − Germany` AV-decoded). Tests whether NLA preserves additive/subtractive composition. ~15 min AV calls.
3. **MAIN-34** — dense interpolation near t=0.421. Tests the phase-transition claim (MAIN-25) at 100× resolution. ~2.4 hr AV calls — long run.
4. **MAIN-41** — TinyLlama replication of content-vs-function PC1. Cross-model invariance test. ~15 min on TinyLlama.
5. **MAIN-38** — sink-dim knockout. Behavioral test of sink-as-scaffolding (MAIN-24). ~30 min, base model.
6. **MAIN-68** — rebuild country_test set with strict prompts; re-run self-validation. Cheap follow-up to MAIN-47's null result.

Researcher's note: the PR has been deferred until "substantial research direction and a lot of testing" (user direction). The branch is in a clean stopping state; resume from MAIN-44 for the natural next experiment.

## How to resume after compact

```bash
cd /home/ai/ai-projects/llm/.claude/worktrees/nla-research
git log --oneline -5      # confirm tip is 2135289
git status                # confirm clean (only testing/.cache showing)

# Run audit to verify state hasn't drifted
PYTHONPATH=$PWD/testing /home/ai/ai-projects/llm/testing/.venv/bin/python \
    testing/examples/nla_audit_findings.py        # expect 93 PASS / 0 FAIL

# Read the figure catalog
$EDITOR research/observations/figures/INVENTORY.md
```

Then pick the next ticket (start with MAIN-44 per the recommendation above) and follow the linear skill's working flow: transition to In Progress with a starting comment, do the work, close with resolution comment + commit reference.
