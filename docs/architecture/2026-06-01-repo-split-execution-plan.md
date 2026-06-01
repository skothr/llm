# Repo Split — Execution Plan (MAIN-264)

**Date:** 2026-06-01 · **Branch:** `refactor/repo-reorg` · **Status:** PLANNING ONLY — no history rewrite until explicit greenlight.
**Supersedes** the STAY verdict in `2026-06-01-repo-split-rfc.md`; that RFC's investigation governs the HOW. Decision + rationale: see the RFC's `## DECISION` section.

This plan is grounded in a complete tracked-path inventory taken on 2026-06-01 at
`refactor/repo-reorg@a775a1d` (post-#68 master `be271e1` + cleanup). Every path
below was verified present; no placeholders.

---

## 0. Target repos + dependency DAG

```
llm_surgeon      Python library (model-surgery ops) + its full test suite. Standalone.
   ├── llm-research    theory KB + research arcs (Fellows showcase). Pins llm_surgeon.
   └── llm-gui-react   FROZEN React/FastAPI GUI. Pins a llm_surgeon snapshot.
llobotomy        C++/ImGui frontend + bundled llm_engine_cpp. Standalone. History-preserve, then rework.
```

`llm_surgeon` is the shared root; nothing depends on `llm-gui-react`, `llm-research`, or `llobotomy`.

---

## 1. Complete path → repo mapping (the contract — nothing orphaned, nothing duplicated)

| Source path (tracked) | Count | → Repo | Placed at |
|---|---|---|---|
| `theory/` | 344 | llm-research | `theory/` |
| `research/` | 87 | llm-research | `research/` (LFS: `**/figures/*.png`, `**/data/*.pt`) |
| `testing/examples/nla_*`, `README_NLA.md`, `_layer_hooks.py` | 46 | llm-research | `examples/` |
| `testing/llm_surgeon/` | 19 | llm_surgeon | `llm_surgeon/` (package root) |
| `testing/tests/` | 24 | llm_surgeon | `tests/` |
| `testing/experiments/` (`test_full_recipe.yaml`) | 1 | llm_surgeon | `experiments/` |
| `testing/prompts/` (`default.json`) | 1 | llm_surgeon | `prompts/` |
| `testing/conftest.py`, `pyproject.toml`, `pyrightconfig.json`, `.gitignore` | 4 | llm_surgeon | repo root |
| `testing/examples/{probe_demo,quickstart,qwen_load_check}.py` | 3 | llm_surgeon | `examples/` |
| `testing/docs/superpowers/` (toolkit cluster — see §5) | ~34 | llm_surgeon | `docs/design-history/` |
| `testing/gui/` | 112 | llm-gui-react | repo root |
| `testing/docs/superpowers/` (GUI cluster — see §5) | 6 | llm-gui-react | `docs/design-history/` |
| `testing/gui_cpp/` | 50 | llobotomy | `gui_cpp/` |
| `testing/llm_engine_cpp/` | 62 | llobotomy | `llm_engine_cpp/` (sibling of `gui_cpp/`) |
| `LICENSE` | 1 | **all 4** | repo root (GPLv3 — see §3e) |
| `README.md`, `CLAUDE.md`, `.gitignore`, `.gitattributes`, `pyrightconfig.json` | 5 | **recreated per-repo** | see §3e |
| `docs/architecture/*-repo-split-*.md` (RFC + this plan) | 2 | archived monorepo | stays as the split record |

**Not tracked, not carried** (each repo bootstraps locally): `lib/llama.cpp` (gitignored), `testing/.cache/` (G-Drive models, gitignored), `testing/experiments.db` (gitignored).

---

## 2. Method: clone-and-filter (non-destructive to the monorepo)

**Do NOT rewrite the monorepo in place.** For each target repo, work on a *fresh clone* so the monorepo, its 9 worktrees, and all branches stay intact and recoverable:

```bash
git clone --no-local /home/ai/ai-projects/llm /tmp/split/<repo>
cd /tmp/split/<repo>
git filter-repo <path selectors + renames>      # rewrites THIS clone only
# scaffold (§3e), wire deps (§4), verify (§8), then push to a new GitHub repo (§9)
```

The monorepo is **archived read-only** at the end (§9), so rollback = delete the new repos; nothing is lost. `git filter-repo` is already installed under `~/.local` (used for the 2026-05-31 rewrite).

---

## 3. Per-repo extraction recipes

Run each from a fresh `--no-local` clone (§2). `filter-repo` rewrites all of history to contain only the kept paths, renamed to the target layout.

### 3a. llm_surgeon

```bash
git filter-repo \
  --path testing/llm_surgeon/ --path testing/tests/ \
  --path testing/experiments/ --path testing/prompts/ \
  --path testing/conftest.py --path testing/pyproject.toml \
  --path testing/pyrightconfig.json --path testing/.gitignore \
  --path testing/examples/probe_demo.py --path testing/examples/quickstart.py \
  --path testing/examples/qwen_load_check.py \
  --path-glob 'testing/docs/superpowers/**'  \
  --path LICENSE \
  --invert-paths --path-regex '^testing/docs/superpowers/(plans|specs)/2026-04-(11|13|15)-(live-probe-gui|gui-polish-reliability|memory-optimization)' \
  --path-rename testing/examples/:examples/ \
  --path-rename testing/docs/superpowers/:docs/design-history/ \
  --path-rename testing/:
```
(The `--invert-paths --path-regex` line drops the 6 GUI-cluster docs from this repo; they go to gui-react. `--path-rename testing/:` strips the `testing/` prefix last, after the more specific renames.)

### 3b. llm-research

```bash
git filter-repo \
  --path theory/ --path research/ \
  --path-glob 'testing/examples/nla_*' \
  --path testing/examples/README_NLA.md --path testing/examples/_layer_hooks.py \
  --path .gitattributes --path LICENSE \
  --path-rename testing/examples/:examples/
```
Keeps `theory/` and `research/` at root (LaTeX `Makefile` lives in `theory/`). `.gitattributes` carries the two LFS rules (both scoped to `research/**`). LFS object re-push: §7. The subliminal arc is **not** on master → carried separately, §6.

### 3c. llm-gui-react (frozen)

```bash
git filter-repo \
  --path testing/gui/ \
  --path-regex '^testing/docs/superpowers/(plans|specs)/2026-04-(11|13|15)-(live-probe-gui|gui-polish-reliability|memory-optimization)' \
  --path LICENSE \
  --path-rename testing/docs/superpowers/:docs/design-history/ \
  --path-rename testing/gui/:
```

### 3d. llobotomy

```bash
git filter-repo \
  --path testing/gui_cpp/ --path testing/llm_engine_cpp/ \
  --path LICENSE \
  --path-rename testing/:
```
Result: `gui_cpp/` and `llm_engine_cpp/` as **siblings at root** — `gui_cpp/CMakeLists.txt:89`'s `add_subdirectory(../llm_engine_cpp ...)` and `llmengine::llm_engine` link resolve **unchanged**. Add a thin root `CMakeLists.txt` that `add_subdirectory(gui_cpp)` for a one-command build (optional; building from `gui_cpp/` already works).

### 3e. Per-repo scaffolding (new files, added after filter-repo)

- **`LICENSE`** — GPLv3, carried into all 4 (llm_surgeon is GPLv3 per CLAUDE.md license note; the others are same-author derivatives → inherit GPLv3).
- **`README.md`** — fresh per repo (purpose, build/test, the cross-repo dep note). `llm-gui-react`'s README leads with a **FROZEN / superseded-by-llobotomy** banner.
- **`CLAUDE.md`** — per-repo, trimmed from the monorepo's: keep the worktree HARD RULE + the repo-specific build/test sections; drop the other three projects' sections.
- **`.gitignore`** — per repo, from the monorepo's relevant lines (e.g. gui-react keeps the `node_modules`/`.vite`/playwright lines; llobotomy keeps C++ build-dir ignores; research keeps `.cache`/LaTeX-artifact lines).
- **`pyrightconfig.json`** — the 3 Python repos (surgeon, research, gui-react); not llobotomy.
- **`.gitattributes`** — **only llm-research** (the LFS rules); the others have no LFS objects.

---

## 4. Cross-repo dependency wiring

`llm-research` and `llm-gui-react` both import `llm_surgeon` (research: the `nla_*` scripts; gui-react: 3 backend routes — `inspect.py`, `probes.py`, `sessions.py`). After the split:

- **Dev**: `pip install -e ../llm_surgeon` (sibling editable), documented in each repo's README.
- **Pinned**: `pyproject.toml` declares `llm_surgeon @ git+https://github.com/skothr/llm_surgeon@<tag>` (or a published version once llm_surgeon tags releases). `llm-gui-react` pins a **specific commit** (frozen → never floats).
- llm_surgeon must grow a real package version (`pyproject.toml` `[project] version`) and a v0.1.0 tag at extraction so the dependents can pin it.

---

## 5. Superpowers doc routing (per the keep-and-relocate decision)

All ~40 docs map 1:1 to shipped code (verified: `surgery/export/benchmark/inspect/recipe/probe` modules; `attribution_patch[_per_head|_per_neuron]`, `edge_attribution_patch`, `extract_circuit`; `decode-{head,neuron,residual}` routes; `Circuit/EdgeAttribution/PerNeuron/PerHead/LensTrace/LogitLens` panels). They are design lineage, not abandoned plans.

- **GUI cluster → llm-gui-react** `docs/design-history/` (6 docs): `2026-04-{11,13,15}` × {`live-probe-gui`, `gui-polish-reliability`, `memory-optimization`} (plans + specs). *(`memory-optimization` is borderline library/GUI — routed to GUI as the multi-model-comparison driver; revisit if it reads as library.)*
- **Everything else → llm_surgeon** `docs/design-history/` (~34 docs): foundational design + phase-plan, Phases 1–6, the Apr-17 refactor trio, the 3.5–3.12 attribution series, probe-module.
- **Drop**: `archive/2026-04-08-llm-surgeon-design-v1.md` (superseded by v2, which also shipped).

---

## 6. Pre-split branch reconciliation (the messy part — honest)

**Hazard:** ~8 branches predate the 2026-05-31 history rewrite (MAIN-273). They read as "500–600 commits ahead" of master because they carry the *old* SHA line. **They cannot be merged** — that would re-introduce the purged node_modules + non-LFS figures. Their genuine *unique* work must be cherry-picked onto post-rewrite master (or carried into the target new repo), exactly the pattern chosen for subliminal.

**This does NOT block the 4-repo extraction** (the rewrite + PR #68 already consolidated master). It is a separable follow-on per branch.

**Triage procedure** (per branch — run after the new repos exist):
```bash
# 1. Is its content already on master (pre-rewrite dup)?  Diff the worktree against master:
git diff origin/master..<branch> --stat        # large + node_modules/figures noise → mostly dup
# 2. Find genuine unique commits (real new work, not replayed old history):
git log --oneline --no-merges origin/master..<branch>   # inspect subjects
# 3a. unique work wanted → cherry-pick the specific commits into the target NEW repo
#     (rewrite paths to the new layout first, or re-apply the diff by hand for small sets)
# 3b. already-on-master / abandoned → delete the branch
```

| Branch | Pushed? | Likely target | Action |
|---|---|---|---|
| `feat/subliminal-transfer` | yes (clean, post-rewrite) | llm-research | **carry separately** (decided): cherry-pick its 8 commits into the new research repo |
| `docs/nla-arc-attribution` | yes (pre-rewrite) | llm-research | triage unique NLA-attribution work → cherry-pick; else delete |
| `feat/engine-arch-coverage` | yes (pre-rewrite; real engine work) | llobotomy | triage per-arch ablation / deep tests → cherry-pick into llobotomy |
| `session/llobotomy` | yes (pre-rewrite) | llobotomy | triage vs the planned rework — user call: cherry-pick or supersede |
| `session/nla-research` | yes (pre-rewrite) | llm-research | likely already on master via #68 — verify, then delete |
| `learn/pytorch-tutorials` | local-only | (own archive) | **keep** — unique gitignored tutorials; push to its own small repo or archive |
| `session/theory`, `session/nla-research-backup`, `wip/multi-session-checkpoint` | local-only | — | **verify on master, then delete** (decided) |
| `refactor/research-arcs` | merged (#68) | — | delete post-confirm |

---

## 7. LFS handling (llm-research only)

`filter-repo` preserves the LFS *pointer* files but not the LFS *objects*' remote. After extraction:
```bash
cd /tmp/split/llm-research
git lfs install --local
git lfs fetch --all /home/ai/ai-projects/llm   # pull objects from the monorepo's LFS store
git remote add origin <new-research-remote>
git lfs push --all origin                       # re-upload objects to the new repo
```
Verify: `git lfs ls-files` lists the 52 `research/arcs/**` objects; a fresh clone re-renders figures + replays the audit.

---

## 8. Per-repo verification (gate before declaring a repo done)

- **llm_surgeon**: fresh clone → `pip install -e .` → `pytest tests/ -v` green (24-file suite, GUI-free). `pyright` clean.
- **llm-research**: `pip install -e ../llm_surgeon` → `make -C theory all` (LaTeX) → re-render one arc's figures from committed `.pt` → `nla_audit_findings.py` replays. `git lfs ls-files` intact.
- **llm-gui-react**: `cd frontend && tsc --noEmit && vite build && npm run e2e` (9-test smoke); backend imports the pinned `llm_surgeon`.
- **llobotomy**: `cmake -S gui_cpp -B build && cmake --build build` — the `../llm_engine_cpp` sibling include resolves; `llm_engine_cpp/tests` build + run.

---

## 9. GitHub + cutover

1. Create 4 repos under `skothr` via `gh` (bot identity): `llm_surgeon`, `llm-research`, `llm-gui-react`, `llobotomy`.
2. Push each extracted history (**user pushes** — bot is read-only). Research: `git lfs push --all` (§7).
3. **Archive the monorepo**: keep `/home/ai/ai-projects/llm` and the GitHub `llm` repo read-only; replace its README with a pointer to the 4 repos + retain the RFC + this plan as the split record. Do **not** delete — it is the rollback + the pre-rewrite-branch source.
4. Remove the 9 worktrees once their branches are reconciled (§6): `git worktree remove …`.
5. **Linear**: split MAIN-264 into 4 per-repo sub-issues (extraction + scaffold + verify each); relabel `project:llm` → per-repo labels as each lands.

---

## 10. Sequence (ordered)

1. Merge `refactor/repo-reorg` (cleanup + RFC + this plan) to master. ← clean-up-first lands here.
2. Fast-forward local `master` to `origin/master`.
3. Branch triage (§6): list genuine unique work per branch; confirm the delete set with the user.
4. Extract the 4 repos from master (§3) into `/tmp/split/*`; scaffold (§3e); wire deps (§4).
5. Per-repo verification (§8). LFS for research (§7).
6. Create GitHub repos; user pushes; LFS push (§9).
7. Carry-separately work: subliminal + any triaged branch work (§6) into the new repos.
8. Archive monorepo; remove reconciled worktrees; Linear sub-issues (§9).

Steps 1–3 are reversible prep. Step 4 onward operates on throwaway clones — the monorepo is untouched until step 8's archival.

---

## 11. Risks + rollback

- **Rollback is cheap**: the monorepo is never rewritten in place (§2). If an extraction is wrong, delete the new repo + re-run. Nothing is lost until the archival step, which is itself reversible (un-archive).
- **Branch reconciliation incompleteness** (§6) — the main residual risk. Mitigation: the monorepo is retained as the source-of-truth for every pre-rewrite branch; reconciliation can happen lazily after the repos exist.
- **Cross-repo dev friction** (§4) — editable sibling installs need a documented bootstrap; CI per repo must clone/pin `llm_surgeon`.
- **Per-repo CI/CD is new work** — none of the 4 has a pipeline yet; each needs one (out of scope for this plan; track as sub-issues).
- **`memory-optimization` doc routing** (§5) is a judgment call — trivially movable.

---

## 12. Open decisions for the user (before step 4)

1. **Repo names** — `llm_surgeon` (underscore, matches the package), `llm-research`, `llm-gui-react`, `llobotomy`. Confirm or adjust.
2. **GitHub owner/visibility** — all under `skothr`; public or private at creation? (Fellows showcase suggests `llm-research` public.)
3. **Branch triage sign-off** (§6 table) — confirm the delete set + which branches' unique work to cherry-pick.
4. **Execution timing** — greenlight to run steps 1–2 now (merge cleanup, FF master), with step 4+ gated on a second explicit go.
