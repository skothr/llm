# RFC (MAIN-264): Split the `llm` repo into 4, or stay monorepo + clean up?

**Date:** 2026-06-01 · **Branch:** `refactor/repo-reorg` · **Status:** decision-ready, PLANNING ONLY (no execution).
**Verdict (synthesized + adversarially reviewed):** **STAY MONOREPO + CLEAN UP**, with armed tripwires for a future split.

---

## DECISION (2026-06-01) — user override → **SPLIT** (cleanup first)

The RFC's STAY recommendation was technically sound on the evidence it weighed
(no clean internal cut of `testing/`, branch/worktree teardown cost, LFS re-push).
The user overrode it on **product grounds the technical RFC under-weighted**, and
those grounds dissolve the RFC's strongest objection rather than ignoring it:

1. **`llm_surgeon` is intended as a general, reusable surgery-ops library** → it
   deserves its own repo, independent of any GUI or research consumer.
2. **The React/FastAPI GUI (`testing/gui`) and llobotomy (`testing/gui_cpp`) are
   redundant; llobotomy is the rework** and is to be **restarted with new info**.
   Freezing the React GUI (pin-and-forget) instead of carrying it live **severs
   the `tests` ↔ `gui.backend` ↔ `llm_surgeon` coupling** that was the RFC's #1
   "no clean cut" — the coupling existed *because* the GUI was in-tree.
3. **The research arcs are to be showcased for the Anthropic Fellows program** →
   a focused `llm-research` repo keeps the showcase coherent, not "all over the place."

**Locked target mapping** (from the user's 2026-06-01 decisions):

| New repo | Source subtree(s) | History | Notes |
|---|---|---|---|
| `llm-research` | `theory/` + `research/` + research-flavored `testing/examples/*` | preserve | Fellows showcase; focused |
| `llm_surgeon` | `testing/llm_surgeon/` + `tests/` + `experiments/` + `prompts/` + surgery `examples/` | preserve | general library; **no GUI dep** |
| `llm-gui-react` | `testing/gui/` | preserve | **frozen/deprecated**; pins a snapshot of `llm_surgeon` |
| `llobotomy` | `testing/gui_cpp/` **+** `testing/llm_engine_cpp/` (bundled) | preserve, then rework | existing `add_subdirectory(llm_engine_cpp)` keeps working |

**Sequence:** clean up first (this branch) → fast-forward local master to origin →
push/triage all local-only branches + quiesce all 9 worktrees (hard gate) →
per-repo `git filter-repo` extraction → per-repo verification → archive the monorepo.
The RFC's §3 if-split must-fixes and the four investigation appendices below now
govern the **HOW**; a dedicated split-execution plan supersedes the STAY verdict.

Produced by a 6-agent RFC workflow (boundaries/cross-deps · branches/history/LFS ·
junk inventory · steelman-stay → synthesized RFC → adversarial critique). The
recommendation is the RFC's; the critique's must-fixes apply to the *if-split
contingency* (§3 of the RFC), which is dormant under the STAY verdict. Full
sections below; supporting investigations in the appendix.


---

## The RFC

Confirmed the two load-bearing facts: `feat/subliminal-transfer` has exactly 8 unpushed commits (`07fb779..dd72c0c`, the 9th `f04af40` is already on master via #67/#68) and the branch has **no upstream configured** — zero remotes. The BRANCHES investigation's correction holds. Now writing the synthesis RFC as my final output.

---

# RFC: MAIN-264 — Split the `llm` repo into 4, or stay monorepo + clean up?

**Status:** Decision-ready. Synthesizes four read-only investigations (boundaries/cross-deps, branches/history/LFS, junk inventory, steelman-stay). Honest in both directions per the ticket.
**Base state:** post-#68 master, `be271e1`. Author identity verified: on branch `feat/subliminal-transfer`, **8 unpushed commits on zero remotes** (`git log @{u}..` → fatal: no upstream).

---

## 1. Recommendation: **STAY MONOREPO + CLEAN UP**, with armed tripwires

Three facts decide it:

1. **The proposed split lines fall on edges the monorepo currently makes free, and two of them have no clean cut.** `testing/tests/` imports `gui.backend.*` (≥15 sites) *and* `llm_surgeon.*` — they are literally one Python package rooted at `testing/pyproject.toml`, so `gui` + `tests` + `llm_surgeon` **cannot** be separated regardless of the ticket's mapping. `testing/gui_cpp/CMakeLists.txt:89` does `add_subdirectory(../llm_engine_cpp)` + links `llmengine::llm_engine` — a hard build break on day one of a split. And `testing/examples/` is an **irreducible straddle**: it imports `llm_surgeon` (22 sites) *and* read/writes `research/arcs/.../{data,figures}` via a monorepo-root relative walk (`_nla_artifacts.py:25`). Whichever repo `examples` lands in, one of {package import, data path} becomes a cross-repo edge. The clean 4-way split the ticket imagines does not exist on this tree.

2. **The split's headline benefit is already delivered, and its headline justification is already spent.** The per-component-review benefit ("review just the engine") is produced *today* by the worktree + `project:` scope-label + per-session-branch discipline — evidenced by 9 live worktrees and a clean PR-per-branch merge cadence (PRs #1/#4/#5 engine-only, #2/#3/#7/#16 gui_cpp-only). And the "repo too heavy" justification was retired by the MAIN-273 LFS migration: the 9.3 G working tree is ~910 M `lib/llama.cpp` + ~123 M `node_modules` + model cache — all gitignored/vendored, **none of which a split moves**. The git-tracked footprint is ~780–805 files; the figure bloat is already on LFS.

3. **The only two genuinely clean cut-lines are extractable later at near-zero cost, so staying is not lossy.** `theory/` has zero code cross-refs into `testing/`/`research/`; `llm_engine_cpp` C++ has zero Python/research dependency. These are the cheap wins, and they remain cheap whenever a tripwire (§6) fires. Staying now forfeits nothing that a future split can't capture; splitting now pays ≥7 break-fixes, ≥2 submodule/versioned-dep relationships, and a *second* LFS history rewrite to buy isolation the worktrees already provide.

Counter-honesty: the split is **not wrong in principle**. If an external contributor or a public release of one component arrives (tripwire 1/2, §6), the cost calculus inverts and splitting becomes correct. The recommendation is "stay *now*," not "never."

---

## 2. Decision gates — MAIN-264's five open questions

| # | Question | Answer (under STAY) | Status |
|---|---|---|---|
| **a** | Cross-repo refs (rewrite / submodule / versioned dep)? | **N/A under stay — and that is itself the strongest argument for staying.** The refs that *would* need handling: `testing/examples/` → `llm_surgeon` (22 import sites), `research/arcs/.../README.md:614` relative link `../../../testing/llm_surgeon/probe/_nla.py` (+ ~9 prose citations), `gui/backend` → `llm_surgeon` (~60 sites), `tests` → `gui.backend.*` (≥15). Under stay all remain in-tree imports; no rewrite, no submodule, no pin. The NLA observation files are dated lab-notebook entries whose value *is* their fixed historical paths — rewriting them post-hoc damages citability. | **ANSWERED (moot)** |
| **b** | `lib/llama.cpp` ownership? | **Stays shared external infra at repo root.** It is *not in the repo* — `.gitignore` excludes `llama.cpp/`; it's a local CUDA-built checkout at `/home/ai/ai-projects/llm/lib/llama.cpp`. Consumed by BOTH the C++ side (`llm_engine_cpp/CMakeLists.txt:118` hardcodes the absolute path; `scripts/{demo,verify}.sh`) and the Python side (`llm_surgeon/{llama_engine,gguf_reader,gguf_writer,export}.py`). It has no single-repo owner under any split. The hardcoded **absolute** path is fragile regardless — **fix-on-touch: parameterize via `-DLLAMA_CPP_ROOT=` / env var** (do-regardless, §5). | **ANSWERED** |
| **c** | History-split tool? | **N/A under stay (no second rewrite).** *If forced to split:* `git filter-repo`, not `git subtree split` — it preserves merge topology (the un-squashed PR #11 merge `9059768` carrying 36 arc commits survives), supports `--path`/`--path-rename`, and treats LFS pointer blobs as ordinary content so a follow-up `git lfs push --all` re-homes binaries cleanly. It is the *same tool* MAIN-273 used, proven on this tree. Subtree would flatten merges and is LFS-blind. | **ANSWERED (tool chosen, not executed)** |
| **d** | `gui_cpp` ↔ `gui/backend` transition window? | **Do NOT split during it — finish it first.** The HFProxy/HttpBackend transition (`gui_cpp/docs/HFPROXY_PLAN.md:5-7`, phases 1–3) is mid-flight and spans **three** of the four proposed repos: `gui_cpp` (→llobotomy) → `HFProxyEngine` (in `llm_engine_cpp`) → FastAPI backend (in `llm_surgeon`). Splitting now freezes a moving wire API across repo lines, turning every iteration into a 3-repo change with no atomic commit to bisect. This is **tripwire-adjacent**: a clean finish *enables* a future split rather than blocking it. | **ANSWERED (sequencing constraint)** |
| **e** | Linear labels? | **Keep the existing `project:<basename>` umbrella-and-sub scheme**, not 4 flat repo labels. Umbrella labels preserve cross-component queries ("all open work touching the GUI stack") that 4 flat labels fragment. No change needed. | **ANSWERED** |

No gate is a blocking unknown. The one item needing the *user's* explicit input is not a gate but a scope confirmation: the 4-repo boundary set was *inferred* by the investigations (no on-disk split plan exists), and `testing/gui`'s placement is ambiguous (React frontend pairs with the `llm_surgeon` backend, but its C++ successor `gui_cpp` would go to a separate engine/llobotomy repo). Under STAY this is moot; it only matters if a tripwire fires.

---

## 3. IF SPLIT (the contingency plan — execute only if a §6 tripwire fires)

This section exists so a future split is *mechanical*, not a fresh design exercise. It is the honest pro-split path, fully specified.

### 3.0 Hard prerequisite — PUSH EVERYTHING FIRST (work is lost otherwise)
A split rewrites history; any local-only branch not on a remote at split time is **destroyed**. Verified must-push-first list:

| Branch | Risk | Action |
|---|---|---|
| `feat/subliminal-transfer` | **8 commits, zero remotes** (`@{u}` → no upstream; confirmed this session) | `git push -u origin feat/subliminal-transfer` — **top priority** |
| `session/theory` | 531 commits, no remote tracking | `git push -u origin session/theory` |
| `session/nla-research` | 601 commits, `origin: gone` (deleted post-#11) | re-push or the long arc history is local-only |

Then create a third safety net (atop MAIN-273's existing `origin/backup/pre-rewrite-master`):
```bash
git push origin be271e1:refs/heads/backup/pre-split-master
```

### 3.1 Dir → repo mapping (corrected from the ticket; reflects the real coupling)

| Repo | Contents | Correction vs ticket |
|---|---|---|
| **llm-theory** | `theory/` | Clean. Optional un-nest `--path-rename theory/:./`. Drop research-scoped LFS rules from its `.gitattributes` (no LFS objects here). |
| **llm-surgeon** | `testing/{llm_surgeon, gui, tests, examples, experiments, prompts, docs, pyproject.toml}` | `tests`+`gui`+`llm_surgeon` are **forced together** (gate a/D). `experiments/` goes HERE not research — `tracking.py:11` writes `parent.parent/"experiments/experiments.db"` assuming it's a sibling of `llm_surgeon/`. `prompts/` HERE (`probe/_attribution.py` + React `store.ts` read it). `examples/` is the straddle — placing it here keeps import A intact but pushes the research-data path (B) across the boundary. |
| **llm-engine** (bundle gui_cpp + engine) | `testing/{llm_engine_cpp, gui_cpp}` | **Bundle them** — the `add_subdirectory(../llm_engine_cpp)` link (F) has no clean cut; bundling is the cheapest resolution of the ticket's "bundle vs standalone TBD." |
| **llm-research** | `research/` (incl. `arcs/`, `ARC_PROCESS.md`, `archive/`) | Owns **100% of the 52 LFS objects**. The examples↔research data straddle means either `examples` also lands here (breaking import A) or a versioned-dep/submodule bridges it. |
| shared | `LICENSE` → all four; `CLAUDE.md`/`README.md`/`.gitignore`/`pyrightconfig.json` forked per-repo | `lib/llama.cpp` is external prereq for both surgeon and engine. |

### 3.2 History-split method + concrete commands
`git filter-repo` on a fresh `--mirror` clone per repo (it refuses non-fresh clones). Tool lives at `~/.local/.../git_filter_repo.py` (hidden by sandbox read-deny) — copy to `$TMPDIR` and run `python3 $TMPDIR/git_filter_repo.py`.

```bash
# llm-research — owns ALL 52 LFS objects
git clone --mirror git@github.com:skothr/llm.git llm-research.git && cd llm-research.git
git filter-repo --path research/ --path .gitattributes --path LICENSE
git remote add origin git@github.com:skothr/llm-research.git
git push --all && git push --tags
git lfs fetch --all && git lfs push --all origin     # RE-HOME binaries (see 3.4)

# llm-theory (no LFS) — drop research LFS rules afterward
git clone --mirror git@github.com:skothr/llm.git llm-theory.git && cd llm-theory.git
git filter-repo --path theory/ --path LICENSE --path-rename theory/:./

# llm-engine (no LFS) — bundle engine + gui_cpp
git clone --mirror git@github.com:skothr/llm.git llm-engine.git && cd llm-engine.git
git filter-repo --path testing/llm_engine_cpp/ --path testing/gui_cpp/ --path LICENSE

# llm-surgeon (no LFS) — the forced-together Python package
git clone --mirror git@github.com:skothr/llm.git llm-surgeon.git && cd llm-surgeon.git
git filter-repo \
  --path testing/llm_surgeon/ --path testing/gui/ --path testing/tests/ \
  --path testing/examples/ --path testing/experiments/ --path testing/prompts/ \
  --path testing/docs/ --path testing/pyproject.toml --path LICENSE
```
Prefer N positive-`--path` clones over `--invert-paths` subtraction — easier to audit ownership.

### 3.3 Branch / worktree migration table

| Branch | Tip | True delta vs `be271e1` | Class | → Repo | Pre-split action |
|---|---|---|---|---|---|
| `feat/subliminal-transfer` | `dd72c0c` | **8 (LOCAL-ONLY)** | active | llm-research | **PUSH FIRST** |
| `session/theory` | `90718a6` | 531 (no remote) | active | llm-theory | **PUSH FIRST** |
| `session/nla-research` | `61c0c77` | 601 (`origin: gone`) | active | llm-research | **RE-PUSH FIRST** |
| `session/llobotomy` | `69ca315` | 533 (pushed) | active | llm-engine | safe |
| `feat/engine-arch-coverage` | `84b8da7` | 566 (pushed) | active | llm-engine | safe |
| `refactor/research-arcs` | `48b2a07` | 0 (== master via #68) | merged | — | retire |
| `docs/nla-arc-attribution` | `f84c6a9` | 0 (on master via #67) | merged | — | retire (content→research) |
| `session/nla-research-backup` | `4293b67` | 29 "unique" = patch-id dups | safety-net | — | retire |
| `wip/multi-session-checkpoint` | `2922f55` | 527, ancestor of nothing | safety-net | — | retire |
| `learn/pytorch-tutorials` | `7f3e1ac` | 580 (gitignored scratch) | personal | **none** | does NOT migrate — archive locally |
| `origin/backup/pre-rewrite-master`, `origin/master-rewritten` | — | — | safety-net | — | retire after split confidence |
| `origin/feat/gui-ui-polish`, `origin/feat/llm-engine-modelview` | — | merged | merged | — | retire (verify #1) |

Note: `git cherry` over-reports here (every long branch shows the stale pre-rewrite root `70b4327` as a phantom `+`); use `git rev-list be271e1..<branch>` for true delta. All branches are legitimate descendants of the rewritten master — none sit on foreign roots, they are merely behind.

### 3.4 LFS ownership + re-push (the load-bearing gotcha)
- **100% of the 52 LFS objects** (37 figures + 15 `.pt`, ~31 MB local store) are under `research/arcs/nla-verbalizer/` → **all to llm-research**. The other three repos carry zero LFS and should *drop* the research-scoped LFS rules from their `.gitattributes`.
- **filter-repo preserves only the 130-byte pointer blobs, NOT the binaries.** After split + new remote, the new repo references OIDs absent from its GitHub LFS store. Mandatory re-home: `git lfs fetch --all <old-origin>` then `git lfs push --all <new-origin>`. Skip it and fresh clones get `smudge filter lfs failed` — the same class of failure MAIN-273 hit.

### 3.5 Cross-ref handling (per edge — rewrite vs submodule vs versioned dep)
| Edge | Spans | Resolution |
|---|---|---|
| `examples`→`llm_surgeon` import (A) | research↔surgeon | **versioned dep** (`pip install llm-surgeon==X`) or submodule |
| `examples`↔`research/data` walk (B) | within research (if examples→research) | OK in-repo; else explicit `data/` resolver contract |
| `research/README:614` relative link (C) | research↔surgeon | **rewrite** to repo-qualified URL / pinned ref |
| `gui_cpp`→`llm_engine_cpp` CMake (F) | within llm-engine (bundled) | **bundle** (no cross-repo edge) |
| `HFProxyEngine`→`gui/backend` HTTP (G) | engine↔surgeon | **freeze + version the wire API** — only survivable if transition (gate d) is finished first |
| `llm_engine_cpp`→`lib/llama.cpp` (H) | external | submodule under llm-engine OR documented `-DLLAMA_CPP_ROOT=` prereq |

### 3.6 Sequenced migration plan
1. **Push-everything-first** (§3.0) — subliminal/theory/nla-research + `backup/pre-split-master` tag.
2. **Finish the gui_cpp↔backend HTTP transition** (gate d) — do not split mid-flight.
3. **Retire merged/superseded branches** (§3.3 retire rows) — verify #1 merged first.
4. **Resolve the do-regardless cleanup** (§5) so the split copies a clean tree.
5. **filter-repo each repo** (§3.2) on fresh mirror clones, llm-research first (LFS).
6. **Re-home LFS** (§3.4) into llm-research's new store; verify a fresh clone + `lfs pull`.
7. **Rewrite cross-refs** (§3.5) as first commits in the new repos.
8. **Stand up 4 CI pipelines** (none exist today — this is net-new work, not a port).
9. **Retain all safety nets** (`pre-rewrite-master`, `master-rewritten`, `pre-split-master`, `wip/multi-session-checkpoint`) until every new repo verifies clone+lfs+build.

**Push steps require the user:** SSH push is sandbox-blocked and bot `skothr-cc` is read-only. All `git push` / `git lfs push` are handed off.

---

## 4. IF STAY (the recommended path's positive program)

Not "do nothing" — tighten the monorepo so the two real cut-lines stay cheap and the shared-infra fragility gets fixed in place:

1. **Parameterize the `lib/llama.cpp` path** — replace the hardcoded `/home/ai/ai-projects/llm/lib/llama.cpp` in `llm_engine_cpp/CMakeLists.txt:118` + `scripts/{demo,verify}.sh` with `-DLLAMA_CPP_ROOT=` / env var. Fixes the single fragile external pin regardless of split. (Also a gate-b answer.)
2. **Lift `testing/.cache` to an env-var-configured path** — it's referenced by ~25 files across examples/tests/gui/surgeon/engine via relative walks. An env-var contract removes the monorepo-root assumption and makes a *future* extraction mechanical.
3. **Resolve the `examples`↔`research` straddle in place** — introduce an explicit `data/` resolver contract (one function that locates `research/arcs/.../data`) instead of the 3-level `parent` walk in `_nla_artifacts.py:25`. This is the single edge that makes the package boundary ambiguous; naming it explicitly costs nothing now and de-risks any future split.
4. **Document the `llmengine::llm_engine` interface as an internal library contract** — it already *is* a clean CMake target; a short interface doc makes a future engine extraction (tripwire 4) mechanical.
5. **Add the one CI pipeline that doesn't exist yet** — single workflow, per-component job matrix (theory-build / pyright+pytest / clang-tidy+cmake / frontend-e2e). Gets ~90% of "independent CI signal" at ~25% of 4-pipeline cost.
6. **Keep the `project:<basename>` umbrella label scheme** (gate e) — no change.

---

## 5. Do-regardless cleanup (sequenced, safe, reference-checked)

The tree is unusually clean — no stray `.db`/`node_modules`/`.pyc`/`.DS_Store`/LaTeX-`.aux` tracked; binary deliverables already on LFS. The inventory is small and centers on **two genuine gitignore/tracking contradictions** plus one orphan. **All references were checked before recommending removal.**

**Sequence (lowest-risk → housekeeping):**

1. **Resolve gitignore↔tracking contradiction #1 — `theory/archive/`** (`.gitignore:51` ignores it, but 5 files tracked). Docs treat it as an intentional committed v1 snapshot → **drop line 51 from `.gitignore`** (make ignore agree with reality). Sub-cleanup: the one safe *delete* is `theory/archive/2026-05-03-pre-expansion/build/llm-core-architecture/llm-core-architecture.tex` — a build-output `.tex` committed inside an archive (double-junk).
2. **Resolve gitignore↔tracking contradiction #2 — root `docs/superpowers/`** (`.gitignore:64` ignores it, but 8 files tracked; all 2026-04-11→04-15 GUI/probe specs). Referenced only from *within themselves* — no code or top-level doc cites the **root-level** path (all live citations point at `testing/docs/superpowers/` and `theory/docs/superpowers/`). **Move them under `testing/docs/superpowers/`** (where the live ones live + where the ignore intent permits tracking) OR `git rm --cached` to honor the rule. **Do NOT hard-delete** — only copy of the GUI design history.
3. **Archive the one orphan doc** — `testing/docs/superpowers/specs/2026-04-08-llm-surgeon-design-v1.md`: **zero inbound references** (grep `llm-surgeon-design-v1` → nothing), superseded by the v2 (7 inbound refs), still targets the abandoned "LLaMA 3 8B 32-layer" framing. Safe to ARCHIVE; safe to DELETE if pre-v2 archaeology isn't valued.
4. **Optional housekeeping** — drop 10 of 11 redundant `.gitkeep` files (dirs now have 4–99 siblings). **KEEP** `theory/sources/forums/.gitkeep` (0 siblings — dir vanishes without it).

**Confirmed NOT junk (looks removable, is load-bearing — do not touch):**
- All 51 superpowers docs *except* the v1 orphan — cross-referenced as design-of-record; e.g. `_intervention.py:274` docstring cites `2026-04-17-phase3-activation-patching-design.md`; `CLAUDE.md:115,150` + `theory/README.md` cite the `theory/docs/superpowers/` tree as the live structural home.
- The 52 LFS `.pt`/`.png` arc deliverables — declared reproducible deliverables in `research/ARC_PROCESS.md`.
- `llm_engine_cpp/libs/{cpp-httplib,nlohmann_json}/` — intentional vendored MIT single-header deps, wired into CMake.
- `research/archive/` — a *properly-curated* archive (its README declares it deliberate archaeology — the model to follow).

The single highest-value, lowest-risk action is **#1 + #2 (resolve the two gitignore contradictions)** — a genuine inconsistency independent of the split decision, and it is precisely what makes "is this dir supposed to be in the repo?" ambiguous today.

---

## 6. Risks + reversibility of the recommended path (STAY)

**Risk of staying (honest):**
- **Atomic-cross-component capability is retained but the volume that exercises it is low** — measured over the last 300 commits: 2 touch `research/`+`llm_surgeon`, 3 touch `examples/`+`llm_surgeon`, 0 touch `surgeon`+`cpp`. The benefit is real (single reviewable diff, single revert across the `examples↔llm_surgeon` lockstep) but modest. Not the load-bearing argument.
- **Deferred-split debt accrues silently** — if the engine boundary or the examples↔research straddle is *not* documented now (§4.3/4.4), a future forced split is a fresh design exercise rather than a mechanical one.
- **The §5 contradictions left unresolved** keep the "what's in the repo" question ambiguous and complicate any later extraction.

**Reversibility (the deciding asymmetry):**
- **STAY is fully reversible.** Nothing about staying forecloses a future split — `theory/` and `llm_engine_cpp` remain zero-inbound-dependency and extractable on demand. The §4 cleanups *increase* future splittability. Cost of changing course later: run §3 when a tripwire fires.
- **SPLIT is expensively reversible.** Un-splitting means re-merging 4 histories, reconciling 4 diverged `.gitignore`/`CLAUDE.md`/CI configs, and a *third* LFS rewrite. A split executed *during* the gui_cpp↔backend transition (gate d) is barely reversible — the in-flight wire API would have diverged across repo lines.

**Tripwires that flip the recommendation to SPLIT (re-open MAIN-264 when any one fires):**
1. **An external contributor lands on exactly one component** — multi-clone friction inverts; scoped repo becomes cheaper than monorepo onboarding.
2. **A component goes public / open-source standalone** — needs its own repo, license-scope, release cadence, issue tracker. (License already clean per CLAUDE.md → mechanically ready.)
3. **Single-pipeline CI wall-clock crosses ~10–15 min** — per-repo CI to parallelize independent components starts paying for itself. (Cannot fire until CI exists at all.)
4. **The `llmengine::llm_engine` interface gains a ≥3rd consumer** — the engine has earned standalone-library status; submodule/package cost justified by reuse, not fiat.

**Bottom line:** Stay monorepo now. Execute the §5 cleanup (resolve both gitignore contradictions + archive the v1 orphan) and the §4 positive program (parameterize `lib/llama.cpp`, env-var `testing/.cache`, name the examples↔research resolver, document the engine boundary, add one CI). Finish the gui_cpp↔backend transition before reconsidering. Arm the four tripwires. The §3 contingency is fully specified so that, if a tripwire fires, the split is a mechanical execution — push-everything-first, filter-repo, re-home LFS — not a redesign.

---

## Adversarial critique

I have everything I need. The HFProxy straddle is confirmed real and spans three proposed repos (gui_cpp + llm_engine_cpp docs, plus the gui/backend FastAPI target). The RFC's gate-d sequencing constraint is sound. The safety-net refs are confirmed (origin/backup/pre-rewrite-master, origin/master-rewritten exist as remotes).

One final check resolved by the above: the RFC's §3.0 lists `feat/subliminal-transfer` (8 commits, no upstream — CONFIRMED), `session/theory` (no remote — CONFIRMED `upstream=` empty), and `session/nla-research` (origin: gone — CONFIRMED `track=[gone]`). Those push-first warnings are all accurate. But the table OMITS `feat/engine-arch-coverage` analysis correctness and, critically, the RFC's own branch `refactor/repo-reorg` is missing from the migration table entirely.

I have enough verified evidence to write the critique. Here is my assessment.

---

# Adversarial critique: MAIN-264 split-vs-stay RFC

I verified the RFC's load-bearing claims against the tree at `be271e1` (which IS the correct investigation base — see Finding 1). Most structural claims hold. The recommendation (STAY) is sound and not rationalized. But there are real gaps, one of them a blocker if a split is ever executed from this RFC as written.

## Lens 2 first (is the recommendation sound / biased?): mostly YES, sound

The STAY recommendation is **well-supported and honestly argued**, not rationalized. I confirmed the three deciding facts:
- The forced-together Python package is real: `tests → gui.backend` (15 occurrences, verified), `tests → llm_surgeon`, `gui/backend → llm_surgeon` (38 occurrences across 4 files), all rooted at one `testing/pyproject.toml`. These genuinely cannot be cut cleanly.
- `gui_cpp/CMakeLists.txt:89` `add_subdirectory(.../llm_engine_cpp)` + `:120` link `llmengine::llm_engine` — verified, a real day-one build break.
- `_nla_artifacts.py:25` `_REPO_ROOT = parent.parent.parent` then `/research/arcs/nla-verbalizer/data` — verified, the irreducible examples↔research straddle is real.

The reversibility asymmetry (STAY reversible, SPLIT needs a third LFS rewrite) is the strongest argument and it is correct. The tripwires are concrete and falsifiable. **No bias detected toward staying** — the contingency plan (§3) is fully and honestly specified, the counter-honesty paragraphs are genuine. This is the rare RFC that argues its own losing side fairly.

## Findings

### BLOCKER-1 — `refactor/repo-reorg` (the RFC author's OWN branch) is absent from the push-first list and migration table
`refactor/repo-reorg` exists locally, is **10 commits ahead of master** (`b8123f2..be271e1`), and its upstream is set to `origin/master` (not its own remote). `git rev-list master..refactor/repo-reorg` = 10. I could not confirm a dedicated `origin/refactor/repo-reorg` remote exists. If this branch carries the RFC + reorg scaffolding and is only pushed to `origin/master`'s ref (not its own), a history rewrite could strand it exactly like `feat/subliminal-transfer`. The RFC's §3.0 "verified must-push-first list" claims to be complete but never audited its own working branch.
**Fix:** Add `refactor/repo-reorg` to §3.0. Run `git rev-parse refactor/repo-reorg@{u}` and `git ls-remote origin refactor/repo-reorg`; if no dedicated remote, `git push -u origin refactor/repo-reorg` before any rewrite. Severity blocker because it is the branch the split work itself lives on.

### SHOULD-FIX-2 — Base-state label is locally ambiguous; investigations ran on `be271e1` but local `master` is 10 behind
The RFC says "post-#68 master, `be271e1`." Verified: `be271e1` is PR #68 (`refactor/research-arcs` merge) and equals **`origin/master`** and the tip of `refactor/repo-reorg` — but **local `master` is `38ac87d` (PR #67), [behind 10]**. `be271e1` is NOT an ancestor of local master. Anyone resuming on the local main checkout and following the RFC's branch-delta numbers will compute different deltas (the §3.3 table's "delta vs `be271e1`" only matches if you fetch/fast-forward first). The investigations are correct *for origin/master*; the RFC just never states that the local checkout must be synced first.
**Fix:** State explicitly "all deltas computed against `origin/master` = `be271e1`; sync local master before acting (`git fetch && git merge --ff-only`)." One line.

### SHOULD-FIX-3 — LFS object breakdown is transposed (52 total is right; 37+15 is wrong)
The RFC repeats "52 LFS objects (37 figures + 15 `.pt`)." Verified on `be271e1`: **36 `.png` + 16 `.pt` = 52.** The total is right; the split is off by one in each direction. Also the `.gitattributes` rules are `research/**/figures/*.png` and `research/**/data/*.pt` (glob, not the literal `research/arcs/nla-verbalizer/` path the RFC implies in §3.4). Minor, but it is a load-bearing number in the "re-home LFS" step — and the RFC's §5 asserts "no `.db`/junk; binaries on LFS" while the actual data dir is `.pt` files under `data/`, not figures only.
**Fix:** Correct to "36 figures + 16 `.pt` = 52, matched by globs `research/**/figures/*.png` and `research/**/data/*.pt`."

### SHOULD-FIX-4 — `learn/pytorch-tutorials` handling is defensible but the stated rationale is wrong
The RFC says "personal / **gitignored** scratch → does NOT migrate." I confirmed `git check-ignore learn/pytorch/01_tensors.py` returns the path (commit `eaeefda` added `learn/` to `.gitignore`) AND the 6 files are committed/tracked on that branch (`7f3e1ac`). So they are **force-added-then-ignored**: real tracked content that a `filter-repo --path` selection on the 4 target dirs would silently drop. The conclusion (don't migrate, archive locally) is fine, but "gitignored scratch" understates it — these are tracked blobs in branch history that vanish on rewrite with no target repo. The branch's "580 commit delta" is the phantom-root artifact the RFC itself warns about; only `7f3e1ac` touches `learn/pytorch/`.
**Fix:** Reword to "force-added-then-`.gitignore`d (tracked blobs at `learn/pytorch/*.py`, only on this branch); deliberately excluded from all 4 repos — copy the 6 files out to a personal location before rewrite, they are unrecoverable from the new repos."

### NIT-5 — Cross-ref counts are approximate and slightly off
`examples → llm_surgeon`: RFC "22 import sites," actual 21 files / 33 occurrences. `gui/backend → llm_surgeon`: RFC "~60 sites," actual 4 files / 38 occurrences. These don't change any conclusion (coupling is real either way) but the precise-looking "22"/"≥15"/"~60" should be labeled as occurrence-vs-file counts. `tests → gui.backend` "≥15" matches exactly (15 occurrences). README:614 relative link verified verbatim.

### NIT-6 — §5 "fix-on-touch" line-number for the hardcoded llama.cpp path is slightly off
RFC cites `llm_engine_cpp/CMakeLists.txt:118` for the hardcoded absolute path. Actual: `:117` `set(LLAMA_CPP_ROOT` with the default, and `:107` documents the `-DLLAMA_CPP_ROOT=` override that **already exists**. So the path is already parameterized with a default fallback, not a raw hardcode. The fix (drop the absolute default / require the flag) is still valid, but the RFC overstates the fragility — there IS an override today.
**Fix:** "CMakeLists already accepts `-DLLAMA_CPP_ROOT=` (line 107); the issue is only the absolute-path *default* at line 117. Tighten the default, don't add the flag."

## What the split would BREAK that the plan DOES catch (credit where due)
The RFC correctly identifies: the forced Python-package union, the CMake `add_subdirectory` break, the examples↔research data straddle, the HFProxy 3-repo straddle (verified: plan docs split across `gui_cpp/docs/` and `llm_engine_cpp/docs/`, target is `gui/backend`), filter-repo-preserves-pointers-not-binaries, the three unpushed/gone branches, and the safety-net retention. These are real and well-handled.

## What it MISSES (split-breakage gaps)
- **The author's own branch** (BLOCKER-1).
- **`learn/pytorch/` tracked-but-ignored blobs** (SHOULD-FIX-4) — the only files in the tree that map to no target repo and are silently lost.
- **Worktree invalidation is unaddressed.** 9 live worktrees point at `/home/ai/ai-projects/llm/.git`. A 4-way split replaces that repo. Every worktree's `.git` gitlink and every branch checked out in them becomes orphaned. The RFC migrates *branches* but never says "remove all 9 worktrees (`git worktree remove`) before rewrite; they cannot survive the repo identity change." This is a concrete post-split breakage with no fix in the plan.
- **`reversibility of a 4-way split` is asserted ("expensively reversible") but not demonstrated.** No un-split procedure is sketched. The RFC says re-merging 4 histories + a third LFS rewrite, but doesn't note that the 4 repos will have diverged (independent commits) by the time anyone wants to un-split, making it not a clean inverse. Acceptable for a contingency section, but the "expensively reversible" should be "effectively one-way once independent commits land in any of the 4."

## Is the RFC decision-ready?
**Yes for the STAY decision** — that recommendation is sound, evidence-backed, unbiased, and the gates are genuinely answered. **No for executing the §3 contingency as written** — it has one blocker (author's own branch unaccounted) and a silent data-loss path (`learn/` blobs, worktree orphaning). Since the recommendation is STAY and §3 is explicitly "execute only if a tripwire fires," the RFC is **decision-ready to APPROVE the stay**, with §3 flagged as "verify the push-first list and worktree teardown before relying on it."

## TOP 3 MUST-FIX before acting
1. **Audit and push the author's own `refactor/repo-reorg` branch** (BLOCKER-1) — it is 10 commits ahead with upstream pointed at `origin/master`, not its own ref; confirm it has a dedicated remote or it joins `feat/subliminal-transfer` as strandable. The push-first list claims completeness while omitting the branch the RFC lives on.
2. **State the base-state contract** (SHOULD-FIX-2): "deltas are vs `origin/master`=`be271e1`; local master is behind 10 — `git fetch && git merge --ff-only` before acting." Without this, anyone on the local checkout reads wrong numbers.
3. **Add worktree-teardown + correct the `learn/` rationale** to §3.0/§3.6: remove all 9 worktrees before any rewrite (they orphan on repo replacement), and copy out the 6 tracked-but-ignored `learn/pytorch/*.py` files (lost in every target repo). These are the two silent-data-loss paths the migration table doesn't cover.

(Plus the cheap factual corrections: LFS breakdown 36+16 not 37+15; CMake path already has a `-DLLAMA_CPP_ROOT=` override; cross-ref counts are occurrence/file approximations.)

---

# Appendix — investigations

## A. Boundaries + cross-dependencies

Decisive finding for the test boundary: `testing/tests/` (the pytest suite the ticket maps to llm_surgeon) imports BOTH `llm_surgeon.*` (the package) AND `gui.backend.routes.sessions` / `gui.backend.app` — so the tests depend on `gui/` being importable as a package too. This confirms `gui/` and `llm_surgeon/` and `tests/` are one Python package rooted at `testing/` (the `gui.backend` import path resolves because `testing/` is on `sys.path` via `conftest.py` and `gui/__init__.py` exists). They cannot be split into separate repos without rewriting these imports or vendoring. And `llm_engine_cpp` C++ has zero Python/research dependency — it's a clean standalone library.

I have everything needed. Final summary follows.

---

# Repo-split boundary map — MAIN-264 (post-#68 master, `be271e1`)

## 1. Directory → proposed-repo mapping

| Dir | MAIN-264 target | Verdict | Notes |
|---|---|---|---|
| `theory/` | **llm-research** | Clean | Zero code refs into `testing/` or `research/` (grep for `testing/llm_surgeon`, `testing/examples`, `research/arcs` in `theory/` → empty). Self-contained KB + LaTeX. Splittable as-is. |
| `research/` | **llm-research** | Clean *if examples come too* | No Python; markdown + LFS `data/*.pt` + `figures/*.png`. But its data is written/read by `testing/examples/` scripts (see §2). |
| `research/arcs/nla-verbalizer/{data,figures}` | **llm-research** | LFS-bound | Tracked by `.gitattributes` rules `research/**/data/*.pt` and `research/**/figures/*.png`. Any split tool must carry LFS pointers + the LFS objects (note: MAIN-273 history rewrite already migrated these into LFS). |
| `docs/superpowers/` | **llm_surgeon** | Clean (but gitignored) | Plans/specs are all GUI + probe + memory-opt (the FastAPI/React GUI lineage). **Note: `docs/superpowers/` is gitignored** (`.gitignore` last line) — not in the repo, local-only. |
| `testing/llm_surgeon/` | **llm_surgeon** | Core | The Python package proper. |
| `testing/tests/` | **llm_surgeon** | Ambiguous — see §2 | Imports `llm_surgeon.*` **and** `gui.backend.*`. Bound to gui. |
| `testing/gui/` (FastAPI + React) | **llm_surgeon** | Core, but tri-coupled | Backend imports `llm_surgeon` ~60 sites; it's also the HTTP target of the C++ `HFProxyEngine` (→ llobotomy/engine). |
| `testing/examples/` | **llm-research** (per ticket) | **CONFLICTED** — straddles two repos | Imports `llm_surgeon` package (→ llm_surgeon) **and** reads/writes `research/arcs/.../{data,figures}` (→ llm-research). See §2. |
| `testing/experiments/` | **llm-research** (per ticket) | Ambiguous | Holds `test_full_recipe.yaml` + the (gitignored) `experiments.db` that `llm_surgeon/tracking.py` writes via `Path(__file__).parent.parent/"experiments/experiments.db"` → assumes `experiments/` is a sibling of `llm_surgeon/` under one root. Belongs with **llm_surgeon**, not research. |
| `testing/prompts/` | (unassigned in ticket) | → **llm_surgeon** | `default.json` read by `llm_surgeon/probe/_attribution.py` and the React `store.ts`. Belongs with the GUI/surgeon. |
| `testing/docs/` | (unassigned) | → **llm_surgeon** | All phase plans/specs for surgeon + probe + eval. |
| `testing/gui_cpp/` | **llobotomy** | Cross-links into engine + backend | `add_subdirectory(../llm_engine_cpp)` + links `llmengine::llm_engine`; runtime HTTP to `gui/backend`. |
| `testing/llm_engine_cpp/` | **llm_engine_cpp** (bundle-or-standalone TBD) | Standalone-clean C++, but two external pins | No Python/research dep. But hardcodes `/home/ai/ai-projects/llm/lib/llama.cpp` and its `HFProxyEngine` targets `gui/backend`. |
| top-level configs (`CLAUDE.md`, `pyrightconfig.json`, `.gitattributes`, `.gitignore`, `LICENSE`, `README.md`) | shared | Must be forked/split | See §3. |

## 2. Cross-boundary dependency graph (what cites what across a proposed split line)

**A. `testing/examples/` → `llm_surgeon` package (Python import; 22 refs)**
Examples `import llm_surgeon` / `from llm_surgeon.probe import ...`. Concrete: `nla_scan.py:25-26`, `nla_trajectory.py:28-29`, `nla_faithfulness.py:36-37`, `quickstart.py:3`, `probe_demo.py:24`, +14 more.
→ **Breaks** if examples→llm-research and surgeon→llm_surgeon. Fix: `pip install llm-surgeon` as a versioned dep, or submodule.

**B. `testing/examples/` ↔ `research/arcs/` data (filesystem, relative-path walk)**
`testing/examples/_nla_artifacts.py:25` → `_REPO_ROOT = Path(__file__).resolve().parent.parent.parent`, then `CACHE = _REPO_ROOT/"testing/.cache/nla_artifacts"` and DATA = `research/arcs/nla-verbalizer/data/` (the #68 committed-LFS-with-cache-fallback retrofit). Also `nla_data_manifest.py:4`, `nla_audit_findings.py:50,81`, `nla_visualize_geometry.py:5`, `nla_mid_seq_native_compare.py:12-13` (write figures into `research/arcs/.../figures/`).
→ This is the conflict: examples import the surgeon package (pull toward llm_surgeon) but read/write research data via a 3-level `parent` walk that assumes the monorepo root (pull toward llm-research). The MAIN-264 mapping (examples→llm-research) keeps B intact but **forces A across a repo boundary**.

**C. `research/arcs/.../README.md` → `testing/llm_surgeon` source (relative `../` link)**
`research/arcs/nla-verbalizer/README.md:614`: `[testing/llm_surgeon/probe/_nla.py](../../../testing/llm_surgeon/probe/_nla.py)`. Plus ~9 prose citations `llm_surgeon.probe.nla_verbalize` across the arc's observations.
→ **Breaks** (dead relative link) if research→llm-research and surgeon→llm_surgeon. Fix: rewrite to a repo-qualified URL or pinned-version ref.

**D. `testing/tests/` → BOTH `llm_surgeon.*` and `gui.backend.*`**
Import counts in `testing/tests/`: `llm_surgeon.surgery` ×46, `llm_surgeon.probe` ×31, `llm_surgeon.benchmark` ×25, … **plus `gui.backend.routes.sessions` ×8, `gui.backend.app` ×6, `gui.backend.routes.probes` ×1**.
→ Tests depend on `gui/` being an importable package. Splitting `gui` away from `tests`/`llm_surgeon` breaks the suite. They are one package (rooted at `testing/`, `conftest.py:5` puts `testing/` on `sys.path`; `gui/__init__.py` makes `gui.backend.*` resolvable). **Keep gui + tests + llm_surgeon together.**

**E. `testing/gui/backend/` → `llm_surgeon` package (Python import; ~60 sites)**
`routes/sessions.py` (≥15 import sites), `routes/probes.py` (≥9), `routes/inspect.py` (×3), `sessions.py` (×3), `gui/tests/test_sessions.py:9`. All `from llm_surgeon...`.
→ Intra-repo if both land in llm_surgeon (the ticket's intent). Confirms gui belongs with llm_surgeon. `llm_surgeon` core has **no** fastapi/uvicorn dep (it's the `[project.optional-dependencies] gui` extra in `testing/pyproject.toml`) — so the dependency is one-directional (gui→surgeon), which is the right shape for one repo.

**F. `testing/gui_cpp/` → `testing/llm_engine_cpp/` (CMake relative add_subdirectory + link)**
`gui_cpp/CMakeLists.txt:89` `add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../llm_engine_cpp llm_engine_cpp_build)`; `:120` `target_link_libraries(llobotomy PRIVATE ... llmengine::llm_engine)`; `appstate.hpp:32` `using llmengine::HFProxyEngine;`.
→ **Breaks** if llobotomy and llm_engine_cpp are separate repos. Fix: submodule, FetchContent, or bundle them (the ticket's "bundle w/ llobotomy" option). This is the strongest argument for bundling engine+llobotomy.

**G. C++ `HFProxyEngine` (in llm_engine_cpp) → Python `gui/backend` (runtime HTTP)** — the in-flight transition (question d)
`gui_cpp/src/main.cpp:470` default `http://127.0.0.1:8000`; `HFProxyEngine` (defined in `llm_engine_cpp/src/hf_proxy_engine.cpp`, per its CMake header) is "a thin C++ wrapper around the existing FastAPI server at `testing/gui/backend/`" (`gui_cpp/docs/HFPROXY_PLAN.md:5-7`, route-by-route map at `:43-67`).
→ This transition spans **three** proposed repos: llobotomy (gui_cpp main) → llm_engine_cpp (HFProxyEngine class) → llm_surgeon (the FastAPI backend it calls). It is a runtime HTTP contract (a wire API), not a build/import edge — so it survives a split *if the API contract is frozen and versioned*, but a split freezes the in-flight transition across repo lines mid-flight. Today it's live and iterating (Phases 1-3 in HFPROXY_PLAN.md). Splitting now means coordinating API changes across three repos.

**H. `llm_engine_cpp` → `lib/llama.cpp` (absolute path + script path)** — question b
`llm_engine_cpp/CMakeLists.txt:118` default `LLAMA_CPP_ROOT "/home/ai/ai-projects/llm/lib/llama.cpp"`; `scripts/demo.sh:42` `$REPO_ROOT/lib/llama.cpp/convert_hf_to_gguf.py`; `scripts/verify.sh:77` references the same prebuilt path. **`lib/llama.cpp` is NOT in the repo** — `.gitignore` line `llama.cpp/` excludes it; it's a local external checkout (exists at `/home/ai/ai-projects/llm/lib/llama.cpp`, 27 dirs).
→ Already external/shared infra, not owned by any repo. The hardcoded absolute path is fragile regardless of split. Fix on split: submodule under llm_engine_cpp, or keep as a documented external prereq (`-DLLAMA_CPP_ROOT=...`).

**I. React frontend → FastAPI backend (runtime HTTP, intra-gui)**
`gui/frontend/vite.config.ts:82,86` proxy → `http://127.0.0.1:8000` + `ws://127.0.0.1:8000`; `gui/run.sh` boots both. Intra-`gui` — not a cross-repo edge under the ticket's mapping (both → llm_surgeon).

### Would-break list (concrete, ordered by severity)
1. `gui_cpp/CMakeLists.txt:89,120` — relative `add_subdirectory` + link to `llm_engine_cpp`. Hard build break if separated. → **bundle or submodule.**
2. `testing/tests/*` importing `gui.backend.*` (≥15 sites) — suite won't import if gui split from tests. → **keep together.**
3. `testing/examples/*` importing `llm_surgeon` (22 sites) — break if examples→llm-research. → **versioned dep / submodule.**
4. `testing/examples/_nla_artifacts.py:25` + 5 sibling scripts — 3-level `parent` walk to `research/` and `testing/.cache/`; assumes monorepo root. → breaks if examples leave the monorepo root; OK if examples→llm-research alongside research.
5. `research/arcs/nla-verbalizer/README.md:614` — relative link `../../../testing/llm_surgeon/probe/_nla.py`. → rewrite to URL/pinned ref.
6. `llm_engine_cpp/CMakeLists.txt:118` + `scripts/{demo,verify}.sh` — absolute/relative `lib/llama.cpp` path. → submodule or `-D` prereq.
7. `llm_surgeon/tracking.py:11` — `parent.parent/"experiments/experiments.db"` assumes `experiments/` sibling of `llm_surgeon/`. → move `experiments/` into llm_surgeon repo, not llm-research.

## 3. Shared infrastructure (belongs to no single repo)

- **`lib/llama.cpp`** — gitignored, external CUDA-built checkout at `/home/ai/ai-projects/llm/lib/llama.cpp`. Consumed by `llm_engine_cpp` (C++ link), `llm_surgeon/llama_engine.py` + `gguf_reader.py` + `gguf_writer.py` + `export.py` (Python GGUF/convert), and `llm_engine_cpp/scripts/demo.sh` (HF→GGUF convert). **Shared by both the Python (llm_surgeon) and C++ (engine) sides** → cannot live in one repo cleanly; submodule or documented external prereq for both.
- **`testing/.cache/`** — gitignored (`.gitignore` `testing/.cache/`). Holds `models/` (HF safetensors on the G-Drive mount) and `nla_artifacts/`. Referenced by ~25 files across `testing/examples/`, `testing/tests/`, `gui/backend/routes/sessions.py`, `llm_surgeon/verify.py`, `llm_engine_cpp/scripts/demo.sh`. **Crosses every Python-touching repo** → must be a shared, configurable cache location post-split (env var, not relative path).
- **`testing/.venv/`** — gitignored; single venv with `llm-surgeon` installed editable. Backs llm_surgeon + gui + examples + tests. A split fragments this into ≥2 venvs (research vs surgeon) with overlapping deps.
- **Top-level configs:**
  - `pyrightconfig.json` (root + `testing/`) — `extraPaths: ["testing","testing/examples"]` / `[".","examples"]`; both extend the user-global `~/.config/pyright/base.json`. The `examples` extraPath exists *because* examples import the surgeon package — encodes coupling A.
  - `testing/pyproject.toml` — declares package `llm-surgeon`, `packages.find where=["."]`, `pythonpath=["."]`, `testpaths=["tests"]`, and the `gui` extra (fastapi/uvicorn). **The whole of `testing/` is one Python package root** — examples, gui, tests, experiments, prompts all resolve relative to it. This single file is the technical embodiment of "testing/ is one repo's worth of Python."
  - `.gitattributes` — the two LFS rules (`research/**/figures/*.png`, `research/**/data/*.pt`) are research-only; they'd move cleanly to llm-research. But any history-split tool must preserve LFS pointers AND copy the LFS objects (question c).
  - `.gitignore` — mixes concerns across all four proposed repos (llama.cpp build, testing/.cache, frontend node_modules/dist, LaTeX artifacts, theory binaries, docs/superpowers). Splits into per-repo `.gitignore`s.
  - `CLAUDE.md` (22 KB) — single source of the worktree discipline + per-component conventions for all four areas. Splits into four (with the worktree rule duplicated).
  - `LICENSE` (GPL-3.0) + `README.md` — root README documents all four areas (theory/surgeon/gui/research/engine + `lib/llama.cpp`); forks per repo.
- **`research/ARC_PROCESS.md`** (the #68-added SOP, 14 KB) — the dataset/arc reproducibility SOP ("raw data is a deliverable"). Conceptually research-owned, but its hard rule binds `testing/examples/` scripts (which produce the committed `data/`/`figures/`) — so the SOP governs code that the ticket puts in a different repo than the SOP. Another instance of the examples↔research straddle (B).

## 4. Honest counter-argument weight (for the RFC)

The cross-dep graph shows the split is **possible but lossy**, and the seams fall on exactly the edges the monorepo currently makes free:

- **Three genuine break-points need active fixes**, not cosmetic: (1) gui_cpp→llm_engine_cpp CMake edge (F) forces bundle-or-submodule; (2) tests→gui.backend (D) forces gui+tests+surgeon to stay one repo regardless; (3) examples→llm_surgeon (A) vs examples↔research/data (B) is an **irreducible straddle** — `testing/examples/` cannot cleanly belong to either llm-research or llm_surgeon, because it imports one and read/writes the other's data via a monorepo-root relative walk. Whichever repo it lands in, one of {package import, data path} becomes a cross-repo edge needing a versioned-dep/submodule.
- **The in-flight gui_cpp↔gui/backend HTTP transition (G)** spans three of the four proposed repos and is actively iterating (HFPROXY_PLAN phases 1-3). Splitting now freezes a moving API contract across repo lines.
- **Shared infra (§3)** — `lib/llama.cpp`, `testing/.cache`, the venv, and the dataset SOP — has no clean owner; each split multiplies these into per-repo copies or submodules.
- Against that: the monorepo discipline (worktrees + scope labels + per-session branches per `CLAUDE.md`) is already delivering scope-bounded PR diffs — the stated *benefit* of splitting (clean per-component review) is **already achieved without splitting**. The clean separations that *do* exist (theory has zero code cross-deps; llm_engine_cpp C++ has zero Python/research deps) are the cheap wins a split would capture — and both can be captured in-monorepo by tightening the existing scope labels.

**Net:** the cleanest split lines are theory/ (fully independent) and llm_engine_cpp C++ (no inbound Python). Everything in `testing/` Python (llm_surgeon + gui + tests + examples + experiments + prompts) is one tightly-woven package with `testing/pyproject.toml` as its single root, and `examples` additionally welds it to `research/`. A four-repo split incurs ≥7 break-fixes, ≥2 submodule/versioned-dep relationships, four CI pipelines, and fragments four pieces of shared infra — to gain isolation the worktree discipline already provides. The RFC has honest grounds to land on **"stay monorepo + clean up"** (e.g., formalize the `lib/llama.cpp` external-prereq, lift `testing/.cache` to an env-var-configured path, resolve the examples↔research straddle by either co-locating or introducing an explicit `data/` resolver contract), with theory/ and llm_engine_cpp earmarked as the only two pieces whose independence is real enough to extract later if a split is ever forced.

Key files for the RFC to cite: `testing/pyproject.toml`, `testing/conftest.py:5`, `testing/examples/_nla_artifacts.py:25`, `research/arcs/nla-verbalizer/README.md:614`, `testing/gui_cpp/CMakeLists.txt:89,120`, `testing/llm_engine_cpp/CMakeLists.txt:118`, `testing/gui_cpp/docs/HFPROXY_PLAN.md:5-7`, `testing/llm_surgeon/tracking.py:11`, `.gitattributes`, `.gitignore`.

## B. Branches + history + LFS

Confirmed: `docs/nla-arc-attribution`'s content IS on master (commit `38ac87d` = "Merge pull request #67"; `git cherry` shows all its real commits as `-` = already-present-by-patch-id; the lone `+` is the stale pre-rewrite root `70b4327`). The worktree branch is just behind `origin/master` and `origin/docs/nla-arc-attribution: gone` confirms the remote branch was deleted post-merge.

`nla-research-backup`'s 29 "unique" commits are patch-id-equal duplicates of `nla-research` work under different SHAs (the same MAIN-71/MAIN-34/MAIN-48 research commits), making it a true superseded safety-net.

All data verified. Here is the complete analysis.

---

# Repo-Reorg Pre-Split Analysis

Investigated read-only at `/home/ai/ai-projects/llm/.claude/worktrees/repo-reorg` (branch `refactor/repo-reorg` @ `be271e1`, which == `origin/master`). The local `master` worktree shows `[origin/master: behind 10]` — the main checkout is stale, but `refactor/repo-reorg` is current.

## ★ Key corrections to the task's stated assumptions

- **subliminal is 8 unpushed commits, not 9.** The 9th (`f04af40` per-arc reorg) is already on `origin/master` via PR #68. The 8 commits `07fb779..dd72c0c` are the genuine LOCAL-ONLY work. Still: **MUST be pushed before any split or it's lost** — confirmed on zero remotes.
- **`git cherry origin/master <branch>` over-reports** here. Every long branch shows the stale pre-rewrite root `70b4327` as a phantom `+` because the 2026-05-31 rewrite re-SHA'd history. Use `git rev-list be271e1..<branch>` for true delta, and `git merge-base` to confirm lineage. All branches are legitimate descendants of the **rewritten** master (verified: `38ac87d`/`9059768` appear in their first-parent chains) — none sit on foreign/pre-rewrite roots; they are merely *behind* the current tip.

---

## 1. Branch / Worktree migration table

Target repos inferred from path footprint: top-level dirs are `theory/`, `testing/` (which holds `llm_surgeon`, `gui`, `gui_cpp`, `llm_engine_cpp`, `examples`, `tests`), `research/` (arcs), `docs/`. A plausible split is **`llm-theory`** (theory/), **`llm-surgeon`** (testing/llm_surgeon + gui + tests + examples), **`llm-engine`** (testing/llm_engine_cpp + gui_cpp), **`llm-research`** (research/ arcs). No on-disk reorg/split plan doc exists yet (`research/plans/` absent; nothing matching reorg/split/monorepo tracked).

| Branch | Tip | Remote / Worktree | True unmerged (vs be271e1) | Classification | Target repo |
|---|---|---|---|---|---|
| `refactor/repo-reorg` | `be271e1` | local-only, == origin/master | 0 | **active** (this session) | n/a — orchestration branch |
| `refactor/research-arcs` | `48b2a07` | `[origin: gone]`, wt `research-arcs` | 0 (== origin/master content) | **merged — deletable** (PR #68) | retire before split |
| `docs/nla-arc-attribution` | `f84c6a9` | `[origin: gone]`, wt `docs+nla-arc-attribution` | content on master via #67; tip 2 commits behind | **merged — deletable** (PR #67); cherry confirms patch-id-equal | retire before split (content → `llm-research`) |
| `feat/subliminal-transfer` | `dd72c0c` | **NO REMOTE**, wt `subliminal-semantics` | **8 (LOCAL-ONLY)** | **active — MUST PUSH FIRST** | `llm-research` (`research/arcs/subliminal/` + `testing/examples/`) |
| `session/llobotomy` | `69ca315` | `[origin/session/llobotomy]` pushed | 533 | **active** | `llm-engine` (gui_cpp + llm_engine_cpp) |
| `session/theory` | `90718a6` | local-only (no remote tracking) | 531 | **active** | `llm-theory` (theory/kb, theory/series) |
| `session/nla-research` | `61c0c77` | `[origin: gone]`, wt `nla-research` | 601 | **active** (= PR #11 head, but post-#11 work) | `llm-research` (research/arcs/nla-verbalizer) |
| `session/nla-research-backup` | `4293b67` | local-only, no worktree | 558; 29 "unique" are patch-id dups of nla-research | **safety-net** (superseded by nla-research) | retire before split |
| `feat/engine-arch-coverage` | `84b8da7` | `[origin/feat/engine-arch-coverage]` pushed | 566 | **active** | `llm-engine` (llm_engine_cpp) |
| `learn/pytorch-tutorials` | `7f3e1ac` | local-only, wt `learn` | 580 | **stale/personal** — `learn/` is `.gitignore`d scratch (116 KB, not on master) | **none** — does not migrate; keep local or archive separately |
| `wip/multi-session-checkpoint` | `2922f55` | local-only, no worktree | 527; NOT an ancestor of any session branch | **safety-net** (explicit in CLAUDE.md; sessions already split out) | retire before split |
| `origin/backup/pre-rewrite-master` | `5549c3d` | remote-only | — | **safety-net** (rollback ref for the rewrite) | retire after split confidence |
| `origin/master-rewritten` | `38ac87d` | remote-only | — | **safety-net** (rewrite landing ref; == PR #67 merge, now behind by #68) | retire after split confidence |
| `origin/feat/gui-ui-polish` | `58c4a9d` | remote-only, no local | merged (in master) | **merged — deletable** | retire before split |
| `origin/feat/llm-engine-modelview` | `aa171eb` | remote-only, no local | likely merged (PR #1) | **merged — deletable** (verify) | retire before split |

**MUST-PUSH-FIRST list (work lost if split proceeds without it):**
1. `feat/subliminal-transfer` — 8 commits, **zero remotes**. Top priority.
2. `session/theory` — 531 commits, **no remote tracking branch** (only local + worktree). Push before split.
3. `session/nla-research` — `origin: gone` (remote deleted post-#11); 601 commits exist only locally + worktree. Re-push or the long arc history is local-only.

Pushed and safe: `session/llobotomy`, `feat/engine-arch-coverage` (both have live `origin/` tracking).

---

## 2. LFS ownership map

LFS rules (`.gitattributes` at master, committed by the 2026-05-31 rewrite):
- `research/**/figures/*.png filter=lfs diff=lfs merge=lfs -text`
- `research/**/data/*.pt filter=lfs diff=lfs merge=lfs -text`

`git lfs ls-files` = **52 objects, all under `research/arcs/nla-verbalizer/`** (37 figures + 15 `.pt` data files). Local LFS store `/home/ai/ai-projects/llm/.git/lfs` = 31 MB. All 52 show `*` (present locally, not pointer-only). Remote is `git@github.com:skothr/llm.git`; objects live in GitHub's LFS store from the prior `git lfs migrate import --everything` pass (per `project-repo-lfs-rewrite` memory, MAIN-273).

**Ownership: 100% of current LFS objects belong to the `llm-research` target** (the `research/arcs/` subtree). `llm-theory`, `llm-surgeon`, `llm-engine` carry **no** LFS objects. The `.gitattributes` LFS rules are `research/`-scoped, so only the research repo needs the LFS config; the other three target repos should *drop* the LFS rules from their `.gitattributes` to avoid phantom-smudge confusion.

**How filter-repo / subtree handle LFS — the load-bearing gotcha:** history rewrites preserve only the LFS *pointer files* (the 130-byte `version/oid/size` text blobs), NOT the binary objects in `.git/lfs/objects`. After splitting, the new repo's commits reference OIDs that **do not yet exist in the new repo's GitHub LFS store**. You must explicitly re-upload them:
```bash
# in the new llm-research clone, after the split + new remote is set:
git lfs fetch --all <old-origin>     # ensure all 52 objects are local
git lfs push --all <new-origin>      # re-upload to the NEW repo's LFS store
```
Skip this and a fresh clone of the research repo gets dangling pointers (`smudge filter lfs failed`). This is the same class of issue the prior rewrite hit (memory: "without git-lfs, figures show as phantom modifications").

---

## 3. Recommended history-split method

**Use `git filter-repo`, not `git subtree split`.** Rationale for THIS repo:

- **subtree split** produces a single-prefix linear history but: (a) mangles merge commits (the un-squashed PR #11 merge `9059768` carrying 36 real arc commits would be flattened or dropped), (b) has **no LFS awareness** — it rewrites trees and silently strips/relocates pointers, and (c) only handles one prefix at a time with weaker rename/path-move tracking. The repo's value is the *preserved* authorship/dates from the rewrite; subtree would partly undo that.
- **filter-repo** keeps full merge topology, supports `--path`/`--path-rename`/`--invert-paths`, and (critically) treats LFS pointer blobs as ordinary content — it rewrites the pointers in-tree without corrupting them, so a subsequent `git lfs push --all` re-homes the binaries cleanly. This is the **same tool the 2026-05-31 rewrite used**, so the path-handling is already proven on this tree.

**Implication of the prior (2026-05-31) rewrite:** this is a *second* rewrite. Two consequences:
1. Branches that weren't reset onto the rewritten master (`session/theory`, `session/nla-research`, `learn/*`, the wip/backup nets) carry the old pre-rewrite root in their ancestry. After filter-repo runs `--path research/`, those branches' pre-rewrite portions either drop out or re-SHA again. **Push the must-push-first list onto current `origin/master` lineage BEFORE splitting** so filter-repo operates on coherent, rewrite-consistent history — otherwise you re-fragment.
2. Keep `origin/backup/pre-rewrite-master` AND add a fresh `backup/pre-split-master` tag/branch before the split (third safety net). Don't delete the rewrite-era safety nets until each new repo verifies clone+`lfs pull` works.

**Concrete command shape (per target repo — run on a fresh `--mirror` clone each time, filter-repo refuses a non-fresh clone):**

```bash
# 0. Push the must-push-first branches first (subliminal, theory, nla-research).
#    Then make a pre-split safety ref:
git push origin be271e1:refs/heads/backup/pre-split-master

# 1. llm-research  (owns ALL 52 LFS objects)
git clone --mirror git@github.com:skothr/llm.git llm-research.git
cd llm-research.git
git filter-repo --path research/ --path .gitattributes --path LICENSE
#   -> .gitattributes retained so LFS rules survive into the new repo
git remote add origin git@github.com:skothr/llm-research.git
git push --all && git push --tags
git lfs fetch --all && git lfs push --all origin     # RE-HOME the binaries

# 2. llm-theory  (no LFS)
git clone --mirror git@github.com:skothr/llm.git llm-theory.git
cd llm-theory.git
git filter-repo --path theory/ --path LICENSE --path-rename theory/:./   # optional un-nest
#   drop research-scoped LFS rules from .gitattributes afterward (no LFS objects here)

# 3. llm-engine  (no LFS)
git clone --mirror git@github.com:skothr/llm.git llm-engine.git
cd llm-engine.git
git filter-repo \
  --path testing/llm_engine_cpp/ \
  --path testing/gui_cpp/ \
  --path LICENSE

# 4. llm-surgeon  (no LFS)
git clone --mirror git@github.com:skothr/llm.git llm-surgeon.git
cd llm-surgeon.git
git filter-repo \
  --path testing/llm_surgeon/ \
  --path testing/gui/ \
  --path testing/tests/ \
  --path testing/examples/ \
  --path testing/experiments/ \
  --path testing/prompts/ \
  --path testing/pyproject.toml \
  --path LICENSE
```

`--invert-paths` is the inverse (e.g. `--path research/ --invert-paths` to *strip* research from a kept-monorepo). For a clean N-way split, prefer N positive `--path` clones over invert-subtraction — easier to audit which repo owns what.

**Sandbox/auth notes for execution (from memory):** `git filter-repo` lives at `~/.local/lib/pythonX/site-packages` (hidden by the /home/skothr read-deny) — copy `git_filter_repo.py` to `$TMPDIR` and run `python3 $TMPDIR/git_filter_repo.py`, or add `~/.local` to allowRead. SSH push is blocked by the sandbox and bot `skothr-cc` is read-only — **all `git push` / `git lfs push` steps must be handed to the user to run.** `git lfs migrate` refuses a dirty tree when figures are raw blobs vs LFS-attributed; neutralize with `git config filter.lfs.clean cat` first if needed.

---

## Open items the user should decide before executing
- **Target repo boundaries are inferred, not specified** — no on-disk reorg plan exists. Confirm the 4-repo split (theory / engine / surgeon / research) and where shared files go: `LICENSE` → all; `CLAUDE.md`, `docs/`, top-level `README.md`, `pyrightconfig.json`, `.gitignore` are cross-cutting and need explicit assignment or duplication.
- **`testing/gui` placement is ambiguous** — the React frontend pairs with `llm_surgeon` (backend) but `gui_cpp` (its C++ successor) goes to `llm-engine`. Either keep both GUIs with surgeon, or split GUI into its own repo.
- **`learn/pytorch-tutorials`** is gitignored personal scratch (116 KB) — does not belong in any product repo. Archive locally or to a personal repo; do not migrate.
- Verify `origin/feat/llm-engine-modelview` and `origin/feat/gui-ui-polish` are fully merged (they appear to be) before deleting.

## C. Junk / cleanup inventory

I have a complete picture. Here is the do-anyway cleanup inventory.

---

# Cleanup Inventory — repo-reorg (do-anyway track)

Repo: `/home/ai/ai-projects/llm/.claude/worktrees/repo-reorg`, branch `refactor/repo-reorg`. 805 tracked files. The big binary artifacts are already in git-LFS (52 files) and are reproducible deliverables, not junk. The repo is unusually clean — no stray `.db`, `node_modules`, `.pyc`, `.tsbuildinfo`, `.DS_Store`, or LaTeX `.aux`/`.log` tracked. `debug-images/` is untracked (not committed). So the inventory is small and centers on three things: superpowers docs, two gitignore/tracking mismatches, and one orphan doc.

## 1. Superpowers docs (51 files) — KEEP (do NOT delete)

These are **not** obsolete process artifacts. They are heavily and load-bearingly cross-referenced as the canonical specs/plans the live code was built from:

- `theory/README.md`, `theory/plans/README.md`, and project `CLAUDE.md` (lines 115, 150) cite `theory/docs/superpowers/specs/...` and `.../plans/` as the **live structural home** for theory specs/plans — deleting them breaks documented repo structure.
- `testing/llm_surgeon/probe/_intervention.py:274` (a code docstring) cites `docs/superpowers/specs/2026-04-17-phase3-activation-patching-design.md` as the design-of-record for the activation-patching implementation.
- Plans cross-reference their own specs by path/commit-SHA throughout (e.g. phase plans → `..._design.md`), forming a coherent spec→plan→commit audit trail.
- 12 distinct spec filenames are cited from outside their own file (`2026-04-08-llm-surgeon-design.md` alone has 7 inbound refs).

Recommendation: **KEEP all 51.** They function as committed design history that the code and docs actively point to. The only sub-item worth flagging (see §2) is that the `theory/docs/superpowers/` tree is the live one while a *separate, gitignored* `docs/superpowers/` tree at repo root holds 8 stale files.

## 2. Gitignore / tracking mismatches — ARCHIVE-INTENT, decide explicitly

Two directories are listed in `.gitignore` yet have committed files (committed *before* the ignore rule; git keeps tracking already-tracked paths, and `git check-ignore` returns exit 1 for them while matching hypothetical new files at exit 0 — confirmed). This is a real "intent says ignore, reality says tracked" inconsistency:

| Path | Tracked files | .gitignore line | Reference check | Recommendation |
|---|---|---|---|---|
| `docs/superpowers/` (repo-root, 8 files: 4 plans + 4 specs, all 2026-04-11→04-15 GUI/probe/memory work) | 8 | line 64 `docs/superpowers/` | Referenced only from *within themselves* (the GUI plans/specs cross-cite each other). No code or top-level doc cites the root-level `docs/superpowers/` path — all live citations point at `testing/docs/superpowers/` and `theory/docs/superpowers/`. | **ARCHIVE then untrack.** These are the original superpowers GUI specs that predate the per-area `testing/`/`theory/` split. They have historical value (they're the design-of-record for the live-probe GUI) but the ignore rule says they shouldn't be tracked at root. Either (a) move them under `testing/docs/superpowers/` to match where the live ones live and where the ignore intent is the `testing/` subtree is tracked, or (b) `git rm --cached` them to honor the existing ignore rule. Do NOT hard-delete — the GUI design history is the only copy. |
| `theory/archive/2026-05-03-pre-expansion/` (5 files: GLOSSARY.md, Makefile, the v1 .tex + .pdf, the v1 HTML viz) | 5 | line 51 `theory/archive/` | `CLAUDE.md` and `theory/README.md` explicitly describe this as "v1 single-LaTeX-doc state preserved as a snapshot" and point at it as the archived predecessor. So the *content* is intentionally retained, but the *gitignore rule contradicts the tracking*. | **KEEP content, RESOLVE the contradiction.** Either drop `theory/archive/` from `.gitignore` (intent: snapshot is meant to be tracked) or `git rm --cached` it (intent: snapshot lives only on disk). The docs treat it as a committed snapshot, so dropping line 51 from `.gitignore` is the consistent fix. |

Sub-flag inside `theory/archive/`: `theory/archive/2026-05-03-pre-expansion/build/llm-core-architecture/llm-core-architecture.tex` is a **build-output `.tex`** committed inside the archive snapshot (double-junk: a build artifact within an archive). If the snapshot stays tracked, this generated `build/` file is the one piece safe to drop from it.

## 3. Orphan / superseded doc — ARCHIVE or DELETE

| Path | Reference check | Recommendation |
|---|---|---|
| `testing/docs/superpowers/specs/2026-04-08-llm-surgeon-design-v1.md` | **Zero inbound references** (grep across the whole repo for `llm-surgeon-design-v1` returns nothing). The current `2026-04-08-llm-surgeon-design.md` (the v2, 7 inbound refs) supersedes it. The v1 still targets the abandoned "LLaMA 3 8B, 32 layers" framing rather than the actual TinyLlama/OpenLlama dev models. | **ARCHIVE** (lowest-risk: it's superseded design history with no live citers). Safe to **DELETE** if you don't value pre-v2 design archaeology — nothing points at it. Not a keep: it's the only file in the superpowers set with zero inbound references and an explicitly-superseded `-v1` suffix. |

## 4. Confirmed-NOT-junk (looks removable, is actually load-bearing)

- **`research/arcs/nla-verbalizer/data/*.pt` (16 files) + `.../figures/*.png` (37 files), ~28 MB** — these dominate `du` but are **git-LFS tracked** and declared reproducible deliverables in `research/ARC_PROCESS.md` ("Raw data is a deliverable"; figure-render scripts and `nla_audit_findings.py` load them). Do not gitignore or delete.
- **`testing/llm_engine_cpp/libs/cpp-httplib/` + `libs/nlohmann_json/` (~1.6 MB, 2 vendored MIT libs)** — intentional bundled deps, wired into `CMakeLists.txt` (lines 43, 47, 80) and `#include`d by `src/hf_proxy_engine.cpp`. Both ship their `LICENSE`. **KEEP** (vendoring single-header libs is a deliberate choice, not stale cruft).
- **`research/archive/` (3 obs + README)** — its own README declares it deliberate archaeology ("kept for archaeology… not load-bearing for an active research claim"). This is a *properly-curated* archive, the model to follow. **KEEP.**
- **`theory/docs/superpowers/specs/2026-04-06-llm-core-architecture-design.md`** — referenced by the phase0 restructure plan as "existing v1 spec; leave in place — historical context for v2." **KEEP** (the phase0 plan would dangle without it).
- **10 of 11 `.gitkeep` files** are now redundant (their dirs have 4–99 real siblings — e.g. `theory/kb/excerpts/.gitkeep` guards a dir with 99 files). Harmless, but **optionally removable** as tidy-up. The one genuinely still-needed `.gitkeep`: `theory/sources/forums/.gitkeep` (0 siblings — dir would vanish without it). **KEEP that one.**

---

## Summary of recommended actions

- **DELETE (zero risk):** none mandatory. Optional: `theory/archive/.../build/llm-core-architecture/llm-core-architecture.tex` (committed build output); 10 redundant `.gitkeep` files (dirs now populated).
- **ARCHIVE / relocate:** root-level `docs/superpowers/` (8 files) → move under `testing/docs/superpowers/` or `git rm --cached` to honor the existing ignore rule; `testing/docs/superpowers/specs/2026-04-08-llm-surgeon-design-v1.md` (orphan, superseded, zero refs).
- **RESOLVE gitignore contradiction:** `theory/archive/` (line 51) and `docs/superpowers/` (line 64) are gitignored but partly tracked — pick tracked-or-not and make `.gitignore` + index agree.
- **KEEP (do not touch):** all 51 superpowers docs except the v1 orphan; the LFS `.pt`/`.png` arc deliverables; the 2 vendored C++ libs; `research/archive/`; the v1 theory design spec; `theory/sources/forums/.gitkeep`.

The single highest-value, lowest-risk action is resolving the two gitignore/tracking mismatches (§2) — that is a genuine inconsistency, independent of any reorg/split decision, and it is what makes the "is this dir supposed to be in the repo?" question ambiguous today.

## D. Steelman: stay monorepo

I have all the evidence I need. The picture is concrete:

- **9.3G working tree is 99% model cache + llama.cpp + node_modules** — all gitignored or vendored; none of it moves under a split. The git-tracked footprint is small (~780 tracked files), and the figure bloat (16.8 MB, 36 files) is already on LFS.
- **`gui_cpp` → `llm_engine_cpp` is a hard `add_subdirectory(../)` CMake link** — splits into two repos breaks the relative path on day one.
- **No project CI exists yet** — so "four CI pipelines" is a future cost, not a present one (honest framing required).
- **9 active worktrees right now**, clean PR-per-branch merge cadence — the discipline is demonstrably working.
- **Cross-component atomic commits are real but low-volume** (2 research+surgeon, 3 examples+surgeon in last 300; 0 surgeon+cpp) — honest: the atomic-commit loss is real but modest, not catastrophic.

Here is the RFC.

---

# RFC: Stay Monorepo + Clean Up (counter-proposal to MAIN-264's 4-way split)

**Status:** Steelman of the STAY option. Honestly weighs both directions. Recommends staying, with explicit tripwires that would flip the decision.

**TL;DR:** The monorepo discipline (worktrees + `project:` scope labels + per-session branches + single venv/config) is already solving the problems the split is meant to solve, with live evidence: 9 active worktrees, a clean PR-per-branch merge cadence, and a successful parallel-session worktree split. The 4-way split's costs are concrete and front-loaded; its benefits are mostly speculative or already-captured. The single strongest pro-split fact — repo heaviness — was already fixed by the LFS migration and is dominated by gitignored model cache that no split would move. Recommendation: stay monorepo, do the targeted cleanups below, and arm four tripwires that would justify revisiting.

---

## 1. What the monorepo discipline is currently buying (and the evidence it works)

The claim "discipline is already working" is not aspirational. Measured state of `/home/ai/ai-projects/llm`:

**a) Worktrees give per-session isolation that the split promises but doesn't improve on.**
`git worktree list` shows **9 live worktrees**, each on its own branch:
`docs/nla-arc-attribution`, `feat/engine-arch-coverage`, `learn/pytorch-tutorials`, `session/llobotomy`, `session/nla-research`, `refactor/repo-reorg`, `refactor/research-arcs`, `feat/subliminal-transfer`, `session/theory`. Each has its own index; `git status` in one shows only that session's mods. The contamination incident the CLAUDE.md cites (commit `c844d24`, a theory commit that swept in 8 unrelated rename-stage files) was a *pre-worktree* failure — the worktree convention was the fix, and it has held since. **A 4-way split does not isolate sessions any better than worktrees already do**; it just replaces one isolation mechanism (cheap, local, no remote round-trip) with another (4 clones, 4 remotes).

**b) `project:` scope labels + per-session branches give lifecycle separation without physical separation.**
The merge log shows clean scope-bounded PRs landing independently: PR #67 (docs/nla), #66 (fix/nla), #56 (docs/claude-md), #16 (gui_cpp IGFD fix), #7/#6 (gui_cpp), #5/#4 (llm_engine), #3/#2 (gui_cpp), #1 (ModelView substrate). Each PR is already a coherent single-scope unit — exactly what split advocates want from separate repos — produced *without* separate repos. The Linear `project:<basename>` label scheme already partitions the issue tracker by component.

**c) Atomic cross-component commits exist and are cheap to keep.**
Measured over the last 300 commits: 2 commits touch `research/` + `llm_surgeon` together, 3 touch `testing/examples/` + `llm_surgeon` together. These are the commits where a research script and the API it calls evolve in lockstep — a single reviewable diff, a single revert if wrong. The volume is low (honesty: this is not the load-bearing argument), but the *capability* costs nothing to retain and is structurally impossible across a repo boundary without a coordinated two-PR dance + version bump.

**d) Single venv / single pyright config / single build context.**
One `testing/.venv`, one `testing/pyproject.toml`, one root + one testing `pyrightconfig.json`. The zero-diagnostics bar is enforced once, repo-wide. `gui_cpp`'s CMake does `add_subdirectory(../llm_engine_cpp)` and links `llmengine::llm_engine` directly — a single build tree, no package-registry round-trip, no version skew.

**Net:** the discipline is buying isolation, lifecycle separation, atomic cross-cuts, and unified tooling — and four of those four are demonstrably operating right now.

---

## 2. What the 4-way split actually costs (concrete, mostly front-loaded)

**a) Breaking a live in-tree build dependency (the hardest cost).**
`testing/gui_cpp/CMakeLists.txt:89` literally does `add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../llm_engine_cpp ...)` and line 120 links `llmengine::llm_engine`. The MAIN-264 ticket flags "bundle w/ llobotomy or standalone TBD" for `llm_engine_cpp` precisely because *this coupling has no clean cut*. Splitting them forces: a git submodule (pinning friction, detached-HEAD foot-guns, LFS-pointer interaction) or a CMake `FetchContent`/`find_package` against a published engine artifact (you now version-release a library that has shipped zero external consumers). Either way the `../llm_engine_cpp` relative path breaks on day one and the tight edit-build-edit loop across the GUI/engine boundary — currently one `cmake --build` — becomes a cross-repo release cycle.

**b) Cross-repo reference rewriting.**
`research/` cites `llm_surgeon` API paths in 15 files; `testing/examples/` has **21 scripts** that `import llm_surgeon`. Under llm-research-vs-llm_surgeon split these become cross-repo references. Options are both lossy: (i) rewrite every citation/path (churns provenance in observation files whose value *is* their fixed historical record), or (ii) submodule + pin (research now carries a versioned `llm_surgeon` dep and every API evolution needs a coordinated bump). The NLA observation files are dated lab-notebook entries — rewriting their internal API paths post-hoc damages the thing that makes them citable.

**c) Four CI pipelines, four tooling configs — but be honest: CI doesn't exist yet.**
There is **no `.github/workflows/` in the project** today (the only workflows on disk are vendored inside `lib/llama.cpp`). So "four CI pipelines" is a *future* cost, not a present regression — splitting now means standing up 4 pipelines instead of 1 when CI is added. Still a real 4× multiplier on config/secrets/caching/runner setup, plus 4× pyright/clang-tidy/pre-commit configs to keep in sync, but framed honestly it's "4× the not-yet-built thing," not "lose your working CI."

**d) LFS re-migration risk — history was already rewritten once.**
Per the repo-LFS memory (MAIN-273, 2026-05-31): master was already force-pushed once to purge `node_modules` from history and move 36 figures (~16.8 MB) to LFS. A history split (`git filter-repo` or `git subtree split`) is a *second* history rewrite, and both tools need careful LFS-pointer handling — a split that mangles the `research/observations/figures/*.png` LFS pointers re-introduces the exact bloat the first rewrite removed, in a new repo, with new SHAs. Every existing clone, worktree, and the `origin/backup/pre-rewrite-master` safety branch would need re-coordination. One history rewrite is survivable; chaining a second one onto a 4-way fan-out multiplies the blast radius.

**e) The in-flight gui_cpp ↔ gui/backend HTTP transition would span two would-be repos.**
The HFProxy/HttpBackend transition (`testing/gui_cpp/docs/HFPROXY_PLAN.md`, `testing/gui_cpp/src/main.cpp`) is mid-flight: `gui_cpp` (→ llobotomy repo) talks to the FastAPI backend in `testing/gui` (→ llm_surgeon repo). Splitting mid-transition means an unfinished protocol negotiation now crosses a repo boundary — every iteration on the wire format becomes a two-repo change with no atomic commit to bisect against. **Splitting during an active cross-component transition is the worst possible timing.**

**f) Contributor/onboarding friction.**
One clone, one `pip install -e .`, one CMake configure today. Four repos means a documented multi-clone bootstrap, inter-repo version compatibility matrix, and "which repo does this bug live in" routing — for a single-owner research workspace with zero external contributors.

---

## 3. What the split actually *solves* — and how each is solvable without splitting

This is the honest core: name each genuine motivation, then show it's already addressed or addressable in place.

| Split motivation | Reality / in-place fix |
|---|---|
| **Repo too heavy / slow clone** | The 9.3 G working tree is **15 M `.cache` + 910 M `lib/llama.cpp` + 123 M `node_modules`** — all gitignored or vendored, none of it moves under a split. The *git-tracked* footprint is ~780 files; the figure bloat (16.8 MB / 36 files) is **already on LFS** (`git lfs ls-files` → 36). `.git` is 161 M. The heaviness argument was the strongest pro-split fact and the LFS migration already spent it. |
| **Cross-session contamination** | Already solved by worktrees (§1a). Splitting adds nothing here. |
| **Different lifecycles (theory vs C++ engine vs research)** | Already expressed by top-level dirs + `project:` scope labels + per-session branches. Independent PRs already merge on independent cadences (§1b). |
| **"I want to review just llm_engine_cpp"** | A scope-bounded branch + `gh pr create` already produces a single-component PR (PRs #1, #4, #5 were engine-only; #2, #3, #7, #16 were gui_cpp-only). `/ultrareview <PR#>` reviews exactly that unit. |
| **Public release of one component** | This is a *real* future need that a split serves — but it's a tripwire (§4), not a present condition. Today nothing is public. |
| **Independent dependency versioning** | A single venv with one resolved dependency set is a *feature* for a research workspace — it guarantees `research` and `llm_surgeon` run against the same torch. Splitting introduces a version-skew surface that doesn't currently exist. |

**Cleanup-instead-of-split punch list (the actual recommendation):**
1. Finish the LFS migration story — confirm no large blobs leak outside the `research/observations/figures/*.png` rule; extend `.gitattributes` if other figure dirs grow.
2. Codify the `gui_cpp`↔`llm_engine_cpp` boundary as an *internal* library contract (it already is `llmengine::llm_engine`) — document the interface so a *future* split is mechanical, without paying for it now.
3. Add the single CI pipeline that doesn't exist yet, with per-component job matrices (one workflow, N jobs) — gets 90% of "independent CI signal" at 25% of the cost.
4. Keep `project:` labels as umbrella-and-sub (see §5), not 4 flat labels — preserves cross-component issue queries.
5. Finish the gui_cpp↔backend HTTP transition *before* even re-opening the split question.

---

## 4. Tripwires — the conditions under which splitting *does* become worth it

The recommendation is "stay," not "never split." These are the specific, falsifiable conditions that flip it. When any one fires, re-open MAIN-264:

1. **An external contributor lands on exactly one component.** The moment someone who shouldn't see/clone the whole tree needs to contribute to (say) `llm_engine_cpp`, the multi-clone friction inverts: a single scoped repo becomes cheaper than onboarding them to the monorepo. (Single-owner today → cost is all on the split side.)
2. **A component goes public / open-source on its own.** If `llm_engine_cpp` or `llobotomy` is published as a standalone library/app with its own release cadence and issue tracker, it needs its own repo, license-scoping, and CI badge. (License is already clean per CLAUDE.md, so this is mechanically ready when the *decision* is made.)
3. **CI wall-clock crosses a pain threshold.** Once the single pipeline exists and a full run exceeds ~10–15 min wall clock (e.g. C++ matrix builds + torch test suite + frontend e2e all serialized), per-repo CI to parallelize independent components starts paying for itself. Until CI exists at all, this can't fire.
4. **The `gui_cpp`↔`llm_engine_cpp` interface stabilizes into a true published API with ≥1 third consumer.** The day a *third* tool (beyond llobotomy) wants to link `llmengine::llm_engine`, the engine has earned standalone-library status and the submodule/package cost is justified by reuse rather than imposed by fiat.

Absent all four, the split is paying real front-loaded costs (§2) to buy benefits that are either already captured (§3) or speculative (§4).

---

## 5. Decisions on the open questions (under STAY)

- **(a) cross-repo refs:** N/A under stay — no rewrite, no submodule, no version-pin. `research`→`llm_surgeon` stays an in-tree import; the 21 example scripts and 15 observation files keep their paths intact.
- **(b) `lib/llama.cpp`:** stays shared infra at repo root (vendored, gitignored from blob-tracking, 910 M). No ownership question to resolve.
- **(c) history-split tool:** N/A — no second history rewrite. (The first one, MAIN-273, is the only one we run.)
- **(d) gui_cpp↔backend transition:** stays a single-repo atomic transition; finish it before reconsidering split (it's tripwire-adjacent — a clean finish *enables* a future split rather than blocking it).
- **(e) Linear labels:** keep the existing `project:<basename>` umbrella-and-sub scheme, not 4 flat repo labels. Umbrella preserves cross-component queries ("all open work touching the GUI stack") that 4 flat labels would fragment.

---

## Honest bottom line

The split is not wrong in principle — tripwire 1 or 2 firing makes it correct. But filed *now*, against *current* state, it pays concrete front-loaded costs (break a live CMake `add_subdirectory` link, rewrite or submodule 36 cross-references, a second LFS history rewrite, splitting an in-flight HTTP transition across two repos, 4× the CI that doesn't yet exist) to solve problems that are already solved (contamination → worktrees; heaviness → LFS + gitignored cache; lifecycle → labels + dirs) or not yet real (public release, external contributors). **Recommendation: stay monorepo, execute the §3 cleanup punch list, document the engine boundary so a future split is mechanical, and arm the four §4 tripwires.**

---

Relevant paths:
- `/home/ai/ai-projects/llm/testing/gui_cpp/CMakeLists.txt` (lines 79-120: the live `add_subdirectory(../llm_engine_cpp)` + `llmengine::llm_engine` link — the hardest split cost)
- `/home/ai/ai-projects/llm/.gitattributes` (the already-done LFS figure rule)
- `/home/ai/ai-projects/llm/testing/gui_cpp/docs/HFPROXY_PLAN.md` (the in-flight gui_cpp↔backend transition)
- `/home/ai/ai-projects/llm/CLAUDE.md` (worktree-per-session discipline + pre-convention split table = the working-discipline evidence)