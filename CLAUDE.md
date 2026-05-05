# Purpose
This is a workspace for robust experimental LLM research using open source models that can be tested and modified locally or fine-tuned.

## Project Structure

`theory/` — LLM theoretical-framework workspace (**GROUND TRUTH**) for theory, math, historical progression, and visualizations. KB-substrate layout (v2 from 2026-05-03):
  - `kb/notes/<area>/<topic>.md` — digested synthesis, one file per topic
  - `kb/excerpts/<paper-key>.md` — verbatim quoted passages from primary sources
  - `kb/index/` — `papers.json` (paper metadata + KB cross-refs), `topics.md` (topic graph + status), `timeline.md` (chronological progression)
  - `kb/glossary.md` — every technical term used in this workspace, with a citation. If you use a term not in the glossary, add it immediately.
  - `sources/papers/` — PDFs of canonical papers (filename: `{paper-key}_{slug}.pdf`)
  - `sources/forums/` — selectively archived blog/forum snapshots (discovery-path provenance only)
  - `archive/2026-05-03-pre-expansion/` — v1 single-LaTeX-doc state preserved as a snapshot
  - `series/` — placeholder for future LaTeX paper series (outlined after KB pass complete; see `theory/docs/superpowers/specs/2026-05-03-theory-expansion-design.md`)
  - `plans/` — research/KB construction running plans
`testing/` — experimental research on local LLMs; uses llama.cpp and ollama for inference
  - `llm_surgeon/` — Python toolkit for layer-level model surgery
  - `tests/` — pytest test suite for llm_surgeon
  - `experiments/` — experiment definitions and database
  - `prompts/` — prompt templates for experiments
  - `gui/` — FastAPI + React/TS GUI for live inspection & intervention
    - `backend/` — FastAPI app (routes, sessions, WebSockets)
    - `frontend/` — Vite + React + Zustand; IndexedDB persistence
    - `frontend/tests/e2e/` — Playwright smoke suite (no backend required)
  - `.cache/models/` — cached HuggingFace models (TinyLlama, OpenLLaMA 3B)
  - `.cache/outputs/` — experiment output directories (modified model variants, baselines)
`lib/` — External libraries
  - `llama.cpp` — llama.cpp source (built with GPU enabled)
`research/` — Research scripts, experimental findings, anomalies, hypotheses
  - `observations/` — see `# Research Observations` below for format


# Build Commands
```bash
# Compile LaTeX document (automatically saves dated copy in archive)
make -C theory all

# View HTML visualization
xdg-open theory/visuals/llm-architecture-diagram.html
```

# Theory KB & citation discipline

The theory workspace at `theory/` is now a knowledge-base substrate (v2,
2026-05-03), not a single LaTeX doc. The previous v1 single-doc deliverable
is archived at `theory/archive/2026-05-03-pre-expansion/`. The future
LaTeX paper series under `theory/series/` is **deferred** until the KB
reaches stable coverage — see
`theory/docs/superpowers/specs/2026-05-03-theory-expansion-design.md`.

## Citation rules — non-negotiable

When making technical claims about LLM architecture, training, inference,
interpretability, evaluation, alignment, or related theory:

1. **Every load-bearing claim cites a source.** Either:
   - `[paper-key §X, eq.Y]` — pointing to a paper in `theory/kb/index/papers.json`
   - `[kb/notes/<area>/<file>#<anchor>]` — pointing into a synthesis note
   - `[kb/excerpts/<paper-key>#<heading>]` — pointing into a verbatim excerpt

2. **Verify against the original PDF before propagating a KB-note claim**
   into LaTeX, code, or commit messages. The KB is digested; the paper is
   canonical.

3. **Analogies and intuitions are tagged, never asserted as fact.** Use
   `[ANALOGY]`, `[INTUITION]`, `[CONTRADICTION]`, `[FORUM-SIGNAL]`, or
   `[SPECULATION]` so they cannot be laundered as formal claims. Analogies
   must always return to the canonical symbolic form.

4. **If a claim depends on something not in the KB, add it before continuing.**
   Don't make claims you can't ground.

5. **Forum/blog citations are valid as discovery signals only** (tier B/C
   in `theory/sources/README.md`). They never solely back a hard technical
   claim — only primary papers (tier A) can.

## Source tiers

- **Tier A (canonical):** arxiv, peer-reviewed venues, official tech reports
  / model cards, reference github repos. Stored under
  `theory/sources/papers/`. Backs hard claims.
- **Tier B (high-signal commentary):** vendor research blogs, respected lab
  blogs, named researchers' writeups. Cite alongside an underlying tier-A
  source.
- **Tier C (community signal):** Reddit/HN/X/HF community. Discovery only;
  never the sole citation.

## Writing-style rule (Feynman bar)

Each topic note follows: formal definition (math + variables defined
underneath) → mechanism (how it computes, with tensor shapes) →
variants/lineage (cited list) → tagged `[INTUITION]` / `[ANALOGY]` (which
always return to canonical symbolic form) → frontier and open questions
(with `[CONTRADICTION]` markers where sources disagree).

When introducing a new technical term in a note, add it to
`theory/kb/glossary.md` with a citation.


# TESTING

## Python toolkit (llm_surgeon)
```bash
# Run the pytest suite
testing/.venv/bin/python -m pytest testing/tests/ -v
```
- Venv: `testing/.venv/` — system python does NOT have torch/pytest
- `llm_surgeon` installed editable: `pip install -e .`

### Primary dev models
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (22 layers, 2048 hidden, 1.1B params) — **default** for examples, fixtures, and fast test iteration.
- `openlm-research/open_llama_3b_v2` (26 layers, 3200 hidden, 3B params) — for slightly-larger-scale sanity checks.
- Models live under `testing/.cache/models/` in the HuggingFace `models--{org}--{name}` layout.

## GUI frontend — verification tiers

Three progressively-stronger checks for frontend changes. Run from
`testing/gui/frontend/`. Each catches a strictly larger class of bug than
the previous tier — run what the change's risk profile warrants, not all
three every time.

```bash
# Tier 1 — Type check (~1s). Catches prop/type regressions.
./node_modules/.bin/tsc --noEmit

# Tier 2 — Production build (~2s). Adds bundler-level checks: module
# resolution, JSX transform, CSS imports.
./node_modules/.bin/vite build

# Tier 3 — Playwright smoke suite (~7s headless).
npm run e2e         # headless, 9 tests
npm run e2e:ui      # watch mode with inspector
```

**Always run Tier 3 after UI or store changes** — the current suite catches
the most common React-crash-on-mount class (a real one was caught: store
fetches assuming response-JSON was an array; see commit 5524e39).

### Smoke-suite structure

Suite under `testing/gui/frontend/tests/e2e/`: `smoke.spec.ts` (9 tests), `fixtures/sample.json` (imported by tests), `tsconfig.json` (scopes `@types/node` to tests only). `playwright.config.ts` at frontend root auto-boots vite via its `webServer` block.

Config notes:
- `reuseExistingServer: true` — piggybacks on `npm run dev` if already open.
- No backend required: tests seed state via the app's own experiment-import
  (`<input type="file">` → `setInputFiles`). This exercises real store +
  IndexedDB paths, not mocks.
- Node-16 constraint: `@playwright/test` pinned to `~1.41.2`, last release
  supporting Node 16. Bump freely once Node ≥18 is installed.
- Both `vite` and `playwright test` spawn subprocesses that touch
  `/dev/urandom` via node's crypto — run them with
  `dangerouslyDisableSandbox: true` when invoking via the Bash tool.

### Adding a new test

1. Seed any needed state via `setInputFiles` on the hidden `input[type="file"]`
   using (or extending) `tests/e2e/fixtures/sample.json`. The fixture's shape
   is `ExperimentFile` from `utils/experiment.ts`.
2. For flows that show `window.prompt` / `window.confirm`, call
   `page.once("dialog", (d) => d.accept(answer))` **before** the triggering
   action — dialogs aren't retroactive.
3. Prefer `getByRole("button", { name: "..." })` over class/style selectors.
   For accessible names that share a substring (e.g. a tag chip and a tab
   button containing `#tag`), pass `exact: true`.
4. Use the shared `isBackendlessNoise()` filter when asserting on console
   errors — 503s from the vite proxy and ERR_ABORTED cancellations are
   expected when running without a live backend.
5. Put new tests in the same `smoke.spec.ts` for now; split by domain once
   the file exceeds ~400 lines.

### When smoke tests aren't enough

The suite intentionally does **not** cover:
- WebSocket streaming behavior (needs the real backend or a mock WS server)
- File-download byte correctness (click is tested; contents aren't)
- Visual layout regressions (add `toHaveScreenshot()` if you want this)

For WebSocket-dependent flows, boot the full stack with `./gui/run.sh` and
test by hand until a mock-WS server is added.


# Type Checking

Zero errors, warnings, AND informations after every edit — for both pyright (Python) and tsc (TypeScript). **The edit-time linter messages (the `<new-diagnostics>` system reminder attached to each edit response) are authoritative — trust them and do NOT run `pyright` via Bash to re-verify.** `pyproject.toml [tool.pyright]` is configured so the linter messages line up with what we want fixed; running pyright separately just costs time.

## Tools (only when explicitly requested or for tsc)

```bash
# TypeScript still requires a manual run (no LSP-equivalent edit hook)
cd testing/gui/frontend && ./node_modules/.bin/tsc --noEmit

# Pyright manual run — only if the user asks, or you suspect a cache mismatch:
.venv/bin/python -m pyright <paths>
```

## Fix patterns

- **Real typing bug** → fix the source. Prefer upstream type tightening so narrowing cascades (e.g., `SessionInfo.model: object` → `Any`; `info.llama: object` → `Optional[LlamaEngine]`).
- **Ad-hoc dynamic class** (`class _Meta: pass; cfg = _Meta(); cfg.x = ...`) → replace with `types.SimpleNamespace(x=..., ...)` — stdlib, typed as dynamic.
- **Test helper for `Optional[X]` guarded by `@pytest.mark.skipif`** → extract a helper like `_tinyllama_blob() -> Path` that asserts and returns the narrowed type.
- **Never** disable rules in `pyproject.toml` to quiet diagnostics — it hides real bugs elsewhere.

### Type-narrowing tier list — reach down the list, not up

Listed from "expresses intent best" to "suppresses signal":

1. **`assert isinstance(x, T)`** — narrows for both runtime AND pyright. Best when you also want a runtime safety check.
2. **`cast(T, expr)`** — narrows the *result* type with no runtime cost. Best when you know the expression's value is type T but pyright sees a wider Union.
3. **`# pyright: ignore[reportXxx]`** — last resort, suppresses a specific rule on one line. Reserve for two cases where there's no static fix:
   - **Mid-chain attribute access** on dynamic objects (e.g., `model.model.layers[L].mlp.down_proj.weight` on pytorch — `mlp` is in `Tensor | Module | Unknown`, no `cast` of the result reaches the `.down_proj` step).
   - **Stub lag** where the runtime API has a kwarg/method the stub doesn't enumerate (e.g., `Metaspace(prepend_scheme=...)` in `tokenizers`, `LlamaConfig(**kwargs)` in HuggingFace — no `cast` of the result reaches the call site).
4. **Bare `# type: ignore`** — never. Always use the rule-scoped `# pyright: ignore[X]` form.

Default to (1) or (2). If you find yourself reaching for (3), check whether the issue is mid-chain or stub-lag — those are the only honest cases. **The base config has `reportUnnecessaryTypeIgnoreComment` ON**, so the moment a stub catches up and an ignore stops doing real work, pyright emits a diagnostic asking us to delete it. Treat every ignore as a temporary debt with a built-in expiry alarm — not a permanent suppression.

## Known stub lag in this repo

- Fully unstubbed packages: `llama_cpp`, `gguf`, `bitsandbytes`.
- Torch stubs lag runtime for: `torch.OutOfMemoryError`, `with torch.device(...)`, `load_state_dict(assign=...)`.
- `reportPrivateImportUsage` is muted in `~/.config/pyright/base.json` because torch's `__init__.pyi` doesn't re-export the bulk of its runtime surface (`torch.float32`, `torch.zeros`, `torch.tensor`, ...). With the rule on, every torch use fires a false positive with no fix in user code.

## Unused symbols — underscore-prefix behavior

**Honored** (rename to `_name` suppresses the warning):
- Local assignments, tuple unpacking (`_first, second, _ = triple`), `for _idx, val in enumerate(...)`, function parameters.

**`reportUnusedFunction` and `reportUnusedClass` are muted in the global base config** — research code legitimately has scratch helpers, and the escape hatches (`# pyright: ignore`, adding to `__all__` solely to silence) are anti-patterns. Use grep / IDE for real dead-code sweeps.

**`reportUnusedImport` is honored** (rename to `_name` does NOT suppress — delete the import, or add to `__all__` for `__init__.py` re-exports).

IDE `★` dead-code hints flag *every* unused name regardless of prefix — that's a separate always-on channel, not tied to the `reportUnusedXxx` rules. When `<new-diagnostics>` shows only `★` items, verify against the CLI before editing — those are frequently IDE-only noise.

## TypeScript language-service hints

`tsc --noEmit` doesn't surface these; the language server does. They're real actionable refactors, not noise:
- `80006` — function may be converted to async (use it)
- `80001` — CommonJS import can be ES6
- `7044` — parameter type could be inferred

# C++ GUI frontend (`testing/gui_cpp/`)

Native C++/Dear ImGui frontend, scaffolded in `testing/gui_cpp/`. Currently a skeleton — will absorb panels incrementally as the React frontend at `testing/gui/frontend/` is replaced.

**Framework conventions live in the imgui-toolkit repo at `/home/ai/ai-projects/imgui-toolkit/skill/SKILL.md`.** The toolkit (sibling project) provides both the foundation library (`ImTool::` namespace — `Application`, `CommandQueue`, `ScopedX`, `Setting<T>`, `KeyBindingManager`, custom widgets, viz atoms) and the host-side conventions skill that describes how to use it. When writing or planning C++/ImGui code in `testing/gui_cpp/`, read that skill first.

The skill is V0 and iterates alongside the toolkit's API. The toolkit also ships its own `docs/CONTRIBUTING.md` (extender disciplines for the toolkit itself) and `docs/DESIGN.md` (full API surface). For host-side work in this project (writing panels, commands, project-specific viz on top of the toolkit), the skill is authoritative. For toolkit-internal work (extending the toolkit), CONTRIBUTING.md is authoritative — but that's done over in the toolkit repo, not here.

## Project-specific layer (this section is the llm-surgeon GUI plan)

**Scope policy**: skill = framework, this section = LLM-specific application of it.

**Cross-reference**: the React frontend at `testing/gui/frontend/` is the feature inventory. Audit it for required panels (sessions list, layer view, logit lens, intervention editor, sample browser, capture history, etc.), not for design — the C++ side is the chance to redesign each panel with custom DrawList components.

**Backend abstraction** (the skill's optional `Backend` interface, applied):
- `HttpBackend` — talks to the existing FastAPI service at `testing/gui/backend/`. Default during transition.
- `NativeBackend` — future, in-process integration with llama.cpp or libtorch. Skip until needed.
- Panels never touch HTTP/WS directly. CLI flag selects backend at startup.
- IPC format starts as JSON-over-WS (parity with current React frontend); consider MsgPack or Arrow IPC if streaming tensors becomes a bottleneck.

**Domain-specific viz categories** to design as `inc/ui/viz/` components:
- **Tensor heatmaps** — residual stream over layers (rows = positions, cols = layers), attention patterns (rows = query positions, cols = key positions), weight matrices
- **Time-series** — residual stream evolution over generation steps, attention scores over positions, top-k logit evolution
- **Distributions** — logit distributions, attention entropy histograms, activation magnitude per layer
- **Token visualization** — tokens with hover-to-show-logits, click-to-pin, drag-to-edit; per-token attention/probe overlays
- **Graph viz** — if going node-based for experiment chains: model-load → prompt → capture → intervene → eval

**Node-graph option**: the skill flags astrolograph's `fluidsV1` node-graph engine as a strong fit for "experiment as graph." Direct conceptual parallel: `staticFieldNode` ≈ "loaded model layer," `fftNode` ≈ "spectral analysis of activations," etc. The execution engine, dependency tracking, and worker thread already exist; new node types are mostly what's needed. Consider after the basic panels are landed.

**License**: llm_surgeon, astrolograph, and algo-syndesi are GPLv3; graphene-v2 and logos are zlib-style permissive. The user owns all sources. All ports into this project are license-clean.

# Research Observations

Record interesting experimental findings in `research/observations/`.
Each observation file should include:
- **Date and context** — what experiment was running, what model, what parameters
- **Finding** — what was observed, why it's interesting or unexpected
- **Evidence** — relevant output/transcript excerpts
- **Reproducibility** — exact commands/code to reproduce the result
- **Hypotheses** — possible explanations
- **Follow-ups** — what experiments could test the hypotheses
- **References** — related published work if known
Format: `YYYY-MM-DD-<descriptive-slug>.md`. No index file needed — scan by filename.
