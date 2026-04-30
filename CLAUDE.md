# Purpose
This is a workspace for robust experimental LLM research using open source models that can be tested and modified locally or fine-tuned.

## Project Structure

`theory/` — LLM theoretical framework (**GROUND TRUTH**): LLM architectures and math, historical progression/timeline, high-level explanations, and visualizations.
  - `build/llm-core-architecture/` — LaTeX source and build artifacts
  - `sources/` — WebSearch-based publication research for the theoretical/mathematical basis
  - `visuals/` — visual materials
  - `GLOSSARY.md` — every technical term used in this workspace must be defined here. If you encounter or use a term not in the glossary, add it immediately.
  - `llm-core-architecture.pdf` — compiled output (kept at theory root)
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

# Theory Document

## Core Document
`theory/build/llm-core-architecture/llm-core-architecture.tex` — LaTeX document covering Transformer architecture from original encoder-decoder through modern decoder-only (LLaMA). 10 sections: Transformer overview, Tokenization, Embeddings, Positional Encoding, Attention, FFN, Normalization/Residuals, Decoder-Only Shift, Output Head, Full Forward Pass.

### Custom LaTeX environments:
- `\begin{implbox}` — green "Implementation Note" callouts
- `\begin{evobox}` — blue "Architectural Evolution" callouts
- `\dimtext{}` — inline dimension annotations

## HTML Companion
`theory/visuals/llm-architecture-diagram.html` — standalone interactive diagram with clickable layers showing tensor shapes and data flow. Dark theme, self-contained (no build step).

## Sources

All claims must be grounded in canonical papers.
- `theory/sources/index.json` — master index (citation key, title, authors, year, URL, local file, summary)
- `theory/sources/papers/` — local PDF copies, named `{key}_{slug}.pdf`
When adding a new source: add entry to `theory/sources/index.json`, download PDF to `theory/sources/papers/`, use the citation key consistently in LaTeX `\cite{}` commands

## Conventions/Rules
- Define every variable in every equation, directly underneath it (brief is fine, but no undefined variables)
- First formalize math, then describe technical aspects and practical use, then elaborate using accessible language and/or analogies
- Ground all architectural claims in specific source papers from core document or original paper(s), with citation keys from `theory/sources/index.json`


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
3. **`# pyright: ignore[reportXxx]`** — suppresses a specific rule on one line. Reserve for two cases where there's no static fix:
   - **Mid-chain attribute access** on dynamic objects (e.g., `model.model.layers[L].mlp.down_proj.weight` on pytorch — `mlp` is in `Tensor | Module | Unknown`, no `cast` of the result reaches the `.down_proj` step).
   - **Stub lag** where the runtime API has a kwarg/method the stub doesn't enumerate (e.g., `Metaspace(prepend_scheme=...)` in `tokenizers`, `LlamaConfig(**kwargs)` in HuggingFace — no `cast` of the result reaches the call site).
4. **Bare `# type: ignore`** — never. Always use the rule-scoped `# pyright: ignore[X]` form so future stub upgrades surface as actionable diagnostics rather than silently leaving the suppression in place.

Default to (1) or (2). If you find yourself reaching for (3), check whether the issue is mid-chain or stub-lag — those are the only honest cases.

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
