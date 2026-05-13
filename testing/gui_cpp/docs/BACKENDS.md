# BACKENDS — the multi-backend roadmap for `llm_engine`

`llm_engine` is intentionally pluggable.  The abstract `llmengine::Model`
interface (`include/llm_engine/model.hpp`) is the only contract every
host (gui_cpp, future CLIs, future tests) talks through; the
implementation behind it is selected at runtime by the
`LLOB_BACKEND` env var.

The first concrete backend, [`HFProxyEngine`](HFPROXY_PLAN.md), proxies
the existing FastAPI server — the lowest-engineering-cost option, picked
to validate the abstraction quickly.  It is **not** the long-term answer.
Several native backends would be cleaner for the workspaces we actually
care about — no Python interpreter, no HTTP boundary, no JSON
serialization, no GIL.  This document maps that future.

---

## 1. Backend matrix

| Backend | Path / target | Best for | Native? | Status |
|---|---|---|---|---|
| `mock`    | `MockModel`              | dev, screenshots, tests   | n/a    | shipped |
| `hf`      | `HFProxyEngine`          | bridge to existing FastAPI server (validates abstraction) | no (Python proxy) | shipped (Phase 1–2C) |
| `gguf`    | `GGUFInspectorEngine`    | weight inspection (raw_tensors, architecture, param breakdown, weight histograms, tensor stats) — NO inference | yes (file reader) | **skeleton** |
| `llama`   | `LlamaCppEngine`         | inference + activation capture (attention, residual, logit lens, qkv stats, head stats) — wraps `lib/llama.cpp` C API | yes (embedded) | **skeleton** |
| `torch`   | `LibtorchEngine`         | training / finetune workspaces, custom kernels, in-process forward passes with full tensor surgery | yes (libtorch) | **skeleton** |
| `native`  | `NativeRuntimeEngine`    | pure-C++ inference + custom kernels (the long-term ambition — own everything from disk to logits) | yes (own runtime) | **skeleton** |

Every "skeleton" entry compiles and is selectable via `LLOB_BACKEND=…`,
but the per-method bodies are stubs that log "not yet implemented" and
fall through to MockModel sentinel returns.  Implementation lands
incrementally per backend; the contract is locked from day one.

---

## 2. Workspace ↔ best-fit backend

Most workspaces benefit from a *combination* of backends rather than one
master implementation.  An obvious near-term goal is composing them
(e.g. GGUF for weight panels + LlamaCpp for inference panels in the
same session) — see §6 below.

| Workspace | Hooks needed | Cleanest backend | Acceptable fallback |
|---|---|---|---|
| Architecture map        | getModelInfo, getParamBreakdown, getLiveActivations | LlamaCpp (info+live) + GGUF (param breakdown) | HFProxy |
| Inference               | getResidualSummary, getLogitLensTrajectory, getOutputLogits, getCurrentTokens | LlamaCpp w/ cb_eval | HFProxy /generate WS |
| Attention               | getAttentionPattern, getQKVStats, getHeadStats | LlamaCpp w/ cb_eval | HFProxy /generate WS |
| Probes / SAE            | getFeatureLibrary, getFeatureCard, getFeatureExamples, getProbeTrainState | Libtorch (probe training) + LlamaCpp (capture) | HFProxy + Python probe trainer |
| Training (full-model)   | getTrainingState, getTrainingMetrics, getTrainingLoss, mutators | Libtorch (only practical native option) | HFProxy + Python trainer |
| Finetune (LoRA)         | getLoRAConfig, getOptimizerConfig, getEvalDiff, getABSample | Libtorch                                       | HFProxy + Python LoRA |
| Datasets                | getDatasets, getSample, getSampleStats, getDatasetDistribution | GGUF doesn't apply — needs external dataset registry | HFProxy + new endpoints |
| Raw tensors             | getStateDict, getTensorMeta, getWeightSlice, getWeightHistogram, getTensorStats, getSingularValues, getTensorSlice2D, getDiffSlice2D | **GGUF (perfect fit)** | HFProxy + new endpoints |
| Logs                    | getEngineMetrics, drainEngineLogs               | any (engine-internal only) | any |

**Reading the table:** the *raw tensors* and *architecture* workspaces
get an enormous quality-of-life upgrade from `GGUFInspectorEngine` — no
network, no Python, just `mmap()` over the file.  The *attention* and
*inference* workspaces want `LlamaCppEngine` because that's where
real-time activation capture happens cheapest.  The *training* /
*finetune* workspaces will likely stay with HFProxy or move to
`LibtorchEngine` — there's no GGUF or llama.cpp path for SGD.

---

## 3. Per-backend scope + design notes

### 3.1 `MockModel` (shipped)

Returns deterministic fake data when `LLOB_USE_MOCK_DATA=ON`, sentinels
otherwise.  Every other backend inherits from MockModel so unwired
methods fall through to its defaults — "this backend doesn't implement
that hook" never crashes the UI; the panel renders `—`.

### 3.2 `HFProxyEngine` (shipped Phase 1–2C)

See [`HFPROXY_PLAN.md`](HFPROXY_PLAN.md) for the per-method status table.

**Strengths:**
- Reuses the React frontend's existing FastAPI surface — zero new
  backend engineering.
- Clean process separation; can colocate the engine on a GPU box and
  run the UI on a laptop.

**Limitations:**
- Per-call latency (~1ms HTTP + JSON parse).
- No HTTP endpoint for raw attention matrices or paged activations
  (those flow through `/generate` WebSocket — Phase 4 work).
- Brings a Python dependency the rest of `llm_engine` doesn't have.
- GIL contention on heavy forward passes.

### 3.3 `GGUFInspectorEngine` (skeleton)

Pure-C++ GGUF file-format reader.  No inference, no GPU, no Python.
Opens a `.gguf` checkpoint and serves every Model::* method that's
about *static* weight data:

- `getModelInfo` — parsed from GGUF header
  (`llama.context_length`, `llama.attention.head_count`, etc.)
- `getStateDict` — enumerate every tensor in the file
- `getTensorMeta` / `getTensorStats` — shape/dtype/device + computed
  norms (lazy — compute on first request, cache)
- `getWeightSlice` / `getWeightHistogram` — `mmap()` direct read
- `getTensorSlice2D` / `getDiffSlice2D` — windowed read, no copies
- `getSingularValues` — Eigen / Armadillo SVD on the slice (the only
  non-trivial dependency this backend needs)

Methods this backend explicitly leaves to MockModel sentinels: anything
that requires a live forward pass (attention, residual, activations,
logit lens), anything training-related, anything dataset-related.

**Implementation plan** (see `src/gguf_inspector_engine.cpp` skeleton):
1. Phase A — header parse + `getModelInfo`
2. Phase B — tensor enumeration → `getStateDict` / `getTensorMeta`
3. Phase C — mmap read path → `getWeightSlice` / `getWeightHistogram`
4. Phase D — stats cache → `getTensorStats` / `getSingularValues`

GGUF spec lives at `https://github.com/ggerganov/ggml/blob/master/docs/gguf.md`.
Vendor a parser or write our own — the format is simple (header magic +
metadata kv-store + tensor index + tensor data); ~500 LOC of C++
covers full read.

### 3.4 `LlamaCppEngine` (skeleton)

Embeds `lib/llama.cpp` (already vendored at the project root, built
with CUDA).  Wraps `llama_context*` and registers a `cb_eval` callback
that intercepts attention / residual tensors as the forward pass
streams through ggml.

Hooks this backend will own:
- `getAttentionPattern(layer, head, seqLen, bias)` — captured from
  the post-softmax attention tensor in cb_eval
- `getActivation(layer, kind, offset, n)` — captured from
  resid_pre / resid_post / mlp_post tensors
- `getQKVStats` / `getHeadStats` — derived from cached Q/K/V tensors
- `getResidualSummary` / `getResidualContribution` — derived from
  cached resid tensors
- `getLogitLensTrajectory` — `lm_head @ resid_post[L]` per layer
- `getOutputLogits` — top-k from final logits
- `getCurrentTokens` — `llama_token_to_piece` over the active sequence

**Why this is the biggest pillar:** all the panels that "feel alive"
(attention heatmaps, residual flow, logit-lens trajectory) get their
data here.  No HTTP, no Python, no GIL.

**Risks:**
- `cb_eval` API stability — the tensor-name conventions used to filter
  for "this is the post-softmax attention" change across llama.cpp
  versions.  Pin a known-good version in `lib/llama.cpp` or write
  a version-detection layer.
- GPU buffer lifetime — captured tensors live in CUDA memory; we copy
  to host before storing in the per-frame cache.
- Threading — llama.cpp's `llama_decode` is not reentrant; protect
  with the same `Impl::mu` we already use for HFProxyEngine.

**Implementation plan**:
1. Phase A — link llama.cpp; bring up `llama_init_from_file` →
   `getModelInfo` from `llama_model_n_params` etc.
2. Phase B — `cb_eval` callback that snapshots one named tensor;
   wire to `getAttentionPattern`
3. Phase C — extend snapshot set to all attention/residual tensors;
   wire each Model::* hook
4. Phase D — generation loop on the worker thread → streams tokens
   into `getCurrentTokens` cache + per-step activations

### 3.5 `LibtorchEngine` (skeleton)

C++ `torch::` API in-process.  The most flexible but heaviest backend
— libtorch links a substantial portion of PyTorch (~hundreds of MB of
.so files).

Hooks this backend will own:
- Anything training-related (`pauseTraining`, `getTrainingMetrics`,
  `getGradFlowPerLayer`, etc.)
- LoRA / finetune (`getLoRAConfig`, `getEvalDiff`, ...)
- Anything that needs a custom forward pass with Python-scripted
  surgery (steering vectors, ablation hooks, probe heads)

**Why it's deferred:** libtorch has no built-in transformer architecture
— we'd be reimplementing the model definition in C++.  Practical only
when there's a specific kernel or training hook we want that neither
HFProxy nor LlamaCpp can deliver.

**Stub status:** the skeleton class exists and registers, but every
method falls through to MockModel.  Concrete implementation is pinned
behind a real workspace need.

### 3.6 `NativeRuntimeEngine` (skeleton)

The long-term ambition: a pure-C++ inference runtime owned by us — read
the checkpoint format directly, allocate tensors with our own arena,
implement attention / FFN / sampling kernels (CUDA + CPU), capture
everything we want at every step.

This is the "from scratch" path the user asked us to leave room for.
It's a multi-month effort but the abstraction is ready: implement
`NativeRuntimeEngine : Model` and the rest of the system links against
it like any other backend.

**Useful reference points** (none required, all permissively licensed):
- llama.cpp — for ggml tensor/quantization primitives
- karpathy/llama2.c — for a minimal reference implementation
- mlx-c — for a clean C-API design over a Metal/CUDA backend

The skeleton file documents this scope and otherwise inherits MockModel.

---

## 4. CMake build options

Each backend gates on a CMake option so users build only what they need:

```cmake
option(LLM_ENGINE_BUILD_HF_PROXY        "Build HFProxyEngine (FastAPI proxy)"      ON)
option(LLM_ENGINE_BUILD_GGUF_INSPECTOR  "Build GGUFInspectorEngine (file reader)"  ON)
option(LLM_ENGINE_BUILD_LLAMA_CPP       "Build LlamaCppEngine (embedded llama.cpp)" OFF)
option(LLM_ENGINE_BUILD_LIBTORCH        "Build LibtorchEngine (in-process torch)"  OFF)
option(LLM_ENGINE_BUILD_NATIVE_RUNTIME  "Build NativeRuntimeEngine (placeholder)"  OFF)
```

**Defaults** are tuned for "lowest-friction full feature exploration":
HF proxy + GGUF inspector are header-only-ish (cpp-httplib + nlohmann/
json + future GGUF reader code) and ship in the standard build.  The
heavier backends (llama.cpp, libtorch, native runtime) require external
libraries and are off by default — turn them on per-host.

The library defines `LLM_ENGINE_HAVE_<BACKEND>=1` for each enabled
backend so `main.cpp`'s factory can `#ifdef`-gate the constructor calls
and emit a sensible "backend not compiled in" error otherwise.

---

## 5. Choosing a backend at runtime

```bash
LLOB_BACKEND=mock   ./build/llobotomy             # default
LLOB_BACKEND=hf     LLOB_BACKEND_URL=http://gpu-box:8000 ./build-real/llobotomy
LLOB_BACKEND=gguf   LLOB_BACKEND_PATH=/models/llama-3.gguf ./build-real/llobotomy
LLOB_BACKEND=llama  LLOB_BACKEND_PATH=/models/llama-3.gguf ./build-real/llobotomy
LLOB_BACKEND=torch  LLOB_BACKEND_PATH=/models/llama-3 ./build-real/llobotomy
LLOB_BACKEND=native LLOB_BACKEND_PATH=/models/llama-3.bin ./build-real/llobotomy
```

`LLOB_BACKEND_URL` and `LLOB_BACKEND_PATH` are the two transport-style
configs.  Per-backend env vars (`LLOB_LLAMA_CTX_SIZE`,
`LLOB_LIBTORCH_DEVICE`, ...) live alongside as needed.

---

## 6. Composition (future)

A real interpretability bench likely wants `GGUFInspectorEngine` for
weight panels AND `LlamaCppEngine` for live activations against the
*same* model file.  Two ways to compose:

- **`CompositeEngine`** that fans each `Model::*` call to a configured
  child based on the method name (delegate weight reads to GGUF,
  delegate activations to LlamaCpp).  Cleanest for users; requires a
  bit of routing logic per hook.
- **Layer the backends** — `LlamaCppEngine` internally instantiates
  `GGUFInspectorEngine` for its weight reads since it's already loaded
  the GGUF file.  Hides the composition behind one Model facade at the
  cost of LlamaCppEngine knowing about GGUFInspectorEngine.

Both are deferred until at least two backends have non-stub
implementations.  The skeletons leave the door open for either.

---

## 7. Adding a new backend (developer guide)

1. Add header `include/llm_engine/<name>_engine.hpp` declaring
   `class <Name>Engine : public MockModel`.
2. Add source `src/<name>_engine.cpp` with constructor + dtor +
   override stubs.  Default every override to log "not yet implemented"
   and call the MockModel base.
3. Add a CMake option `LLM_ENGINE_BUILD_<NAME>` defaulted appropriately.
   When ON: append the source to `add_library(llm_engine ...)` and
   define `LLM_ENGINE_HAVE_<NAME>=1`.  Add any external dependency
   linking inside that branch.
4. Extend `gui_cpp/src/main.cpp`'s `MakeBackend()` factory with an
   `else if (choice == "<name>")` branch, gated by
   `#ifdef LLM_ENGINE_HAVE_<NAME>`.  Emit a clear error log if the user
   selects a backend not compiled in.
5. Update this doc's matrix and per-backend section.
6. Implement methods incrementally — every commit can land one or two
   real overrides; the rest stay as MockModel-inherited sentinels.

---

## 8. What we are NOT doing

- We are not promising backend feature parity.  Each backend covers
  the slice of `Model::*` it's good at; the gaps render `—` in the UI.
- We are not building a benchmark / evaluation framework inside the
  engine.  That's host territory (or a separate library).
- We are not trying to replace llama.cpp / libtorch.  When we want
  what they do, we link them.  `NativeRuntimeEngine` is a deliberate
  long-shot, not a substitute.
