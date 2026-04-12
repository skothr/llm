# Live Probe GUI — Design Spec

## Overview

A browser-based live experimentation dashboard for interactive model probing, intervention, and visualization. Built on the existing `llm_surgeon` toolkit, it provides real-time feedback as you explore model internals.

**Primary use case:** Load a model into GPU memory, run probes and interventions interactively, see results update in real time. Optionally load a second model (or clone + surgery variant) for side-by-side A/B comparison.

**Stack:** FastAPI (Python backend) + React/Vite/TypeScript (frontend) + D3.js (visualizations)

**Deployment:** Local only. Backend binds to `127.0.0.1:8000` (not `0.0.0.0`). CORS origin locked to `http://localhost:5173` (dev) and `http://127.0.0.1:8000` (production). Single-user, same machine (RTX 2080, 8GB VRAM).

---

## Architecture

```
┌─────────────────────────────────────┐
│           Browser (React)           │
│  ┌───────────┐  ┌────────────────┐  │
│  │  Session   │  │ Visualization  │  │
│  │  Controls  │  │    Panels      │  │
│  └─────┬─────┘  └───────▲────────┘  │
│        │                │           │
│        ▼                │           │
│  ┌─────────────────────────────┐    │
│  │     WebSocket + REST API    │    │
│  └─────────────┬───────────────┘    │
└────────────────┼────────────────────┘
                 │
┌────────────────┼────────────────────┐
│  FastAPI       │        (Python)    │
│  ┌─────────────▼───────────────┐    │
│  │      Session Manager        │    │
│  │  (models in GPU memory)     │    │
│  └─────────────┬───────────────┘    │
│        ┌───────┼───────┐            │
│        ▼       ▼       ▼            │
│    ┌──────┐┌───────┐┌────────┐      │
│    │probe ││surgery││inspect │ ...  │
│    └──────┘└───────┘└────────┘      │
│        llm_surgeon toolkit          │
└─────────────────────────────────────┘
```

Two communication channels:
- **REST** — request/response operations: load model, get layer info, list sessions, apply surgery
- **WebSocket** — streaming operations: logit lens (layer-by-layer), live generation (token-by-token), intervention results. Maps to the existing `on_layer` callback pattern.

**Session Manager** holds loaded models in GPU memory. Each session has a user-chosen name (validated as a slug: lowercase alphanumeric + hyphens, max 64 chars) which serves as its ID. Multiple sessions can be active for A/B comparison.

No changes to existing `llm_surgeon` code. The backend imports and calls the toolkit as-is. `on_layer` callbacks are wrapped to push WebSocket messages.

---

## Backend API

### Session Endpoints (REST)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sessions` | Load model by HF ID or local path. Body includes `name` (slug) and `model_id`. Returns session metadata. |
| `GET` | `/sessions` | List active sessions with metadata (name, model_id, layer count, estimated VRAM MB). |
| `GET` | `/sessions/{name}/info` | Layer info, head count, hidden size. Wraps `get_layer_info`. |
| `DELETE` | `/sessions/{name}` | Unload model, free GPU memory via `del model` + `torch.cuda.empty_cache()`. |
| `POST` | `/sessions/{name}/surgery` | Apply surgery operation. Auto-snapshots `state_dict()` before applying. Returns updated layer info + surgery log. |
| `POST` | `/sessions/{name}/surgery/undo` | Restore to the most recent pre-surgery snapshot. Only one level of undo. |
| `POST` | `/sessions/{name}/clone` | Clone session to a new name. See Clone Strategy below. |

All request/response bodies use Pydantic models for validation.

### Probe Endpoints (WebSocket)

| Path | Input | Streaming Output |
|------|-------|-----------------|
| `ws://.../sessions/{name}/logit-lens` | `{prompt, top_k}` | Layer-by-layer predictions via `on_layer`. Final message has full result. |
| `ws://.../sessions/{name}/generate` | See Generation Parameters below | Token-by-token with top-k probabilities per step. |
| `ws://.../sessions/{name}/intervene` | `{prompt, interventions, capture_logit_lens}` | Streamed intervention results with optional logit lens capture. |

### Inspect Endpoints (REST)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sessions/{name}/inspect/influence` | Block influence scores per layer. Body: `{prompts: [...]}`. Response includes progress via streaming JSON lines (one per layer). |
| `POST` | `/sessions/{name}/inspect/attention` | Attention entropy per head per layer. Body: `{prompt}`. |
| `POST` | `/sessions/{name}/inspect/residual-norms` | Residual stream L2 norms through network. Body: `{prompt}`. |

### VRAM Management

The Session Manager tracks estimated VRAM usage per session using `torch.cuda.memory_allocated()`. 

**Clone pre-check:** Before cloning, the manager calls `torch.cuda.mem_get_info()` and compares free memory against the source session's estimated size with a 20% safety margin. If insufficient, the clone fails with HTTP 507 and the response body includes:
- `error`: "Insufficient VRAM"
- `free_mb`: current free VRAM
- `required_mb`: estimated requirement
- `suggestion`: "Unload session '{name}' to free ~{N} MB" (names the largest unloadable session)

CPU offload is out of scope for v1 — models are fully on GPU or not loaded.

### Clone Strategy

`POST /sessions/{name}/clone` does **not** use `copy.deepcopy()`. Instead:

1. Extract `state_dict()` from the source model
2. Instantiate a fresh model via `AutoModelForCausalLM.from_config(source.config)` 
3. Call `load_state_dict()` on the new instance
4. Load to GPU with same quantization config as source

This avoids issues with non-picklable attributes, custom hooks, and weight tying. Weight-tied models (embedding ↔ lm_head) get their ties re-established by `from_config()` + `load_state_dict()` since HuggingFace handles tie registration at init time.

### Surgery & Undo

Surgery mutates the session's model in place. Before each surgery operation, the backend snapshots `model.state_dict()` (kept in CPU memory). The intended workflow for A/B comparison:

1. Load model as "baseline"
2. Clone "baseline" → "variant"
3. Apply surgery to "variant"
4. Run probes on both

If a surgery goes wrong, two recovery paths:
- **Undo** — `POST /sessions/{name}/surgery/undo` restores the most recent snapshot. Only one level of undo (the snapshot is overwritten on each surgery call).
- **Delete + re-clone** — delete the variant, clone from baseline again.

The surgery log in the UI shows the history of operations but is informational only — it cannot replay operations.

### Intervention Serialization

Frontend sends intervention configs as JSON. The backend translates them to `Intervention` objects:

```json
{
  "prompt": "The capital of France is",
  "interventions": [
    {"layer": 5, "sublayer": "ffn", "op": "scale", "params": {"factor": 0.5}},
    {"layer": 10, "sublayer": "attn", "op": "zero_dims", "params": {"dims": [0, 1, 2]}},
    {"layer": 8, "sublayer": "ffn", "op": "noise", "params": {"std": 0.1}}
  ],
  "capture_logit_lens": true
}
```

Backend maps `op` strings to `probe.ops.*` functions: `scale`, `zero_dims`, `clamp`, `noise`. The `params` dict is unpacked as kwargs. Unknown ops return a validation error.

**Tensor-valued ops** (`replace`, `project_out`) use references to previously captured hidden states rather than raw tensor data:

```json
{"layer": 5, "sublayer": "ffn", "op": "replace", "params": {
  "source": {"session": "baseline", "layer": 5, "sublayer": "ffn", "prompt": "The capital of France is"}
}}
```

The backend resolves this by running `extract_hidden_states()` on the referenced session/prompt to get the tensor, then wraps it with `ops.replace()`. Same pattern for `project_out` — the `direction` param references a capture point, and the backend extracts the hidden state vector to use as the projection direction.

This keeps the frontend tensor-free. The cost is an extra forward pass to extract the reference tensor, but these are cached per (session, prompt) pair for the duration of the server process.

### Generation Parameters

The generate endpoint accepts:

```json
{
  "prompt": "Once upon a time",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 0.9,
  "repetition_penalty": 1.1,
  "stop_sequences": ["\n\n", "<|endoftext|>"],
  "prob_top_k": 10
}
```

- `max_tokens`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `stop_sequences` — passed to `model.generate()` via `GenerationConfig`
- `prob_top_k` — how many alternative tokens (with probabilities) to stream per step. Default 10. This is captured by the `LogitsProcessor`, not a generation parameter.

Intervened generation (applying interventions during autoregressive generation) is out of scope for v1. The intervention endpoint runs a single forward pass; the generate endpoint does standard generation.

### WebSocket Lifecycle

**Connection model:** One WebSocket connection per probe operation. The frontend opens a connection, sends a single config message, receives streamed results, and the server closes the connection after sending the `"complete"` message.

**Concurrent operations:** Each session has an asyncio lock. If a probe WebSocket opens while another is active on the same session, the new request waits for the lock (queued). The frontend disables the Run button while a probe is in progress and shows a cancel button instead.

**Cancellation:** The frontend can send `{"type": "cancel"}` on an active WebSocket. The backend sets a cancellation flag checked by the `on_layer` callback. If cancelled, the backend sends `{"type": "cancelled"}` and closes. For generation, `model.generate()` uses a `StoppingCriteria` that checks the flag.

**Keepalive:** The server sends WebSocket ping frames every 30 seconds. If no pong received within 10 seconds, the connection is considered dead and the backend cancels the operation.

**Disconnect mid-stream:** If the WebSocket disconnects unexpectedly, the backend catches the send error on the next callback invocation and aborts the operation. For `model.generate()`, the `StoppingCriteria` checks connection state.

### WebSocket Message Protocol

All WebSocket messages have a `type` field:
- `"data"` — incremental result (one layer, one token)
- `"error"` — exception from toolkit. Includes `message` string. Connection closes after error.
- `"complete"` — operation finished, includes summary/full result. Connection closes after.
- `"cancelled"` — operation was cancelled by client. Connection closes after.

### Error Handling

**Model loading failures:** If download/load fails partway, the session is cleaned up — no partial sessions remain in the manager. The error response includes the exception message.

**Surgery on invalid indices:** The backend validates layer/head indices against current model dimensions before calling the toolkit. Returns HTTP 422 with specific error (e.g., "Layer 25 does not exist — model has 22 layers").

**WebSocket errors:** Any unhandled exception during a probe operation sends an `"error"` message with the exception string, then the connection closes. The frontend displays the error in the visualization area. The session remains valid — the error doesn't corrupt the model state.

**Pydantic validation:** All REST request bodies use Pydantic models. FastAPI returns HTTP 422 with field-level error details for malformed requests.

---

## Frontend

### Layout

```
┌──────────────┬──────────────────────────────┐
│              │                              │
│   SESSION    │                              │
│   SIDEBAR    │     VISUALIZATION AREA       │
│              │                              │
│  - Models    │  (logit lens heatmap,        │
│  - Surgery   │   layer influence bars,      │
│  - Probes    │   A/B diff, generation)      │
│              │                              │
├──────────────┤                              │
│              │                              │
│   PROBE      │                              │
│   CONFIG     │                              │
│              │                              │
│  - Prompt    ├──────────────────────────────┤
│  - Params    │     GENERATION OUTPUT        │
│  - Run       │  (token stream + probs)      │
│              │                              │
└──────────────┴──────────────────────────────┘
```

### Components

**Session Sidebar (top-left):**
- Model loader — HF ID text input or local path dropdown
- Session name input (slug format)
- Active sessions list with estimated VRAM usage
- Clone button per session (with name input for the clone)
- Surgery panel — operation type selector, layer/head pickers, apply button
- Undo button (enabled when a snapshot exists)
- Surgery log history per session (informational)

**Probe Config (bottom-left):**
- Prompt text area
- Operation selector: logit lens, influence, intervene, generate
- Operation-specific parameter controls (top_k, intervention ops, temperature, max_tokens, etc.)
- Target session selector (single or "both" for A/B)
- Run button (disabled while probe is active)
- Cancel button (visible while probe is active)

**Visualization Area (top-right):**
- Tab or stack layout for multiple result panels
- Results persist until cleared; multiple probes can be compared
- Error messages displayed inline when probes fail
- Visualization types:
  - **Logit lens heatmap:** layers on Y-axis, token positions on X-axis, color = confidence. Hover shows top-k predictions per cell.
  - **Layer influence:** horizontal bar chart, one bar per layer showing block influence score.
  - **A/B diff:** two heatmaps side-by-side with optional diff overlay highlighting divergence points.

**Generation Output (bottom-right):**
- Token-by-token stream display
- Each token clickable to show top-k alternatives with probabilities
- Side-by-side layout when running A/B generation

### State Management

Zustand store with three slices:

- **Sessions** — loaded models, metadata, surgery history per session. Source of truth for GPU state.
- **Probes** — current config (prompt, operation, params, target sessions). Result history for scrollback. Active probe connection state.
- **Visualization** — which results are displayed, panel layout, A/B pairing.

### Streaming Data Flow (Logit Lens Example)

1. User types prompt, selects "logit lens", clicks Run
2. Frontend opens WebSocket to `/sessions/{name}/logit-lens`, sends config
3. Backend acquires session lock, calls `probe.logit_lens()` with `on_layer` callback
4. Each callback fires WebSocket message: `{type: "data", layer: 5, sublayer: "ffn", predictions: [...]}`
5. Frontend appends each row to heatmap incrementally — builds layer by layer
6. Final message: `{type: "complete", summary: {...}}`
7. Server closes WebSocket, releases session lock
8. Result pushed to probe history, visualization persists

### A/B Comparison Flow

1. Two sessions loaded (e.g. "baseline" and "no-layer-19")
2. User selects "Run on both"
3. Frontend opens two WebSocket connections in parallel (separate sessions, no lock contention)
4. Both heatmaps build simultaneously side-by-side
5. On completion, diff overlay becomes available

---

## Project Structure

```
testing/
  gui/
    backend/
      __init__.py
      app.py              # FastAPI app, CORS (localhost only), lifespan
      sessions.py         # SessionManager — model loading, cloning, VRAM tracking
      routes/
        __init__.py
        sessions.py       # REST session/surgery endpoints
        probes.py         # WebSocket probe endpoints
        inspect.py        # REST inspect endpoints
    frontend/
      package.json
      vite.config.ts      # includes proxy: { '/api': 'http://127.0.0.1:8000', '/ws': { target: 'ws://127.0.0.1:8000', ws: true } }
      src/
        App.tsx
        components/
          SessionSidebar.tsx
          ProbeConfig.tsx
          VisualizationArea.tsx
          GenerationOutput.tsx
          visualizations/
            LogitLensHeatmap.tsx
            LayerInfluence.tsx
            ABDiff.tsx
        hooks/
          useWebSocket.ts
          useSessions.ts
        state/
          store.ts         # Zustand store
        types/
          api.ts           # Shared types matching backend Pydantic schemas
```

### Dependencies

**Backend** (added to existing venv): `fastapi`, `uvicorn[standard]`

**Frontend**: `react`, `react-dom`, `typescript`, `vite`, `zustand`, `d3`, `@types/d3`

### Dev Workflow

- Backend: `uvicorn testing.gui.backend.app:app --host 127.0.0.1 --port 8000 --reload`
- Frontend: `cd testing/gui/frontend && npm run dev` (Vite on port 5173, proxies `/api/*` and `/ws/*` to backend)
- Combined: `./testing/gui/run.sh` — starts both processes

### Production (Local) Mode

Vite builds static files to `testing/gui/frontend/dist/`. FastAPI mounts them at `/` via `StaticFiles(directory="...", html=True)`. API routes are prefixed under `/api/`. WebSocket routes under `/ws/`. No conflict because API/WS paths are distinct from static file paths.

Single command: `uvicorn testing.gui.backend.app:app --host 127.0.0.1 --port 8000`

---

## Priority Visualizations (v1)

1. **Logit lens heatmap** — layer × position grid, color-coded confidence, hover for top-k
2. **Layer influence bars** — block influence score per layer
3. **Intervention A/B diff** — side-by-side logit lens with diff overlay
4. **Live generation** — token-by-token stream with clickable probability inspection

---

## Scope Boundaries

**In scope for v1:**
- Model loading/unloading with session management
- Surgery operations applied via UI with one-level undo
- The four priority visualizations listed above
- A/B comparison via clone + surgery
- WebSocket streaming for probes and generation
- VRAM tracking and pre-clone checks
- Localhost-only binding and CORS

**Out of scope for v1:**
- Experiment tracking integration (viewing/querying the experiments.db)
- Recipe editor/runner
- Export pipeline UI
- Remote access / multi-user
- CPU offload for models
- Intervened generation (interventions during autoregressive generation)
- Attention pattern visualization
- Hidden state similarity matrix
- Weight SVD visualization
