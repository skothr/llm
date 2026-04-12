# Live Probe GUI — Design Spec

## Overview

A browser-based live experimentation dashboard for interactive model probing, intervention, and visualization. Built on the existing `llm_surgeon` toolkit, it provides real-time feedback as you explore model internals.

**Primary use case:** Load a model into GPU memory, run probes and interventions interactively, see results update in real time. Optionally load a second model (or clone + surgery variant) for side-by-side A/B comparison.

**Stack:** FastAPI (Python backend) + React/Vite/TypeScript (frontend) + D3.js (visualizations)

**Deployment:** Local only. Browser and backend on the same machine (RTX 2080, 8GB VRAM).

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

**Session Manager** holds loaded models in GPU memory. Each session has a name, model reference, and metadata. Multiple sessions can be active for A/B comparison. Two 4-bit quantized TinyLlamas fit comfortably in 8GB; two OpenLLaMA 3Bs would be tight.

No changes to existing `llm_surgeon` code. The backend imports and calls the toolkit as-is. `on_layer` callbacks are wrapped to push WebSocket messages.

---

## Backend API

### Session Endpoints (REST)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sessions` | Load model by HF ID or local path. Returns session ID. |
| `GET` | `/sessions` | List active sessions with metadata (model name, layer count, memory usage). |
| `GET` | `/sessions/{id}/info` | Layer info, head count, hidden size. Wraps `get_layer_info`. |
| `DELETE` | `/sessions/{id}` | Unload model, free GPU memory. |
| `POST` | `/sessions/{id}/surgery` | Apply surgery operation. Returns updated layer info + surgery log. Model mutates in place. |
| `POST` | `/sessions/{id}/clone` | Deep-copy session to create a variant for A/B comparison. |

### Probe Endpoints (WebSocket)

| Path | Input | Streaming Output |
|------|-------|-----------------|
| `ws://.../sessions/{id}/logit-lens` | `{prompt, top_k}` | Layer-by-layer predictions via `on_layer`. Final message has full result. |
| `ws://.../sessions/{id}/generate` | `{prompt, max_tokens, temperature}` | Token-by-token with top-k probabilities per step. |
| `ws://.../sessions/{id}/intervene` | `{prompt, interventions, capture_logit_lens}` | Streamed intervention results with optional logit lens capture. |

### Inspect Endpoints (REST)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sessions/{id}/inspect/influence` | Block influence scores per layer. |
| `POST` | `/sessions/{id}/inspect/attention` | Attention entropy per head per layer. |
| `POST` | `/sessions/{id}/inspect/residual-norms` | Residual stream L2 norms through network. |

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

Backend maps `op` strings to `probe.ops.*` functions: `scale`, `zero_dims`, `clamp`, `noise`, `replace`, `project_out`. The `params` dict is unpacked as kwargs. Unknown ops return a validation error.

### Generation Mechanism

Live generation uses `model.generate()` directly on the HuggingFace model held in the session — not ollama. This keeps generation in the same process as probing, using the same (potentially surgically modified) model. A custom `LogitsProcessor` captures top-k token probabilities at each step and streams them over WebSocket.

### Clone Memory Cost

`POST /sessions/{id}/clone` performs `copy.deepcopy(model)` which approximately doubles GPU memory for that model. The endpoint checks available VRAM before cloning and returns an error if insufficient. The session list endpoint reports per-session memory estimates so the user can gauge headroom.

### WebSocket Message Protocol

All WebSocket messages have a `type` field:
- `"data"` — incremental result (one layer, one token)
- `"error"` — exception from toolkit, includes message string
- `"complete"` — operation finished, includes summary/full result

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
- Active sessions list with memory usage indicators
- Clone button per session
- Surgery panel — operation type selector, layer/head pickers, apply button
- Surgery log history per session

**Probe Config (bottom-left):**
- Prompt text area
- Operation selector: logit lens, influence, intervene, generate
- Operation-specific parameter controls (top_k, intervention ops, temperature, max_tokens)
- Target session selector (single or "both" for A/B)
- Run button

**Visualization Area (top-right):**
- Tab or stack layout for multiple result panels
- Results persist until cleared; multiple probes can be compared
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
- **Probes** — current config (prompt, operation, params, target sessions). Result history for scrollback.
- **Visualization** — which results are displayed, panel layout, A/B pairing.

### Streaming Data Flow (Logit Lens Example)

1. User types prompt, selects "logit lens", clicks Run
2. Frontend opens WebSocket to `/sessions/{id}/logit-lens`, sends config
3. Backend calls `probe.logit_lens()` with `on_layer` callback
4. Each callback fires WebSocket message: `{type: "data", layer: 5, sublayer: "ffn", predictions: [...]}`
5. Frontend appends each row to heatmap incrementally — builds layer by layer
6. Final message: `{type: "complete", summary: {...}}`
7. Result pushed to probe history, visualization persists

### A/B Comparison Flow

1. Two sessions loaded (e.g. "baseline" and "no-layer-19")
2. User selects "Run on both"
3. Frontend opens two WebSocket connections in parallel
4. Both heatmaps build simultaneously side-by-side
5. On completion, diff overlay becomes available

---

## Project Structure

```
testing/
  gui/
    backend/
      __init__.py
      app.py              # FastAPI app, CORS, lifespan
      sessions.py         # SessionManager — model loading, cloning, GPU memory
      routes/
        sessions.py       # REST session/surgery endpoints
        probes.py         # WebSocket probe endpoints
        inspect.py        # REST inspect endpoints
    frontend/
      package.json
      vite.config.ts
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
          api.ts           # Shared types matching backend schemas
```

### Dependencies

**Backend** (added to existing venv): `fastapi`, `uvicorn[standard]`, `websockets`

**Frontend**: `react`, `react-dom`, `typescript`, `vite`, `zustand`, `d3`, `@types/d3`

### Dev Workflow

- `uvicorn` serves API on port 8000
- Vite dev server on port 5173 with proxy to backend
- For local "production": Vite builds static files, FastAPI serves them — single process

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
- Surgery operations applied via UI
- The four priority visualizations listed above
- A/B comparison via clone + surgery
- WebSocket streaming for probes and generation

**Out of scope for v1:**
- Experiment tracking integration (viewing/querying the experiments.db)
- Recipe editor/runner
- Export pipeline UI
- Remote access / multi-user
- Attention pattern visualization
- Hidden state similarity matrix
- Weight SVD visualization
