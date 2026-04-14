# Live Probe GUI: Polish, Reliability & End-to-End

**Date:** 2026-04-13
**Scope:** Fix reliability issues, restructure layout, add missing features, make the full research loop work end-to-end.

## 1. Layout Restructure

### Current
Single 320px sidebar holds sessions, surgery, and probe config stacked vertically. Too cramped for the intervention builder and growing feature set.

### Proposed
```
┌────────────────────┬────────────────────────┐
│ [Sess][Probe][Int] │                        │
├────────────────────┤   Visualization Area   │
│                    │                        │
│  Config Panel      │                        │
│  (tab-dependent)   │                        │
│  320px             │                        │
│                    ├────────────────────────┤
│                    │   Output / Log         │
│                    │                        │
└────────────────────┴────────────────────────┘
```

- Tab bar is scoped to the left panel (does not span full width).
- Right side (visualization + output) is always visible regardless of active tab.
- Results persist across tab switches.

### Tab Contents

**Sessions tab:**
- Model loading combobox (see Section 5)
- Session list with model info, delete button
- Surgery controls: operation dropdown, dynamic param forms (see Section 6), clone, undo

**Probe tab:**
- Prompt textarea
- Operation selector: logit-lens, influence, attention, residual-norms, generate
- Session A/B dropdowns
- Dynamic parameters per operation (top_k, max_tokens, temperature, etc.)
- Run / Cancel buttons

Note: the `ProbeOperation` type in `api.ts` changes from `"logit-lens" | "influence" | "intervene" | "generate"` to `"logit-lens" | "influence" | "attention" | "residual-norms" | "generate"` — "intervene" moves out to its own tab/workflow.

**Intervene tab:**
- Prompt textarea
- Session selector
- Intervention stack builder (see Section 4)
- "Capture logit lens" toggle
- Run / Cancel buttons

## 2. WebSocket Connection Manager

### Problem
`useWebSocket` holds a single `wsRef`. A/B comparison overwrites the first connection. Cancel only affects the last-opened socket.

### Fix
Replace single ref with a keyed `Map<string, WebSocket>`:

```typescript
interface ConnectionManager {
  connect(key: string, path: string, config: Record<string, unknown>, handlers: WsHandlers): void;
  cancel(key: string): void;
  cancelAll(): void;
  disconnect(key: string): void;
}
```

- Callers provide a key (e.g. `"logit-lens-A"`, `"logit-lens-B"`)
- Each connection independently fires its own handlers
- `cancelAll()` tears down all connections (cancel button, unmount)
- No backend changes needed

## 3. Progressive Rendering

### 3a. Model Loading Feedback

- Add `isLoadingModel: boolean` and `loadingModelId: string | null` to Zustand store
- Sessions tab shows spinner with model ID during load
- Clear on response (success or error)

### 3b. Probe Streaming

- Add `pendingResults: Record<string, ProbeResult>` to Zustand store (keyed by result ID, matches WebSocket connection manager keys)
- On WebSocket `connect`: create entry in `pendingResults` with empty `data[]`
- On each `data` message: append to that entry's `data`, trigger store update
- On `complete`: move entry from `pendingResults` into `results[]`, remove from `pendingResults`
- On error/disconnect: finalize partial result with an error indicator, move to `results[]`

Visualization components check `pendingResults` first (for the active result ID), fall back to completed `results`:
- **LogitLensHeatmap**: re-renders with each new row (one row per sublayer per layer — e.g. 44 for TinyLlama 22L, 52 for OpenLLaMA 26L — sub-ms D3 redraw each)
- **GenerationOutput**: tokens appear one-by-one as they arrive
- **LayerInfluence / AttentionEntropy / ResidualNorms**: single REST response, no streaming needed

### A/B Streaming

For A/B runs, `pendingResults` holds two entries (keyed `"${id}-A"` and `"${id}-B"`). Both stream concurrently via separate WebSocket connections. ABDiff can render a partial comparison as rows arrive for both sides.

## 4. Intervention Builder

Dedicated panel in the Intervene tab.

### UI Structure
- Prompt textarea + session dropdown at top
- Scrollable list of intervention cards
- Each card: layer (number input), sublayer (attn/ffn toggle), operation dropdown, dynamic params
- [+ Add Intervention] button appends a new card
- [x] remove button on each card
- "Capture logit lens" checkbox
- Run / Cancel at bottom

### Operations and Their Params

| Op | Params | Form Controls |
|----|--------|--------------|
| scale | factor: float | number input, step 0.1 |
| zero_dims | dims: int[] | comma-separated number input |
| clamp | min: float, max: float | two number inputs |
| noise | std: float | number input, step 0.01 |
| replace | source: {session, prompt, layer, sublayer, position} | session dropdown, prompt input, layer/sublayer/position inputs |
| project_out | source: {session, prompt, layer, sublayer, position} | same as replace |

### State

Array of `InterventionSpec` objects in Zustand store:
```typescript
interface InterventionSpec {
  layer: number;
  sublayer: "attn" | "ffn";
  op: string;
  params: Record<string, unknown>;
}
```

On Run: serialize to the backend `intervene_ws` protocol. No backend changes.

### Results
- If `capture_logit_lens` is on: renders LogitLensHeatmap with modified layers highlighted (border/marker on rows where `modified: true`)
- If off: shows a summary of interventions applied

## 5. Model Selection Combobox

### Backend
New endpoint: `GET /api/models/available`
- Scans `testing/.cache/models/` for directories matching the HuggingFace cache pattern `models--{org}--{name}` (these contain `blobs/`, `refs/`, `snapshots/` subdirectories)
- Converts directory names to model IDs: strip the `models--` prefix, split remainder on `--`, join with `/`. Example: `models--TinyLlama--TinyLlama-1.1B-Chat-v1.0` → strip → `TinyLlama--TinyLlama-1.1B-Chat-v1.0` → split → `["TinyLlama", "TinyLlama-1.1B-Chat-v1.0"]` → join → `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Returns `[{"model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}, ...]`
- Lightweight — just a directory listing, no model loading

### Frontend
- Combobox: dropdown of cached models + text input that filters/autocompletes
- Typing a model ID not in the list is allowed — submits normally, backend downloads from HuggingFace
- Loading indicator distinguishes "loading from cache" vs "downloading + loading" (inferred from whether model ID was in the available list)

## 6. Surgery Param Forms

### Problem
Surgery params are currently a raw JSON textarea (`{"layer_indices": [0]}`).

### Fix
Dynamically generate form controls from the `SURGERY_OPS` schema already served by `GET /api/surgery/operations`:

| Schema Type | Form Control |
|-------------|-------------|
| `"int"` | number input |
| `"float"` | number input, step=0.1 |
| `{"type": "array", "items": "int"}` | comma-separated number input, parsed to `int[]` |

Each operation's params render as labeled inputs when that operation is selected from the dropdown.

Collapsible "Advanced: raw JSON" textarea remains as fallback for edge cases.

No backend changes — uses existing schema from `GET /api/surgery/operations`.

## 7. New Visualizations

All three inspect endpoints already exist in `gui/backend/routes/inspect.py` with full test coverage in `gui/tests/test_inspect_routes.py`. No backend changes needed — this section is frontend-only.

### Layer Influence (existing viz, already implemented)

**Endpoint:** `POST /api/sessions/{name}/inspect/influence` (exists)
**Visualization:** `LayerInfluence.tsx` bar chart already renders this. No changes needed.

### Attention Entropy (new frontend viz for existing endpoint)

**Endpoint:** `POST /api/sessions/{name}/inspect/attention` (exists at `inspect.py:43`)
**Response:** `{ entropy: { "layer_head": [float per position] } }`

**Visualization:** Small-multiples grid.
- Each cell = one attention head
- Cell contains a mini sparkline of entropy across token positions
- Cell background color: mean entropy mapped to blue (low/sharp) → red (high/diffuse)
- Helps spot interesting heads at a glance

### Residual Norms (new frontend viz for existing endpoint)

**Endpoint:** `POST /api/sessions/{name}/inspect/residual-norms` (exists at `inspect.py:59`)
**Response:** `{ norms: [float per layer] }`

**Visualization:** Line chart.
- X-axis: layer index
- Y-axis: L2 norm of residual stream
- Spikes = layers adding significant information
- Plateaus = layers barely changing the stream
- Complements logit lens: high norm change + no prediction flip = interesting layer

Both new viz components added to the Probe tab operation dropdown and VisualizationArea routing.

## 8. Error Handling

### Backend Offline
- `fetchSessions` failure on load shows banner: "Backend offline — start with `./gui/run.sh`"
- Poll every 5 seconds until connected, then auto-refresh

### Model Load Failure
- Bad model ID or OOM → error shown in Sessions panel, loading state cleared
- Error message from backend `HTTPException` detail displayed inline

### WebSocket Disconnect
- Mid-stream disconnect → finalize partial results with "connection lost" indicator
- Partial logit lens heatmap or partial generation output remain visible
- User can re-run from the same prompt

### Session Deleted During Probe
- WS receives "not found" error → surfaced inline, run state cleared

### No React Error Boundary
- App is small enough that per-component error state is sufficient
- Error boundary would show blank screen, worse than partial UI with inline error

## 9. Files Changed

### Backend (Python)
- `gui/backend/routes/sessions.py` — add `GET /api/models/available` endpoint
- No other backend changes needed

### Frontend (TypeScript/React)
- `src/App.tsx` — new tabbed layout
- `src/App.css` — tab bar styles, panel layout grid
- `src/state/store.ts` — add `pendingResults`, `isLoadingModel`, `interventionSpecs`, `activeTab`
- `src/hooks/useWebSocket.ts` — keyed connection Map
- `src/types/api.ts` — add `AttentionEntropyResult`, `ResidualNormsResult` if needed
- `src/components/SessionSidebar.tsx` → `src/components/SessionsPanel.tsx` — refactor, add model combobox, surgery param forms
- `src/components/ProbeConfig.tsx` → `src/components/ProbePanel.tsx` — refactor, add new operations
- New: `src/components/IntervenePanel.tsx` — intervention builder
- New: `src/components/TabBar.tsx` — tab navigation
- New: `src/components/ModelCombobox.tsx` — autocomplete model selector
- New: `src/components/SurgeryParamForm.tsx` — dynamic form from op schema
- `src/components/VisualizationArea.tsx` — add routing for new viz types, support pendingResult
- `src/components/GenerationOutput.tsx` — support pendingResult for streaming tokens
- `src/components/visualizations/LogitLensHeatmap.tsx` — support incremental data
- New: `src/components/visualizations/AttentionEntropy.tsx` — small-multiples grid
- New: `src/components/visualizations/ResidualNorms.tsx` — line chart

### Tests
- `gui/tests/test_probe_ws.py` — no changes (backend unchanged)
- New: `gui/tests/test_models_available.py` — test the new endpoint
- Frontend: manual testing via browser (no frontend test framework in place)

## 10. Out of Scope

- Attention head pattern visualization (full attention matrix heatmap) — future work
- Frontend test framework (Vitest, Playwright) — future work
- Component library / UI toolkit — current approach is sufficient
- Resizable panels — fixed layout for now
- Persistent state across page reloads — results are ephemeral
