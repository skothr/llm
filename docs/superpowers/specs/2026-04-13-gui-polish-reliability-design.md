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

**A/B scope:** A/B comparison applies to logit-lens and generate only (WebSocket streaming). For REST operations (influence, attention, residual-norms), Session B dropdown is disabled — single-session only. Matches current behavior at `ProbeConfig.tsx:87`.

**Influence prompt wrapping:** The backend `InfluenceRequest` takes `prompts: List[str]` (plural) for multi-prompt corpus analysis. The Probe tab has a single prompt textarea. On submit, wrap as `[prompt]`. This underutilizes the feature — a "prompt set" concept is future work.

**REST → ProbeResult bridging:** The three REST operations (influence, attention, residual-norms) bypass the WebSocket `pendingResults` flow. Instead, their `fetch` responses are wrapped as `ProbeResult` objects with `data: [{ type: "complete", ...responseData }]` and added directly to `results[]`. This pattern already exists in `ProbeConfig.tsx:64-84` for influence — attention and residual-norms follow the same pattern.

**Intervene tab:**
- Prompt textarea
- Session selector
- Intervention stack builder (see Section 4)
- "Capture logit lens" toggle
- Run / Cancel buttons

## 2. WebSocket Connection Manager

### Problems
1. `useWebSocket` holds a single `wsRef`. A/B comparison overwrites the first connection. Cancel only affects the last-opened socket.
2. `cancel()` sends `{ type: "cancel" }` over the socket, but **no backend handler reads it**. The `logit_lens_ws` and `intervene_ws` endpoints create a `cancelled = asyncio.Event()` but never call `await ws.receive_text()` after the initial config — nothing ever sets the event. The cancel message is dead code.

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
- `cancel(key)` calls `ws.close()` (not `ws.send({ type: "cancel" })`) — socket closure triggers `WebSocketDisconnect` on the backend, which is the actual cancellation mechanism
- `cancelAll()` closes all connections (cancel button, unmount)
- Remove the dead cancel machinery from backend `logit_lens_ws` (`probes.py`):
  - Line 42: `cancelled = asyncio.Event()` creation
  - Line 55: `cancelled.is_set()` guard in `on_layer` (keep the `if not connected:` guard)
  - Line 95: `cancelled.is_set()` check before sending complete
  - Line 101-102: `elif cancelled.is_set():` branch and `{"type": "cancelled"}` response
- Same cleanup in `intervene_ws` (uses `connected` flag only, no `cancelled` Event — but verify no dead references)
- No other backend changes needed

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
- **LogitLensHeatmap**: receives rows as individual WebSocket messages. Note: the backend runs the full computation in `run_in_executor` with a sync `on_layer` callback that collects results, then sends all rows as a rapid burst after computation finishes. The frontend processes each message individually, so D3 re-renders per row — the visual effect is a fast cascade rather than true one-at-a-time streaming. For small models (<30 layers) this is imperceptible; true progressive streaming (e.g. via `asyncio.Queue` bridging the executor thread) is a future improvement.
- **GenerationOutput**: tokens appear one-by-one as calculated. Unlike logit lens, `generate_ws` runs its loop inline and sends each token immediately — this IS true streaming.
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

Array of `InterventionSpec` objects in Zustand store. Uses the existing `InterventionSpec` type already defined in `api.ts:67-72`:
```typescript
// already exists — no new type needed
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
- `gui/backend/routes/probes.py` — remove dead `cancelled = asyncio.Event()` from `logit_lens_ws` and `intervene_ws`
- No other backend changes needed (all inspect endpoints already exist in `inspect.py`)

### Frontend (TypeScript/React)
- `src/App.tsx` — new tabbed layout
- `src/App.css` — tab bar styles, panel layout grid
- `src/state/store.ts` — add `pendingResults`, `isLoadingModel`, `interventionSpecs`, `activeTab`
- `src/hooks/useWebSocket.ts` — keyed connection Map
- `src/types/api.ts` — update `ProbeOperation` type (drop `"intervene"`, add `"attention" | "residual-norms"`). Existing `AttentionResult` and `ResidualNormsResult` types are already defined and usable as-is. Existing `InterventionSpec` type is also already defined.
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
