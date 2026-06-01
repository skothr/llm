# GUI Polish & Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix WebSocket reliability for A/B comparison, restructure layout into tabbed panels, add progressive rendering, build the intervention builder, add missing visualizations, and make the full research loop work end-to-end.

**Architecture:** Tabbed left panel (Sessions/Probe/Intervene) with independent right-side visualization and output areas. Keyed WebSocket connection manager supports concurrent A/B streams. `pendingResults` in Zustand store enables progressive rendering. One new backend endpoint (`GET /api/models/available`); all other changes are frontend.

**Tech Stack:** Python/FastAPI (backend), React 18 + TypeScript + Zustand + D3.js (frontend), Vite dev server with proxy.

**Spec:** `docs/superpowers/specs/2026-04-13-gui-polish-reliability-design.md`

**Test commands:**
- Backend: `PYTHONPATH=testing testing/.venv/bin/python -m pytest testing/gui/tests/ -v`
- Frontend type-check: `cd testing/gui/frontend && npx tsc --noEmit`
- Frontend build: `cd testing/gui/frontend && npx vite build`

---

## Task 1: Backend — models/available endpoint

**Files:**
- Modify: `testing/gui/backend/routes/sessions.py`
- Create: `testing/gui/tests/test_models_available.py`

- [ ] **Step 1: Write the test**

Create `testing/gui/tests/test_models_available.py`:

```python
import pytest
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from gui.backend.app import app

@pytest.fixture
def mock_cache(tmp_path):
    m1 = tmp_path / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
    m1.mkdir()
    (m1 / "blobs").mkdir()
    (m1 / "refs").mkdir()
    (m1 / "snapshots").mkdir()

    m2 = tmp_path / "models--openlm-research--open_llama_3b_v2"
    m2.mkdir()
    (m2 / "blobs").mkdir()
    (m2 / "refs").mkdir()
    (m2 / "snapshots").mkdir()

    (tmp_path / "CACHEDIR.TAG").touch()
    (tmp_path / "some_random_dir").mkdir()

    return tmp_path

@pytest.mark.asyncio
async def test_models_available(mock_cache, monkeypatch):
    import gui.backend.routes.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "MODELS_CACHE", mock_cache)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models/available")
    assert resp.status_code == 200
    data = resp.json()
    model_ids = [m["model_id"] for m in data]
    assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in model_ids
    assert "openlm-research/open_llama_3b_v2" in model_ids
    assert len(model_ids) == 2

@pytest.mark.asyncio
async def test_models_available_empty(tmp_path, monkeypatch):
    import gui.backend.routes.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "MODELS_CACHE", tmp_path)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models/available")
    assert resp.status_code == 200
    assert resp.json() == []

@pytest.mark.asyncio
async def test_models_available_missing_dir(tmp_path, monkeypatch):
    import gui.backend.routes.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "MODELS_CACHE", tmp_path / "nonexistent")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models/available")
    assert resp.status_code == 200
    assert resp.json() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=testing testing/.venv/bin/python -m pytest testing/gui/tests/test_models_available.py -v`
Expected: FAIL — `MODELS_CACHE` attribute not found, no route for `/api/models/available`

- [ ] **Step 3: Implement the endpoint**

Add to `testing/gui/backend/routes/sessions.py`, after the imports and before `_NAME_RE`:

```python
from pathlib import Path

MODELS_CACHE = Path(__file__).resolve().parent.parent.parent.parent / ".cache" / "models"

def _scan_model_cache(cache_dir: Path) -> list:
    if not cache_dir.exists():
        return []
    models = []
    for entry in cache_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("models--"):
            parts = entry.name[len("models--"):].split("--", 1)
            if len(parts) == 2:
                models.append(parts[0] + "/" + parts[1])
    return sorted(models)
```

Add the route after `surgery_operations()`:

```python
@router.get("/models/available")
async def list_available_models():
    return [{"model_id": m} for m in _scan_model_cache(MODELS_CACHE)]
```

Note: `Path` is already imported via `from pathlib import Path` in the existing `sys.path` block inside `load_session`. Move the import to the top of the file (module level).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=testing testing/.venv/bin/python -m pytest testing/gui/tests/test_models_available.py -v`
Expected: 3 passed

- [ ] **Step 5: Run full backend test suite**

Run: `PYTHONPATH=testing testing/.venv/bin/python -m pytest testing/gui/tests/ -v`
Expected: 39 passed (36 existing + 3 new)

- [ ] **Step 6: Commit**

```bash
git add testing/gui/backend/routes/sessions.py testing/gui/tests/test_models_available.py
git commit -m "feat(gui): add GET /api/models/available endpoint

Scans HuggingFace cache directory for locally available models.
Returns model IDs derived from cache directory names."
```

---

## Task 2: Backend — remove cancel dead code

**Files:**
- Modify: `testing/gui/backend/routes/probes.py`

- [ ] **Step 1: Remove dead cancel code from logit_lens_ws**

In `testing/gui/backend/routes/probes.py`, function `logit_lens_ws`:

1. Remove line 42: `cancelled = asyncio.Event()`
2. Change line 55-56 from:
   ```python
   if cancelled.is_set() or not connected:
   ```
   to:
   ```python
   if not connected:
   ```
3. Change line 95 from:
   ```python
   if connected and not cancelled.is_set():
   ```
   to:
   ```python
   if connected:
   ```
4. Remove lines 101-102:
   ```python
   elif cancelled.is_set():
       await _send_json(ws, {"type": "cancelled"})
   ```

After edits, the `logit_lens_ws` function should have no references to `cancelled`.

- [ ] **Step 2: Verify intervene_ws has no dead cancel references**

Search `intervene_ws` in the same file for any `cancelled` references. It should only use `connected`. Confirm no changes needed.

- [ ] **Step 3: Run test suite**

Run: `PYTHONPATH=testing testing/.venv/bin/python -m pytest testing/gui/tests/ -v`
Expected: 39 passed (all existing tests still pass)

- [ ] **Step 4: Commit**

```bash
git add testing/gui/backend/routes/probes.py
git commit -m "fix(gui): remove dead cancel code from logit_lens_ws

The cancelled asyncio.Event was created but never set — no backend
handler reads cancel messages from the client. Cancellation works
via socket closure (WebSocketDisconnect) instead."
```

---

## Task 3: Frontend foundation — types, store, WebSocket hook

**Files:**
- Modify: `testing/gui/frontend/src/types/api.ts`
- Rewrite: `testing/gui/frontend/src/hooks/useWebSocket.ts`
- Rewrite: `testing/gui/frontend/src/state/store.ts`

- [ ] **Step 1: Update api.ts**

In `testing/gui/frontend/src/types/api.ts`:

1. Remove `WsCancelled` interface and remove it from `WsMessage` union:

   Replace:
   ```typescript
   export interface WsCancelled {
     type: "cancelled";
   }

   export type WsMessage = LogitLensData | GenerateData | InterveneData | WsComplete | WsError | WsCancelled;
   ```
   With:
   ```typescript
   export type WsMessage = LogitLensData | GenerateData | InterveneData | WsComplete | WsError;
   ```

2. Change `ProbeOperation` and add `ResultOperation`:

   Replace:
   ```typescript
   export type ProbeOperation = "logit-lens" | "influence" | "intervene" | "generate";
   ```
   With:
   ```typescript
   export type ProbeOperation = "logit-lens" | "influence" | "attention" | "residual-norms" | "generate";
   export type ResultOperation = ProbeOperation | "intervene";
   export type ConfigTab = "sessions" | "probe" | "intervene";
   ```

3. Update `ProbeResult` to use `ResultOperation`:

   Replace:
   ```typescript
   export interface ProbeResult {
     id: string;
     operation: ProbeOperation;
   ```
   With:
   ```typescript
   export interface ProbeResult {
     id: string;
     operation: ResultOperation;
   ```

- [ ] **Step 2: Rewrite useWebSocket.ts**

Replace the entire content of `testing/gui/frontend/src/hooks/useWebSocket.ts`:

```typescript
import { useRef, useCallback } from "react";
import type { WsMessage } from "../types/api";

interface WsHandlers {
  onMessage: (msg: WsMessage) => void;
  onComplete: (msg: WsMessage) => void;
  onError: (message: string) => void;
}

export function useWebSocket() {
  const connectionsRef = useRef<Map<string, WebSocket>>(new Map());

  const connect = useCallback(
    (key: string, path: string, config: Record<string, unknown>, handlers: WsHandlers) => {
      const existing = connectionsRef.current.get(key);
      if (existing && existing.readyState === WebSocket.OPEN) {
        existing.close();
      }

      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${window.location.host}${path}`);
      connectionsRef.current.set(key, ws);

      ws.onopen = () => {
        ws.send(JSON.stringify(config));
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data) as WsMessage;
        if (msg.type === "data") {
          handlers.onMessage(msg);
        } else if (msg.type === "complete") {
          handlers.onComplete(msg);
        } else if (msg.type === "error") {
          handlers.onError((msg as { message: string }).message);
        }
      };

      ws.onerror = () => {
        handlers.onError("WebSocket connection error");
      };

      ws.onclose = () => {
        connectionsRef.current.delete(key);
      };

      return ws;
    },
    []
  );

  const cancel = useCallback((key: string) => {
    const ws = connectionsRef.current.get(key);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
    connectionsRef.current.delete(key);
  }, []);

  const cancelAll = useCallback(() => {
    connectionsRef.current.forEach((ws) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    });
    connectionsRef.current.clear();
  }, []);

  const disconnect = useCallback((key: string) => {
    const ws = connectionsRef.current.get(key);
    if (ws) {
      ws.close();
      connectionsRef.current.delete(key);
    }
  }, []);

  return { connect, cancel, cancelAll, disconnect };
}
```

- [ ] **Step 3: Rewrite store.ts**

Replace the entire content of `testing/gui/frontend/src/state/store.ts`:

```typescript
import { create } from "zustand";
import type {
  SessionSummary,
  SessionInfo,
  SurgeryOperation,
  ProbeOperation,
  ProbeResult,
  ConfigTab,
  InterventionSpec,
  WsMessage,
} from "../types/api";

async function apiError(resp: Response): Promise<Error> {
  const body = await resp.json().catch(() => ({ detail: resp.statusText }));
  const detail = body.detail;
  if (typeof detail === "string") return new Error(detail);
  if (Array.isArray(detail)) {
    return new Error(detail.map((d: { msg?: string; loc?: string[] }) =>
      `${(d.loc || []).join(".")}: ${d.msg || "validation error"}`
    ).join("; "));
  }
  return new Error(JSON.stringify(detail));
}

interface StoreState {
  activeTab: ConfigTab;
  backendOnline: boolean;

  sessions: SessionSummary[];
  sessionInfo: Record<string, SessionInfo>;
  surgeryOps: SurgeryOperation[];

  isLoadingModel: boolean;
  loadingModelId: string | null;
  availableModels: string[];

  prompt: string;
  operation: ProbeOperation;
  targetSession: string;
  targetSessionB: string | null;
  isRunning: boolean;

  results: ProbeResult[];
  activeResultId: string | null;
  pendingResults: Record<string, ProbeResult>;

  interventionSpecs: InterventionSpec[];
  captureLogitLens: boolean;
  intervenePrompt: string;
  interveneSession: string;

  setActiveTab: (tab: ConfigTab) => void;
  fetchSessions: () => Promise<void>;
  fetchSessionInfo: (name: string) => Promise<void>;
  fetchSurgeryOps: () => Promise<void>;
  fetchAvailableModels: () => Promise<void>;
  deleteSession: (name: string) => Promise<void>;
  applySurgery: (name: string, operation: string, params: Record<string, unknown>) => Promise<void>;
  undoSurgery: (name: string) => Promise<void>;
  cloneSession: (name: string, targetName: string) => Promise<void>;

  setPrompt: (prompt: string) => void;
  setOperation: (op: ProbeOperation) => void;
  setTargetSession: (name: string) => void;
  setTargetSessionB: (name: string | null) => void;
  setRunning: (running: boolean) => void;

  addResult: (result: ProbeResult) => void;
  clearResults: () => void;
  setActiveResult: (id: string | null) => void;
  setPendingResult: (id: string, result: ProbeResult) => void;
  updatePendingResult: (id: string, msg: WsMessage) => void;
  finalizePendingResult: (id: string, extraData?: WsMessage) => void;
  removePendingResult: (id: string) => void;

  addIntervention: () => void;
  removeIntervention: (index: number) => void;
  updateIntervention: (index: number, spec: InterventionSpec) => void;
  clearInterventions: () => void;
  setCaptureLogitLens: (v: boolean) => void;
  setIntervenePrompt: (prompt: string) => void;
  setInterveneSession: (name: string) => void;
}

export const useStore = create<StoreState>((set, get) => ({
  activeTab: "sessions",
  backendOnline: false,

  sessions: [],
  sessionInfo: {},
  surgeryOps: [],

  isLoadingModel: false,
  loadingModelId: null,
  availableModels: [],

  prompt: "",
  operation: "logit-lens",
  targetSession: "",
  targetSessionB: null,
  isRunning: false,

  results: [],
  activeResultId: null,
  pendingResults: {},

  interventionSpecs: [],
  captureLogitLens: false,
  intervenePrompt: "",
  interveneSession: "",

  setActiveTab: (tab) => set({ activeTab: tab }),

  fetchSessions: async () => {
    try {
      const resp = await fetch("/api/sessions");
      const data = await resp.json();
      set({ sessions: data, backendOnline: true });
      if (get().surgeryOps.length === 0) {
        get().fetchSurgeryOps();
      }
    } catch {
      set({ backendOnline: false });
    }
  },

  fetchSessionInfo: async (name: string) => {
    const resp = await fetch(`/api/sessions/${name}/info`);
    const data = await resp.json();
    set((s) => ({ sessionInfo: { ...s.sessionInfo, [name]: data } }));
  },

  fetchSurgeryOps: async () => {
    try {
      const resp = await fetch("/api/surgery/operations");
      const data = await resp.json();
      set({ surgeryOps: data });
    } catch { /* backend not ready */ }
  },

  fetchAvailableModels: async () => {
    try {
      const resp = await fetch("/api/models/available");
      const data = await resp.json();
      set({ availableModels: data.map((m: { model_id: string }) => m.model_id) });
    } catch { /* backend not ready */ }
  },

  deleteSession: async (name: string) => {
    await fetch(`/api/sessions/${name}`, { method: "DELETE" });
    await get().fetchSessions();
  },

  applySurgery: async (name: string, operation: string, params: Record<string, unknown>) => {
    const resp = await fetch(`/api/sessions/${name}/surgery`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ operation, params }),
    });
    if (!resp.ok) throw await apiError(resp);
    await get().fetchSessions();
    await get().fetchSessionInfo(name);
  },

  undoSurgery: async (name: string) => {
    const resp = await fetch(`/api/sessions/${name}/surgery/undo`, { method: "POST" });
    if (!resp.ok) throw await apiError(resp);
    await get().fetchSessions();
    await get().fetchSessionInfo(name);
  },

  cloneSession: async (name: string, targetName: string) => {
    const resp = await fetch(`/api/sessions/${name}/clone`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_name: targetName }),
    });
    if (!resp.ok) throw await apiError(resp);
    await get().fetchSessions();
  },

  setPrompt: (prompt) => set({ prompt }),
  setOperation: (operation) => set({ operation }),
  setTargetSession: (name) => set({ targetSession: name }),
  setTargetSessionB: (name) => set({ targetSessionB: name }),
  setRunning: (running) => set({ isRunning: running }),

  addResult: (result) => set((s) => ({
    results: [result, ...s.results],
    activeResultId: result.id,
  })),
  clearResults: () => set({ results: [], activeResultId: null }),
  setActiveResult: (id) => set({ activeResultId: id }),

  setPendingResult: (id, result) => set((s) => ({
    pendingResults: { ...s.pendingResults, [id]: result },
    activeResultId: id,
  })),

  updatePendingResult: (id, msg) => set((s) => {
    const pending = s.pendingResults[id];
    if (!pending) return s;
    return {
      pendingResults: {
        ...s.pendingResults,
        [id]: { ...pending, data: [...pending.data, msg] },
      },
    };
  }),

  finalizePendingResult: (id, extraData) => set((s) => {
    const pending = s.pendingResults[id];
    if (!pending) return s;
    const finalData = extraData ? [...pending.data, extraData] : pending.data;
    const { [id]: _, ...remaining } = s.pendingResults;
    return {
      pendingResults: remaining,
      results: [{ ...pending, data: finalData }, ...s.results],
      activeResultId: pending.id,
    };
  }),

  removePendingResult: (id) => set((s) => {
    const { [id]: _, ...remaining } = s.pendingResults;
    return { pendingResults: remaining };
  }),

  addIntervention: () => set((s) => ({
    interventionSpecs: [...s.interventionSpecs, { layer: 0, sublayer: "ffn", op: "scale", params: { factor: 1.0 } }],
  })),
  removeIntervention: (index) => set((s) => ({
    interventionSpecs: s.interventionSpecs.filter((_, i) => i !== index),
  })),
  updateIntervention: (index, spec) => set((s) => ({
    interventionSpecs: s.interventionSpecs.map((existing, i) => i === index ? spec : existing),
  })),
  clearInterventions: () => set({ interventionSpecs: [] }),
  setCaptureLogitLens: (v) => set({ captureLogitLens: v }),
  setIntervenePrompt: (prompt) => set({ intervenePrompt: prompt }),
  setInterveneSession: (name) => set({ interveneSession: name }),
}));
```

- [ ] **Step 4: Type-check**

Run: `cd testing/gui/frontend && npx tsc --noEmit`
Expected: type errors in `App.tsx`, `ProbeConfig.tsx`, `SessionSidebar.tsx` (old components referencing removed types). These will be fixed in subsequent tasks. The foundation files themselves should have no internal errors.

- [ ] **Step 5: Commit**

```bash
git add testing/gui/frontend/src/types/api.ts testing/gui/frontend/src/hooks/useWebSocket.ts testing/gui/frontend/src/state/store.ts
git commit -m "feat(gui): foundation — types, store, WebSocket connection manager

- Add ResultOperation, ConfigTab types; update ProbeOperation
- Keyed WebSocket Map for concurrent A/B connections
- Add pendingResults for progressive rendering
- Add intervention builder state
- Add backend online tracking and model loading state"
```

---

## Task 4: Layout, TabBar, and CSS

**Files:**
- Rewrite: `testing/gui/frontend/src/App.tsx`
- Rewrite: `testing/gui/frontend/src/App.css`
- Create: `testing/gui/frontend/src/components/TabBar.tsx`

- [ ] **Step 1: Create TabBar.tsx**

Create `testing/gui/frontend/src/components/TabBar.tsx`:

```tsx
import { useStore } from "../state/store";
import type { ConfigTab } from "../types/api";

const TABS: { id: ConfigTab; label: string }[] = [
  { id: "sessions", label: "Sessions" },
  { id: "probe", label: "Probe" },
  { id: "intervene", label: "Intervene" },
];

export function TabBar() {
  const activeTab = useStore((s) => s.activeTab);
  const setActiveTab = useStore((s) => s.setActiveTab);

  return (
    <div className="tab-bar">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          className={`tab-button ${activeTab === tab.id ? "active" : ""}`}
          onClick={() => setActiveTab(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Rewrite App.css**

Replace the entire content of `testing/gui/frontend/src/App.css`:

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
  background: #1a1a2e;
  color: #e0e0e0;
}

.app-layout {
  display: grid;
  grid-template-columns: 320px 1fr;
  height: 100vh;
}

.config-panel {
  display: flex;
  flex-direction: column;
  background: #16213e;
  border-right: 1px solid #0f3460;
  overflow: hidden;
}

.tab-bar {
  display: flex;
  border-bottom: 1px solid #0f3460;
  flex-shrink: 0;
}

.tab-button {
  flex: 1;
  padding: 8px 4px;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  border: none;
  border-bottom: 2px solid transparent;
  background: transparent;
  color: #8888aa;
  cursor: pointer;
  border-radius: 0;
}

.tab-button:hover {
  background: rgba(26, 82, 118, 0.3);
}

.tab-button.active {
  color: #e0e0e0;
  border-bottom-color: #4ecdc4;
  background: transparent;
}

.config-content {
  flex: 1;
  padding: 16px;
  overflow-y: auto;
}

.main-area {
  display: grid;
  grid-template-rows: 1fr 280px;
}

.visualization-area {
  padding: 16px;
  overflow: auto;
}

.generation-output {
  padding: 16px;
  border-top: 1px solid #0f3460;
  overflow: auto;
}

h2 {
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #a0a0c0;
  margin-bottom: 12px;
}

button {
  background: #0f3460;
  color: #e0e0e0;
  border: 1px solid #1a5276;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}

button:hover {
  background: #1a5276;
}

button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

input, textarea, select {
  background: #0d1b2a;
  color: #e0e0e0;
  border: 1px solid #1a5276;
  padding: 6px 8px;
  border-radius: 4px;
  font-size: 13px;
  width: 100%;
}

textarea {
  resize: vertical;
  min-height: 60px;
  font-family: monospace;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

.cursor-blink {
  animation: blink 1s step-end infinite;
}
```

- [ ] **Step 3: Rewrite App.tsx**

Replace the entire content of `testing/gui/frontend/src/App.tsx`:

```tsx
import { useEffect } from "react";
import { useStore } from "./state/store";
import { TabBar } from "./components/TabBar";
import { SessionsPanel } from "./components/SessionsPanel";
import { ProbePanel } from "./components/ProbePanel";
import { IntervenePanel } from "./components/IntervenePanel";
import { VisualizationArea } from "./components/VisualizationArea";
import { GenerationOutput } from "./components/GenerationOutput";

export default function App() {
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSurgeryOps = useStore((s) => s.fetchSurgeryOps);
  const activeTab = useStore((s) => s.activeTab);
  const backendOnline = useStore((s) => s.backendOnline);

  useEffect(() => {
    fetchSessions();
    fetchSurgeryOps();
  }, [fetchSessions, fetchSurgeryOps]);

  useEffect(() => {
    if (backendOnline) return;
    const interval = setInterval(() => { fetchSessions(); }, 5000);
    return () => clearInterval(interval);
  }, [backendOnline, fetchSessions]);

  return (
    <div className="app-layout">
      <aside className="config-panel">
        <TabBar />
        <div className="config-content">
          {activeTab === "sessions" && <SessionsPanel />}
          {activeTab === "probe" && <ProbePanel />}
          {activeTab === "intervene" && <IntervenePanel />}
        </div>
      </aside>
      <main className="main-area">
        <div className="visualization-area">
          <VisualizationArea />
        </div>
        <div className="generation-output">
          <GenerationOutput />
        </div>
      </main>
    </div>
  );
}
```

Note: This references `SessionsPanel`, `ProbePanel`, and `IntervenePanel` which don't exist yet. TypeScript will error until those are created in Tasks 6-8.

- [ ] **Step 4: Commit**

```bash
git add testing/gui/frontend/src/App.tsx testing/gui/frontend/src/App.css testing/gui/frontend/src/components/TabBar.tsx
git commit -m "feat(gui): tabbed layout with TabBar

Restructure from sidebar layout to tabbed config panel.
Tabs scoped to left panel, right side independent."
```

---

## Task 5: Utility components — ModelCombobox, SurgeryParamForm

**Files:**
- Create: `testing/gui/frontend/src/components/ModelCombobox.tsx`
- Create: `testing/gui/frontend/src/components/SurgeryParamForm.tsx`

- [ ] **Step 1: Create ModelCombobox.tsx**

Create `testing/gui/frontend/src/components/ModelCombobox.tsx`:

```tsx
import { useState, useEffect } from "react";
import { useStore } from "../state/store";

export function ModelCombobox({
  value,
  onChange,
}: {
  value: string;
  onChange: (modelId: string) => void;
}) {
  const availableModels = useStore((s) => s.availableModels);
  const [isOpen, setIsOpen] = useState(false);
  const [filter, setFilter] = useState(value);

  useEffect(() => { setFilter(value); }, [value]);

  const filtered = availableModels.filter((m) =>
    m.toLowerCase().includes(filter.toLowerCase())
  );

  return (
    <div style={{ position: "relative" }}>
      <input
        value={filter}
        onChange={(e) => {
          setFilter(e.target.value);
          onChange(e.target.value);
          setIsOpen(true);
        }}
        onFocus={() => setIsOpen(true)}
        onBlur={() => setTimeout(() => setIsOpen(false), 200)}
        placeholder="Model ID (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
      />
      {isOpen && filtered.length > 0 && (
        <div style={{
          position: "absolute",
          top: "100%",
          left: 0,
          right: 0,
          background: "#0d1b2a",
          border: "1px solid #1a5276",
          borderRadius: "0 0 4px 4px",
          maxHeight: 150,
          overflowY: "auto",
          zIndex: 10,
        }}>
          {filtered.map((m) => (
            <div
              key={m}
              style={{ padding: "4px 8px", cursor: "pointer", fontSize: 12 }}
              onMouseDown={() => {
                onChange(m);
                setFilter(m);
                setIsOpen(false);
              }}
              onMouseEnter={(e) => { (e.target as HTMLElement).style.background = "#1a5276"; }}
              onMouseLeave={(e) => { (e.target as HTMLElement).style.background = "transparent"; }}
            >
              {m}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Create SurgeryParamForm.tsx**

Create `testing/gui/frontend/src/components/SurgeryParamForm.tsx`:

```tsx
import { useState } from "react";
import type { SurgeryOperation } from "../types/api";

interface Props {
  operation: SurgeryOperation;
  params: Record<string, unknown>;
  onChange: (params: Record<string, unknown>) => void;
}

export function SurgeryParamForm({ operation, params, onChange }: Props) {
  const [showJson, setShowJson] = useState(false);
  const [jsonText, setJsonText] = useState(JSON.stringify(params, null, 2));

  if (showJson) {
    return (
      <div>
        <textarea
          value={jsonText}
          onChange={(e) => {
            setJsonText(e.target.value);
            try { onChange(JSON.parse(e.target.value)); } catch { /* invalid JSON */ }
          }}
          rows={3}
          style={{ marginBottom: 4 }}
        />
        <button onClick={() => setShowJson(false)} style={{ fontSize: 11 }}>Form view</button>
      </div>
    );
  }

  return (
    <div>
      {Object.entries(operation.params).map(([key, schema]) => {
        const schemaObj = typeof schema === "string" ? { type: schema } : schema;
        const type = schemaObj.type || "string";

        if (type === "int") {
          return (
            <label key={key} style={{ display: "block", fontSize: 12, marginBottom: 4 }}>
              {key}: <input
                type="number"
                step={1}
                value={Number(params[key] ?? 0)}
                onChange={(e) => onChange({ ...params, [key]: parseInt(e.target.value) || 0 })}
                style={{ width: 60 }}
              />
            </label>
          );
        }
        if (type === "float") {
          return (
            <label key={key} style={{ display: "block", fontSize: 12, marginBottom: 4 }}>
              {key}: <input
                type="number"
                step={0.1}
                value={Number(params[key] ?? 0)}
                onChange={(e) => onChange({ ...params, [key]: parseFloat(e.target.value) || 0 })}
                style={{ width: 60 }}
              />
            </label>
          );
        }
        if (type === "array") {
          return (
            <label key={key} style={{ display: "block", fontSize: 12, marginBottom: 4 }}>
              {key}: <input
                value={Array.isArray(params[key]) ? (params[key] as number[]).join(", ") : ""}
                onChange={(e) => onChange({
                  ...params,
                  [key]: e.target.value.split(",").map((s) => parseInt(s.trim())).filter((n) => !isNaN(n)),
                })}
                placeholder="0, 1, 2"
                style={{ width: "100%" }}
              />
            </label>
          );
        }
        return null;
      })}
      <button onClick={() => { setJsonText(JSON.stringify(params, null, 2)); setShowJson(true); }} style={{ fontSize: 11 }}>
        JSON
      </button>
    </div>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add testing/gui/frontend/src/components/ModelCombobox.tsx testing/gui/frontend/src/components/SurgeryParamForm.tsx
git commit -m "feat(gui): add ModelCombobox and SurgeryParamForm

ModelCombobox autocompletes from cached models list.
SurgeryParamForm generates form controls from op schema
with JSON fallback."
```

---

## Task 6: SessionsPanel

**Files:**
- Create: `testing/gui/frontend/src/components/SessionsPanel.tsx`

- [ ] **Step 1: Create SessionsPanel.tsx**

Create `testing/gui/frontend/src/components/SessionsPanel.tsx`:

```tsx
import { useState, useEffect } from "react";
import { useStore } from "../state/store";
import { ModelCombobox } from "./ModelCombobox";
import { SurgeryParamForm } from "./SurgeryParamForm";
import type { SessionSummary, SurgeryOperation } from "../types/api";

export function SessionsPanel() {
  const sessions = useStore((s) => s.sessions);
  const sessionInfo = useStore((s) => s.sessionInfo);
  const surgeryOps = useStore((s) => s.surgeryOps);
  const isLoadingModel = useStore((s) => s.isLoadingModel);
  const loadingModelId = useStore((s) => s.loadingModelId);
  const availableModels = useStore((s) => s.availableModels);
  const backendOnline = useStore((s) => s.backendOnline);
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSessionInfo = useStore((s) => s.fetchSessionInfo);
  const fetchAvailableModels = useStore((s) => s.fetchAvailableModels);
  const deleteSession = useStore((s) => s.deleteSession);
  const applySurgery = useStore((s) => s.applySurgery);
  const undoSurgery = useStore((s) => s.undoSurgery);
  const cloneSession = useStore((s) => s.cloneSession);

  const [loadModelId, setLoadModelId] = useState("");
  const [loadName, setLoadName] = useState("");
  const [loadMode, setLoadMode] = useState<"inspect" | "eval">("inspect");
  const [error, setError] = useState("");

  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [cloneName, setCloneName] = useState("");
  const [surgeryOp, setSurgeryOp] = useState("");
  const [surgeryParams, setSurgeryParams] = useState<Record<string, unknown>>({});

  useEffect(() => { fetchAvailableModels(); }, [fetchAvailableModels]);

  const selectedOpDef = surgeryOps.find((op) => op.name === surgeryOp) || null;

  useEffect(() => {
    if (!selectedOpDef) { setSurgeryParams({}); return; }
    const defaults: Record<string, unknown> = {};
    for (const [key, schema] of Object.entries(selectedOpDef.params)) {
      const s = typeof schema === "string" ? { type: schema } : schema;
      if (s.type === "array") defaults[key] = [];
      else if (s.type === "float") defaults[key] = 0.0;
      else defaults[key] = 0;
    }
    setSurgeryParams(defaults);
  }, [surgeryOp, selectedOpDef]);

  const handleLoad = async () => {
    setError("");
    useStore.setState({ isLoadingModel: true, loadingModelId: loadModelId });
    try {
      const resp = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: loadName, model_id: loadModelId, mode: loadMode }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({ detail: "Failed to load model" }));
        const detail = body.detail;
        throw new Error(
          typeof detail === "string" ? detail :
          Array.isArray(detail) ? detail.map((d: { msg?: string }) => d.msg || "error").join("; ") :
          "Failed to load model"
        );
      }
      await fetchSessions();
      await fetchAvailableModels();
      setLoadModelId("");
      setLoadName("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      useStore.setState({ isLoadingModel: false, loadingModelId: null });
    }
  };

  const handleClone = async (name: string) => {
    if (!cloneName) return;
    try { await cloneSession(name, cloneName); setCloneName(""); }
    catch (e) { setError((e as Error).message); }
  };

  const handleSurgery = async (name: string) => {
    try { await applySurgery(name, surgeryOp, surgeryParams); }
    catch (e) { setError((e as Error).message); }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {!backendOnline && (
        <div style={{ padding: 8, background: "#4a2020", borderRadius: 4, fontSize: 12 }}>
          Backend offline — start with <code>./gui/run.sh</code>
        </div>
      )}

      <div>
        <ModelCombobox value={loadModelId} onChange={setLoadModelId} />
        <input
          placeholder="Session name (slug)"
          value={loadName}
          onChange={(e) => setLoadName(e.target.value)}
          style={{ marginTop: 4, marginBottom: 4 }}
        />
        <div style={{ display: "flex", gap: 4 }}>
          <select value={loadMode} onChange={(e) => setLoadMode(e.target.value as "inspect" | "eval")}>
            <option value="inspect">4-bit (inspect)</option>
            <option value="eval">fp16 (eval)</option>
          </select>
          <button onClick={handleLoad} disabled={isLoadingModel || !loadModelId || !loadName}>
            {isLoadingModel ? "Loading..." : "Load"}
          </button>
        </div>
        {isLoadingModel && (
          <div style={{ fontSize: 12, color: "#8888aa", marginTop: 4 }}>
            {availableModels.includes(loadModelId) ? `Loading ${loadingModelId} from cache...` : `Downloading ${loadingModelId}...`}
          </div>
        )}
      </div>

      {error && <div style={{ color: "#ff6b6b", fontSize: 12 }}>{error}</div>}

      {sessions.map((s: SessionSummary) => (
        <div
          key={s.name}
          style={{
            padding: 8,
            background: selectedSession === s.name ? "#1a5276" : "#0d1b2a",
            borderRadius: 4,
            cursor: "pointer",
          }}
          onClick={() => { setSelectedSession(s.name); fetchSessionInfo(s.name); }}
        >
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <strong>{s.name}</strong>
            <button
              onClick={(e) => { e.stopPropagation(); deleteSession(s.name); }}
              style={{ padding: "2px 6px", fontSize: 11 }}
            >x</button>
          </div>
          <div style={{ fontSize: 11, color: "#8888aa" }}>
            {s.model_id} | {s.mode} | {s.num_layers}L
            {s.has_snapshot && ` | undo: ${s.snapshot_size_mb.toFixed(0)}MB`}
          </div>
        </div>
      ))}

      {selectedSession && (
        <div style={{ borderTop: "1px solid #0f3460", paddingTop: 8 }}>
          <h3 style={{ fontSize: 12, color: "#a0a0c0", marginBottom: 8 }}>Surgery - {selectedSession}</h3>

          <div style={{ marginBottom: 8 }}>
            <input placeholder="Clone name" value={cloneName} onChange={(e) => setCloneName(e.target.value)} style={{ marginBottom: 4 }} />
            <button onClick={() => handleClone(selectedSession)} disabled={!cloneName}>Clone</button>
          </div>

          <div style={{ marginBottom: 8 }}>
            <select value={surgeryOp} onChange={(e) => setSurgeryOp(e.target.value)} style={{ marginBottom: 4 }}>
              <option value="">Select operation...</option>
              {surgeryOps.map((op: SurgeryOperation) => (
                <option key={op.name} value={op.name}>{op.name}</option>
              ))}
            </select>
            {selectedOpDef && (
              <SurgeryParamForm operation={selectedOpDef} params={surgeryParams} onChange={setSurgeryParams} />
            )}
            <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
              <button onClick={() => handleSurgery(selectedSession)} disabled={!surgeryOp}>Apply</button>
              <button
                onClick={() => undoSurgery(selectedSession)}
                disabled={!sessions.find((s) => s.name === selectedSession)?.has_snapshot}
              >Undo</button>
            </div>
          </div>

          {sessionInfo[selectedSession] && (
            <div style={{ fontSize: 12, color: "#8888aa" }}>
              <div>Layers: {sessionInfo[selectedSession].num_layers}</div>
              <div>Heads: {sessionInfo[selectedSession].num_heads}</div>
              <div>Hidden: {sessionInfo[selectedSession].hidden_size}</div>
              <div>Params: {(sessionInfo[selectedSession].total_params / 1e6).toFixed(1)}M</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add testing/gui/frontend/src/components/SessionsPanel.tsx
git commit -m "feat(gui): add SessionsPanel with model combobox and surgery forms

Replaces SessionSidebar. Integrates ModelCombobox for autocomplete
model selection and SurgeryParamForm for dynamic surgery params.
Adds backend offline banner and model loading indicator."
```

---

## Task 7: ProbePanel with progressive rendering

**Files:**
- Create: `testing/gui/frontend/src/components/ProbePanel.tsx`

- [ ] **Step 1: Create ProbePanel.tsx**

Create `testing/gui/frontend/src/components/ProbePanel.tsx`:

```tsx
import { useState } from "react";
import { useStore } from "../state/store";
import { useWebSocket } from "../hooks/useWebSocket";
import type { WsMessage, ProbeOperation } from "../types/api";

const WS_OPS = new Set<ProbeOperation>(["logit-lens", "generate"]);

export function ProbePanel() {
  const prompt = useStore((s) => s.prompt);
  const operation = useStore((s) => s.operation);
  const targetSession = useStore((s) => s.targetSession);
  const targetSessionB = useStore((s) => s.targetSessionB);
  const isRunning = useStore((s) => s.isRunning);
  const sessions = useStore((s) => s.sessions);
  const setPrompt = useStore((s) => s.setPrompt);
  const setOperation = useStore((s) => s.setOperation);
  const setTargetSession = useStore((s) => s.setTargetSession);
  const setTargetSessionB = useStore((s) => s.setTargetSessionB);
  const setRunning = useStore((s) => s.setRunning);
  const addResult = useStore((s) => s.addResult);
  const setPendingResult = useStore((s) => s.setPendingResult);
  const updatePendingResult = useStore((s) => s.updatePendingResult);
  const finalizePendingResult = useStore((s) => s.finalizePendingResult);

  const { connect, cancelAll } = useWebSocket();
  const [topK, setTopK] = useState(10);
  const [maxTokens, setMaxTokens] = useState(64);
  const [temperature, setTemperature] = useState(0.0);
  const [repPenalty, setRepPenalty] = useState(1.0);
  const [stopSeqs, setStopSeqs] = useState("\\n\\n");
  const [error, setError] = useState("");

  const isWs = WS_OPS.has(operation);

  const makeWsHandlers = (resultId: string, isFinalConnection: boolean) => ({
    onMessage: (msg: WsMessage) => { updatePendingResult(resultId, msg); },
    onComplete: (msg: WsMessage) => {
      finalizePendingResult(resultId, msg);
      if (isFinalConnection) setRunning(false);
    },
    onError: (message: string) => {
      finalizePendingResult(resultId);
      setError(message);
      if (isFinalConnection) setRunning(false);
    },
  });

  const getWsConfig = () => {
    if (operation === "logit-lens") return { prompt, top_k: topK };
    return {
      prompt, max_tokens: maxTokens, temperature, prob_top_k: topK,
      repetition_penalty: repPenalty,
      stop_sequences: stopSeqs.split(",").map((s) => s.replace(/\\n/g, "\n").trim()).filter(Boolean),
    };
  };

  const getWsPath = (session: string) => {
    if (operation === "logit-lens") return `/ws/sessions/${session}/logit-lens`;
    return `/ws/sessions/${session}/generate`;
  };

  const handleRun = () => {
    if (!targetSession || !prompt) return;
    setError("");
    setRunning(true);

    const resultId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

    if (isWs) {
      const hasB = !!targetSessionB;

      setPendingResult(resultId, {
        id: resultId, operation, sessionName: targetSession, prompt, data: [], timestamp: Date.now(),
      });
      connect(resultId, getWsPath(targetSession), getWsConfig(), makeWsHandlers(resultId, !hasB));

      if (hasB) {
        const idB = `${resultId}-B`;
        setPendingResult(idB, {
          id: idB, operation, sessionName: targetSessionB!, prompt, data: [], timestamp: Date.now(),
        });
        connect(idB, getWsPath(targetSessionB!), getWsConfig(), makeWsHandlers(idB, true));
      }
    } else {
      let url = "";
      let body: unknown = {};

      if (operation === "influence") {
        url = `/api/sessions/${targetSession}/inspect/influence`;
        body = { prompts: [prompt] };
      } else if (operation === "attention") {
        url = `/api/sessions/${targetSession}/inspect/attention`;
        body = { prompt };
      } else if (operation === "residual-norms") {
        url = `/api/sessions/${targetSession}/inspect/residual-norms`;
        body = { prompt };
      }

      fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
        .then((r) => r.json())
        .then((data) => {
          addResult({
            id: resultId, operation, sessionName: targetSession, prompt,
            data: [{ type: "complete" as const, ...data }],
            timestamp: Date.now(),
          });
          setRunning(false);
        })
        .catch((e) => { setError((e as Error).message); setRunning(false); });
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <textarea
        placeholder="Prompt text..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={3}
      />

      <select value={operation} onChange={(e) => setOperation(e.target.value as ProbeOperation)}>
        <option value="logit-lens">Logit Lens</option>
        <option value="generate">Generate</option>
        <option value="influence">Layer Influence</option>
        <option value="attention">Attention Entropy</option>
        <option value="residual-norms">Residual Norms</option>
      </select>

      <div style={{ display: "flex", gap: 4 }}>
        <select value={targetSession} onChange={(e) => setTargetSession(e.target.value)}>
          <option value="">Session A...</option>
          {sessions.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
        </select>
        <select
          value={targetSessionB ?? ""}
          onChange={(e) => setTargetSessionB(e.target.value || null)}
          disabled={!isWs}
          title={!isWs ? "A/B not available for REST operations" : ""}
        >
          <option value="">Session B (A/B)...</option>
          {sessions.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
        </select>
      </div>

      {(operation === "logit-lens" || operation === "generate") && (
        <div style={{ display: "flex", gap: 8, fontSize: 12 }}>
          <label>top_k: <input type="number" value={topK} onChange={(e) => setTopK(+e.target.value)} style={{ width: 50 }} /></label>
          {operation === "generate" && (
            <>
              <label>max: <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)} style={{ width: 50 }} /></label>
              <label>temp: <input type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(+e.target.value)} style={{ width: 50 }} /></label>
              <label>rep: <input type="number" step="0.1" value={repPenalty} onChange={(e) => setRepPenalty(+e.target.value)} style={{ width: 50 }} /></label>
            </>
          )}
        </div>
      )}

      {operation === "generate" && (
        <div style={{ fontSize: 12 }}>
          <label>stop: <input value={stopSeqs} onChange={(e) => setStopSeqs(e.target.value)} placeholder="comma-separated, use \n for newline" style={{ width: "100%" }} /></label>
        </div>
      )}

      <div style={{ display: "flex", gap: 4 }}>
        {!isRunning ? (
          <button onClick={handleRun} disabled={!targetSession || !prompt}>Run</button>
        ) : (
          <button onClick={() => { cancelAll(); setRunning(false); }} style={{ background: "#6b2020" }}>Cancel</button>
        )}
      </div>

      {error && <div style={{ color: "#ff6b6b", fontSize: 12 }}>{error}</div>}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add testing/gui/frontend/src/components/ProbePanel.tsx
git commit -m "feat(gui): add ProbePanel with progressive rendering

Replaces ProbeConfig. Uses pendingResults for streaming logit lens
and generation. Adds attention and residual-norms operations.
Session B disabled for REST operations."
```

---

## Task 8: IntervenePanel

**Files:**
- Create: `testing/gui/frontend/src/components/IntervenePanel.tsx`

- [ ] **Step 1: Create IntervenePanel.tsx**

Create `testing/gui/frontend/src/components/IntervenePanel.tsx`:

```tsx
import { useState } from "react";
import { useStore } from "../state/store";
import { useWebSocket } from "../hooks/useWebSocket";
import type { InterventionSpec, WsMessage } from "../types/api";

const INTERVENTION_OPS = [
  { name: "scale", params: [{ key: "factor", type: "float", default: 1.0, step: 0.1 }] },
  { name: "zero_dims", params: [{ key: "dims", type: "int[]", default: "" }] },
  { name: "clamp", params: [{ key: "min", type: "float", default: -1.0, step: 0.1 }, { key: "max", type: "float", default: 1.0, step: 0.1 }] },
  { name: "noise", params: [{ key: "std", type: "float", default: 0.1, step: 0.01 }] },
  { name: "replace", params: [{ key: "source", type: "source" }] },
  { name: "project_out", params: [{ key: "source", type: "source" }] },
];

function defaultParamsForOp(opName: string): Record<string, unknown> {
  const opDef = INTERVENTION_OPS.find((o) => o.name === opName) || INTERVENTION_OPS[0];
  const params: Record<string, unknown> = {};
  for (const p of opDef.params) {
    if (p.type === "source") params[p.key] = { session: "", prompt: "", layer: 0, sublayer: "ffn", position: 0 };
    else if (p.type === "int[]") params[p.key] = [];
    else params[p.key] = p.default;
  }
  return params;
}

function InterventionCard({
  spec, index, sessions, onUpdate, onRemove,
}: {
  spec: InterventionSpec;
  index: number;
  sessions: { name: string }[];
  onUpdate: (index: number, spec: InterventionSpec) => void;
  onRemove: (index: number) => void;
}) {
  const opDef = INTERVENTION_OPS.find((o) => o.name === spec.op) || INTERVENTION_OPS[0];

  const updateParam = (key: string, value: unknown) => {
    onUpdate(index, { ...spec, params: { ...spec.params, [key]: value } });
  };

  return (
    <div style={{ padding: 8, background: "#0d1b2a", borderRadius: 4, marginBottom: 4 }}>
      <div style={{ display: "flex", gap: 4, marginBottom: 4 }}>
        <label style={{ fontSize: 12 }}>
          L<input type="number" value={spec.layer} onChange={(e) => onUpdate(index, { ...spec, layer: +e.target.value })} style={{ width: 36 }} />
        </label>
        <select value={spec.sublayer} onChange={(e) => onUpdate(index, { ...spec, sublayer: e.target.value as "attn" | "ffn" })} style={{ width: 60 }}>
          <option value="ffn">FFN</option>
          <option value="attn">Attn</option>
        </select>
        <select value={spec.op} onChange={(e) => onUpdate(index, { ...spec, op: e.target.value, params: defaultParamsForOp(e.target.value) })} style={{ flex: 1 }}>
          {INTERVENTION_OPS.map((o) => <option key={o.name} value={o.name}>{o.name}</option>)}
        </select>
        <button onClick={() => onRemove(index)} style={{ padding: "2px 6px", fontSize: 11 }}>x</button>
      </div>

      {opDef.params.map((p) => {
        if (p.type === "float") {
          return (
            <label key={p.key} style={{ display: "block", fontSize: 12, marginBottom: 2 }}>
              {p.key}: <input type="number" step={p.step} value={Number(spec.params[p.key] ?? p.default)} onChange={(e) => updateParam(p.key, +e.target.value)} style={{ width: 70 }} />
            </label>
          );
        }
        if (p.type === "int[]") {
          return (
            <label key={p.key} style={{ display: "block", fontSize: 12, marginBottom: 2 }}>
              {p.key}: <input
                value={Array.isArray(spec.params[p.key]) ? (spec.params[p.key] as number[]).join(", ") : ""}
                onChange={(e) => updateParam(p.key, e.target.value.split(",").map((s) => parseInt(s.trim())).filter((n) => !isNaN(n)))}
                placeholder="0, 1, 2"
                style={{ width: 120 }}
              />
            </label>
          );
        }
        if (p.type === "source") {
          const source = (spec.params[p.key] || { session: "", prompt: "", layer: 0, sublayer: "ffn", position: 0 }) as {
            session: string; prompt: string; layer: number; sublayer: string; position: number;
          };
          const updateSource = (field: string, value: unknown) => updateParam(p.key, { ...source, [field]: value });
          return (
            <div key={p.key} style={{ fontSize: 12, paddingLeft: 8, borderLeft: "2px solid #1a5276", marginTop: 4 }}>
              <label style={{ display: "block", marginBottom: 2 }}>
                session: <select value={source.session} onChange={(e) => updateSource("session", e.target.value)} style={{ width: "100%" }}>
                  <option value="">Select...</option>
                  {sessions.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
                </select>
              </label>
              <label style={{ display: "block", marginBottom: 2 }}>
                prompt: <input value={source.prompt} onChange={(e) => updateSource("prompt", e.target.value)} />
              </label>
              <div style={{ display: "flex", gap: 4 }}>
                <label>L<input type="number" value={source.layer} onChange={(e) => updateSource("layer", +e.target.value)} style={{ width: 36 }} /></label>
                <select value={source.sublayer} onChange={(e) => updateSource("sublayer", e.target.value)} style={{ width: 60 }}>
                  <option value="ffn">FFN</option>
                  <option value="attn">Attn</option>
                </select>
                <label>pos<input type="number" value={source.position} onChange={(e) => updateSource("position", +e.target.value)} style={{ width: 40 }} /></label>
              </div>
            </div>
          );
        }
        return null;
      })}
    </div>
  );
}

export function IntervenePanel() {
  const sessions = useStore((s) => s.sessions);
  const interventionSpecs = useStore((s) => s.interventionSpecs);
  const captureLogitLens = useStore((s) => s.captureLogitLens);
  const intervenePrompt = useStore((s) => s.intervenePrompt);
  const interveneSession = useStore((s) => s.interveneSession);
  const isRunning = useStore((s) => s.isRunning);
  const addIntervention = useStore((s) => s.addIntervention);
  const removeIntervention = useStore((s) => s.removeIntervention);
  const updateIntervention = useStore((s) => s.updateIntervention);
  const setCaptureLogitLens = useStore((s) => s.setCaptureLogitLens);
  const setIntervenePrompt = useStore((s) => s.setIntervenePrompt);
  const setInterveneSession = useStore((s) => s.setInterveneSession);
  const setRunning = useStore((s) => s.setRunning);
  const setPendingResult = useStore((s) => s.setPendingResult);
  const updatePendingResult = useStore((s) => s.updatePendingResult);
  const finalizePendingResult = useStore((s) => s.finalizePendingResult);

  const { connect, cancelAll } = useWebSocket();
  const [error, setError] = useState("");

  const handleRun = () => {
    if (!interveneSession || !intervenePrompt || interventionSpecs.length === 0) return;
    setError("");
    setRunning(true);

    const resultId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

    setPendingResult(resultId, {
      id: resultId,
      operation: captureLogitLens ? "intervene" : "intervene",
      sessionName: interveneSession,
      prompt: intervenePrompt,
      data: [],
      timestamp: Date.now(),
    });

    connect(resultId, `/ws/sessions/${interveneSession}/intervene`, {
      prompt: intervenePrompt,
      interventions: interventionSpecs,
      capture_logit_lens: captureLogitLens,
    }, {
      onMessage: (msg: WsMessage) => { updatePendingResult(resultId, msg); },
      onComplete: (msg: WsMessage) => { finalizePendingResult(resultId, msg); setRunning(false); },
      onError: (message: string) => { finalizePendingResult(resultId); setError(message); setRunning(false); },
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <textarea
        placeholder="Prompt text..."
        value={intervenePrompt}
        onChange={(e) => setIntervenePrompt(e.target.value)}
        rows={3}
      />

      <select value={interveneSession} onChange={(e) => setInterveneSession(e.target.value)}>
        <option value="">Select session...</option>
        {sessions.map((s) => <option key={s.name} value={s.name}>{s.name}</option>)}
      </select>

      <div style={{ maxHeight: 300, overflowY: "auto" }}>
        {interventionSpecs.map((spec, i) => (
          <InterventionCard key={i} spec={spec} index={i} sessions={sessions} onUpdate={updateIntervention} onRemove={removeIntervention} />
        ))}
      </div>

      <button onClick={addIntervention} style={{ fontSize: 12 }}>+ Add Intervention</button>

      <label style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 4 }}>
        <input type="checkbox" checked={captureLogitLens} onChange={(e) => setCaptureLogitLens(e.target.checked)} />
        Capture logit lens
      </label>

      <div style={{ display: "flex", gap: 4 }}>
        {!isRunning ? (
          <button onClick={handleRun} disabled={!interveneSession || !intervenePrompt || interventionSpecs.length === 0}>Run</button>
        ) : (
          <button onClick={() => { cancelAll(); setRunning(false); }} style={{ background: "#6b2020" }}>Cancel</button>
        )}
      </div>

      {error && <div style={{ color: "#ff6b6b", fontSize: 12 }}>{error}</div>}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add testing/gui/frontend/src/components/IntervenePanel.tsx
git commit -m "feat(gui): add IntervenePanel with intervention stack builder

Supports all six intervention ops with dynamic param forms.
Source picker for replace/project_out. Optional logit lens capture."
```

---

## Task 9: Visualization updates — VisualizationArea, GenerationOutput, LogitLensHeatmap

**Files:**
- Rewrite: `testing/gui/frontend/src/components/VisualizationArea.tsx`
- Rewrite: `testing/gui/frontend/src/components/GenerationOutput.tsx`
- Modify: `testing/gui/frontend/src/components/visualizations/LogitLensHeatmap.tsx`

- [ ] **Step 1: Rewrite VisualizationArea.tsx**

Replace the entire content of `testing/gui/frontend/src/components/VisualizationArea.tsx`:

```tsx
import { useStore } from "../state/store";
import { LogitLensHeatmap } from "./visualizations/LogitLensHeatmap";
import { LayerInfluence } from "./visualizations/LayerInfluence";
import { ABDiff } from "./visualizations/ABDiff";
import { AttentionEntropy } from "./visualizations/AttentionEntropy";
import { ResidualNorms } from "./visualizations/ResidualNorms";
import type { ProbeResult } from "../types/api";

function getResult(
  results: ProbeResult[],
  pending: Record<string, ProbeResult>,
  id: string | null,
): ProbeResult | undefined {
  if (!id) return undefined;
  return pending[id] || results.find((r) => r.id === id);
}

export function VisualizationArea() {
  const results = useStore((s) => s.results);
  const pendingResults = useStore((s) => s.pendingResults);
  const activeResultId = useStore((s) => s.activeResultId);
  const setActiveResult = useStore((s) => s.setActiveResult);
  const clearResults = useStore((s) => s.clearResults);

  const all = [...Object.values(pendingResults), ...results];
  const vizResults = all.filter((r) => r.operation !== "generate");
  const activeResult = getResult(results, pendingResults, activeResultId);
  const isPending = activeResultId ? activeResultId in pendingResults : false;

  const abPair = activeResult && !activeResult.id.includes("-B")
    ? all.find((r) => r.id === `${activeResult.id}-B`)
    : null;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <h2>
          Visualization
          {isPending && <span style={{ color: "#4ecdc4", fontSize: 11, marginLeft: 8 }}>streaming...</span>}
        </h2>
        {vizResults.length > 0 && (
          <button onClick={clearResults} style={{ fontSize: 11 }}>Clear All</button>
        )}
      </div>

      {vizResults.length > 1 && (
        <div style={{ display: "flex", gap: 4, marginBottom: 8, flexWrap: "wrap" }}>
          {vizResults.filter((r) => !r.id.includes("-B")).map((r) => (
            <button
              key={r.id}
              onClick={() => setActiveResult(r.id)}
              style={{
                fontSize: 11, padding: "2px 8px",
                background: r.id === activeResultId ? "#1a5276" : "#0d1b2a",
              }}
            >
              {r.operation} | {r.sessionName}
              {all.find((b) => b.id === `${r.id}-B`) ? " (A/B)" : ""}
              {r.id in pendingResults ? " ..." : ""}
            </button>
          ))}
        </div>
      )}

      {activeResult ? (
        abPair && activeResult.operation === "logit-lens" ? (
          <ABDiff resultA={activeResult} resultB={abPair} />
        ) : activeResult.operation === "logit-lens" || activeResult.operation === "intervene" ? (
          <LogitLensHeatmap result={activeResult} />
        ) : activeResult.operation === "influence" ? (
          <LayerInfluence result={activeResult} />
        ) : activeResult.operation === "attention" ? (
          <AttentionEntropy result={activeResult} />
        ) : activeResult.operation === "residual-norms" ? (
          <ResidualNorms result={activeResult} />
        ) : (
          <p style={{ color: "#666" }}>No visualization for {activeResult.operation}</p>
        )
      ) : (
        <p style={{ color: "#666" }}>Run a probe to see results here</p>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Rewrite GenerationOutput.tsx**

Replace the entire content of `testing/gui/frontend/src/components/GenerationOutput.tsx`:

```tsx
import { useState } from "react";
import { useStore } from "../state/store";
import type { GenerateData, ProbeResult } from "../types/api";

function displayToken(text: string): string {
  if (text === "") return "<empty>";
  return text.replace(/\n/g, "\\n").replace(/\t/g, "\\t").replace(/\r/g, "\\r").replace(/ /g, "\u00B7");
}

export function GenerationOutput() {
  const results = useStore((s) => s.results);
  const pendingResults = useStore((s) => s.pendingResults);

  const all = [...Object.values(pendingResults), ...results];
  const genResults = all
    .filter((r) => r.operation === "generate")
    .sort((a, b) => {
      const aB = a.id.endsWith("-B") ? 1 : 0;
      const bB = b.id.endsWith("-B") ? 1 : 0;
      return aB - bB || a.timestamp - b.timestamp;
    });

  if (genResults.length === 0) {
    return (
      <div>
        <h2>Generation Output</h2>
        <p style={{ color: "#666" }}>Run generate to see token stream here</p>
      </div>
    );
  }

  return (
    <div>
      <h2>Generation Output</h2>
      <div style={{ display: "flex", gap: 16 }}>
        {genResults.slice(0, 2).map((r) => (
          <GenerationPanel key={r.id} result={r} isPending={r.id in pendingResults} />
        ))}
      </div>
    </div>
  );
}

function GenerationPanel({ result, isPending }: { result: ProbeResult; isPending: boolean }) {
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  const tokens = result.data.filter(
    (m): m is GenerateData => m.type === "data" && "token" in m && "step" in m
  );

  const selectedToken = selectedStep !== null ? tokens.find((t) => t.step === selectedStep) || null : null;

  return (
    <div style={{ flex: 1 }}>
      <div style={{ fontSize: 12, color: "#8888aa", marginBottom: 4 }}>
        {result.sessionName} - "{result.prompt.slice(0, 30)}"
        {isPending && <span style={{ color: "#4ecdc4", marginLeft: 4 }}>generating...</span>}
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 14, lineHeight: 1.8 }}>
        <span style={{ color: "#6688aa" }}>{result.prompt}</span>
        {tokens.map((tok) =>
          tok.token === "<eos>" ? (
            <span key={tok.step} onClick={() => setSelectedStep(selectedStep === tok.step ? null : tok.step)}
              style={{ cursor: "pointer", background: selectedStep === tok.step ? "#1a5276" : "transparent", color: "#4a6a4a", fontSize: 11, padding: "1px 3px", borderRadius: 2, border: "1px solid #3a5a3a" }}>eos</span>
          ) : (
            <span key={tok.step} onClick={() => setSelectedStep(selectedStep === tok.step ? null : tok.step)}
              style={{ cursor: "pointer", background: selectedStep === tok.step ? "#1a5276" : "transparent", borderRadius: 2, padding: "0 1px", whiteSpace: "pre" }}>{tok.token}</span>
          )
        )}
        {isPending && <span className="cursor-blink" style={{ color: "#4ecdc4" }}>|</span>}
      </div>

      {selectedToken && (
        <div style={{ marginTop: 8, padding: 8, background: "#0d1b2a", borderRadius: 4, fontSize: 12 }}>
          <div style={{ color: "#a0a0c0", marginBottom: 4 }}>Step {selectedToken.step} - top alternatives:</div>
          {selectedToken.top_k.map((alt, i) => (
            <div key={i} style={{ display: "flex", gap: 8 }}>
              <span style={{ color: i === 0 ? "#4ecdc4" : "#888", fontFamily: "monospace" }}>{displayToken(alt.token)}</span>
              <span style={{ color: "#666" }}>{(alt.prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Add modified-layer highlighting to LogitLensHeatmap**

In `testing/gui/frontend/src/components/visualizations/LogitLensHeatmap.tsx`, inside the `dataMessages.forEach` loop (after the existing `g.append("text")` for the cell label, around line 87), add a modified-layer indicator:

After:
```typescript
      msg.predictions.forEach((posPreds, posIdx) => {
```

Check if this message has a `modified` flag. Add before the existing `const topPred = posPreds[0];` line:

Replace the line:
```typescript
    dataMessages.forEach((msg, rowIdx) => {
      const label = `L${msg.layer}.${msg.sublayer}`;
```

With:
```typescript
    dataMessages.forEach((msg, rowIdx) => {
      const isModified = "modified" in msg && (msg as { modified?: boolean }).modified;
      const label = `L${msg.layer}.${msg.sublayer}`;
```

Then after the row label text element (the `g.append("text")` that writes `label`), add:

```typescript
      if (isModified) {
        g.append("rect")
          .attr("x", -8)
          .attr("y", rowIdx * cellH)
          .attr("width", 3)
          .attr("height", cellH - 1)
          .attr("fill", "#ff6b6b")
          .attr("rx", 1);
      }
```

- [ ] **Step 4: Commit**

```bash
git add testing/gui/frontend/src/components/VisualizationArea.tsx testing/gui/frontend/src/components/GenerationOutput.tsx testing/gui/frontend/src/components/visualizations/LogitLensHeatmap.tsx
git commit -m "feat(gui): update viz area for pendingResults and streaming

VisualizationArea routes attention/residual-norms/intervene results.
GenerationOutput shows blinking cursor during streaming.
LogitLensHeatmap highlights modified layers for intervention results."
```

---

## Task 10: AttentionEntropy visualization

**Files:**
- Create: `testing/gui/frontend/src/components/visualizations/AttentionEntropy.tsx`

- [ ] **Step 1: Create AttentionEntropy.tsx**

Create `testing/gui/frontend/src/components/visualizations/AttentionEntropy.tsx`:

```tsx
import { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import type { ProbeResult } from "../../types/api";

interface Props {
  result: ProbeResult;
}

export function AttentionEntropy({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);

  const completeMsg = result.data.find((m) => m.type === "complete") as
    | { type: "complete"; entropy: Record<string, number[]> }
    | undefined;

  useEffect(() => {
    if (!svgRef.current || !completeMsg?.entropy) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const entries = Object.entries(completeMsg.entropy).map(([key, values]) => {
      const parts = key.split("_").map(Number);
      return { layer: parts[0], head: parts[1], values, mean: d3.mean(values) || 0 };
    }).sort((a, b) => a.layer - b.layer || a.head - b.head);

    if (entries.length === 0) return;

    const numLayers = d3.max(entries, (d) => d.layer)! + 1;
    const numHeads = d3.max(entries, (d) => d.head)! + 1;
    const cellSize = 60;
    const sparkHeight = 30;
    const margin = { top: 30, right: 20, bottom: 20, left: 40 };
    const gap = 4;
    const width = margin.left + numHeads * (cellSize + gap) + margin.right;
    const height = margin.top + numLayers * (cellSize + gap) + margin.bottom;

    svg.attr("width", width).attr("height", height);

    const maxEntropy = d3.max(entries, (d) => d.mean) || 1;
    const bgScale = d3.scaleSequential(d3.interpolateRdYlBu).domain([maxEntropy, 0]);
    const numPositions = entries[0].values.length || 1;
    const sparkX = d3.scaleLinear().domain([0, numPositions - 1]).range([2, cellSize - 2]);
    const sparkYMax = d3.max(entries, (d) => d3.max(d.values) || 0) || 1;
    const sparkY = d3.scaleLinear().domain([0, sparkYMax]).range([cellSize - 4, cellSize - sparkHeight]);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    d3.range(numHeads).forEach((h) => {
      g.append("text").attr("x", h * (cellSize + gap) + cellSize / 2).attr("y", -8)
        .attr("text-anchor", "middle").attr("font-size", 10).attr("fill", "#8888aa").text(`H${h}`);
    });
    d3.range(numLayers).forEach((l) => {
      g.append("text").attr("x", -4).attr("y", l * (cellSize + gap) + cellSize / 2)
        .attr("text-anchor", "end").attr("dominant-baseline", "middle").attr("font-size", 10).attr("fill", "#8888aa").text(`L${l}`);
    });

    entries.forEach((d) => {
      const x = d.head * (cellSize + gap);
      const y = d.layer * (cellSize + gap);

      g.append("rect").attr("x", x).attr("y", y).attr("width", cellSize).attr("height", cellSize)
        .attr("fill", bgScale(d.mean)).attr("rx", 3).attr("opacity", 0.3).style("cursor", "pointer")
        .on("mouseenter", (event) => {
          setTooltip({ x: event.pageX + 10, y: event.pageY - 10,
            content: `L${d.layer} H${d.head}\nmean: ${d.mean.toFixed(3)}\nmin: ${d3.min(d.values)!.toFixed(3)}\nmax: ${d3.max(d.values)!.toFixed(3)}` });
        })
        .on("mouseleave", () => setTooltip(null));

      const line = d3.line<number>().x((_, i) => x + sparkX(i)).y((v) => y + sparkY(v));
      g.append("path").datum(d.values).attr("d", line)
        .attr("fill", "none").attr("stroke", "#e0e0e0").attr("stroke-width", 1).style("pointer-events", "none");

      g.append("text").attr("x", x + cellSize / 2).attr("y", y + 12)
        .attr("text-anchor", "middle").attr("font-size", 9).attr("fill", "#ccccdd").style("pointer-events", "none")
        .text(d.mean.toFixed(2));
    });
  }, [completeMsg]);

  if (!completeMsg?.entropy) return <p style={{ color: "#666" }}>No attention entropy data</p>;

  return (
    <div style={{ position: "relative" }}>
      <h3 style={{ fontSize: 13, color: "#a0a0c0", marginBottom: 8 }}>Attention Entropy - {result.sessionName}</h3>
      <div style={{ overflowX: "auto" }}><svg ref={svgRef} /></div>
      {tooltip && (
        <div style={{ position: "fixed", left: tooltip.x, top: tooltip.y, background: "#16213e", border: "1px solid #1a5276", borderRadius: 4, padding: "6px 10px", fontSize: 12, whiteSpace: "pre", pointerEvents: "none", zIndex: 100 }}>
          {tooltip.content}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add testing/gui/frontend/src/components/visualizations/AttentionEntropy.tsx
git commit -m "feat(gui): add AttentionEntropy small-multiples visualization

D3 grid of sparklines per attention head. Background color by mean
entropy (blue=sharp, red=diffuse). Hover tooltips."
```

---

## Task 11: ResidualNorms visualization

**Files:**
- Create: `testing/gui/frontend/src/components/visualizations/ResidualNorms.tsx`

- [ ] **Step 1: Create ResidualNorms.tsx**

Create `testing/gui/frontend/src/components/visualizations/ResidualNorms.tsx`:

```tsx
import { useRef, useEffect } from "react";
import * as d3 from "d3";
import type { ProbeResult } from "../../types/api";

interface Props {
  result: ProbeResult;
}

export function ResidualNorms({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  const completeMsg = result.data.find((m) => m.type === "complete") as
    | { type: "complete"; norms: number[] }
    | undefined;

  useEffect(() => {
    if (!svgRef.current || !completeMsg?.norms) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const norms = completeMsg.norms;
    const margin = { top: 20, right: 20, bottom: 35, left: 60 };
    const width = 500;
    const height = 200;

    svg.attr("width", width + margin.left + margin.right)
       .attr("height", height + margin.top + margin.bottom);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3.scaleLinear().domain([0, norms.length - 1]).range([0, width]);
    const y = d3.scaleLinear().domain([0, d3.max(norms) || 1]).range([height, 0]).nice();

    g.append("g").attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x).ticks(Math.min(norms.length, 20)).tickFormat((d) => `${d}`))
      .selectAll("text").attr("fill", "#8888aa").attr("font-size", 10);
    g.selectAll(".domain, .tick line").attr("stroke", "#333");

    g.append("g").call(d3.axisLeft(y).ticks(5))
      .selectAll("text").attr("fill", "#8888aa").attr("font-size", 10);
    g.selectAll(".domain, .tick line").attr("stroke", "#333");

    g.append("path").datum(norms)
      .attr("d", d3.line<number>().x((_, i) => x(i)).y((d) => y(d)))
      .attr("fill", "none").attr("stroke", "#4ecdc4").attr("stroke-width", 2);

    g.selectAll("circle").data(norms).join("circle")
      .attr("cx", (_, i) => x(i)).attr("cy", (d) => y(d))
      .attr("r", 3).attr("fill", "#4ecdc4");

    g.append("text").attr("x", width / 2).attr("y", height + 30)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#8888aa").text("Layer");
    g.append("text").attr("transform", "rotate(-90)").attr("x", -height / 2).attr("y", -45)
      .attr("text-anchor", "middle").attr("font-size", 11).attr("fill", "#8888aa").text("L2 Norm");
  }, [completeMsg]);

  if (!completeMsg?.norms) return <p style={{ color: "#666" }}>No residual norm data</p>;

  return (
    <div>
      <h3 style={{ fontSize: 13, color: "#a0a0c0", marginBottom: 8 }}>Residual Stream Norms - {result.sessionName}</h3>
      <svg ref={svgRef} />
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add testing/gui/frontend/src/components/visualizations/ResidualNorms.tsx
git commit -m "feat(gui): add ResidualNorms line chart visualization

D3 line chart showing L2 norm per layer. Complements logit lens
for spotting layers that significantly modify the residual stream."
```

---

## Task 12: Cleanup and verification

**Files:**
- Delete: `testing/gui/frontend/src/components/SessionSidebar.tsx`
- Delete: `testing/gui/frontend/src/components/ProbeConfig.tsx`
- Delete: `testing/gui/frontend/src/hooks/useSessions.ts`

- [ ] **Step 1: Delete old components**

```bash
rm testing/gui/frontend/src/components/SessionSidebar.tsx
rm testing/gui/frontend/src/components/ProbeConfig.tsx
rm testing/gui/frontend/src/hooks/useSessions.ts
```

- [ ] **Step 2: TypeScript check**

Run: `cd testing/gui/frontend && npx tsc --noEmit`
Expected: no errors

- [ ] **Step 3: Vite build**

Run: `cd testing/gui/frontend && npx vite build`
Expected: build succeeds with no errors

- [ ] **Step 4: Run backend tests**

Run: `PYTHONPATH=testing testing/.venv/bin/python -m pytest testing/gui/tests/ testing/tests/test_probe.py -v`
Expected: all tests pass (39 GUI + 26 probe = 65 total)

- [ ] **Step 5: Commit**

```bash
git add -A testing/gui/frontend/src/
git commit -m "chore(gui): remove old SessionSidebar, ProbeConfig, useSessions

Replaced by SessionsPanel, ProbePanel, and direct store usage."
```

- [ ] **Step 6: End-to-end manual verification checklist**

Start the app: `cd testing/gui && bash run.sh`

Open http://localhost:5173 in browser. Verify:

1. **Layout** — three tabs (Sessions, Probe, Intervene) at top of left panel, visualization area on right
2. **Sessions tab** — model combobox autocompletes cached models, can load model with spinner
3. **Sessions tab** — surgery operations show dynamic form controls (not JSON textarea), JSON fallback works
4. **Probe tab** — Session B dropdown disabled for influence/attention/residual-norms operations
5. **Probe tab** — run logit lens → heatmap appears in visualization area (fast cascade rendering)
6. **Probe tab** — run generate → tokens stream one-by-one with blinking cursor
7. **Probe tab** — run A/B logit lens → both connections stream, ABDiff renders
8. **Probe tab** — cancel during generation → stops cleanly
9. **Probe tab** — run influence → bar chart renders
10. **Probe tab** — run attention → small-multiples grid with sparklines
11. **Probe tab** — run residual-norms → line chart renders
12. **Intervene tab** — add interventions, set params, run → results in viz area
13. **Intervene tab** — capture logit lens checkbox → heatmap with red modified indicators
14. **Error handling** — stop backend, see "Backend offline" banner, restart, auto-reconnects
15. **Persistence** — switching tabs preserves all results in visualization area
