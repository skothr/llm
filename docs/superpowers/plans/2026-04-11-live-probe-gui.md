# Live Probe GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-based live experimentation dashboard for interactive model probing, surgery, and visualization on top of the existing `llm_surgeon` toolkit.

**Architecture:** FastAPI backend wrapping `llm_surgeon` with REST + WebSocket endpoints. React/Vite/TypeScript frontend with Zustand state management and D3.js visualizations. Models held in GPU memory via a SessionManager. WebSocket streams probe results layer-by-layer using the existing `on_layer` callback pattern.

**Tech Stack:** Python (FastAPI, uvicorn), TypeScript (React 18, Vite, Zustand, D3.js)

**Spec:** `docs/superpowers/specs/2026-04-11-live-probe-gui-design.md`

---

## File Structure

```
testing/gui/
  run.sh                          # Dev startup script (backend + frontend)
  backend/
    __init__.py
    app.py                        # FastAPI app, CORS, lifespan, static file mount
    sessions.py                   # SessionManager: load, clone, delete, VRAM tracking, undo snapshots
    hidden_state_cache.py         # LRU cache for extracted hidden states (500MB cap)
    routes/
      __init__.py                 # Router aggregation
      sessions.py                 # REST: session CRUD, surgery, undo, clone, discovery
      probes.py                   # WebSocket: logit-lens, generate, intervene
      inspect.py                  # REST: influence, attention, residual-norms
  tests/
    __init__.py
    conftest.py                   # Shared fixtures: tiny model, FastAPI test client
    test_sessions.py              # SessionManager unit tests
    test_session_routes.py        # Session REST endpoint tests
    test_surgery_routes.py        # Surgery + undo + discovery endpoint tests
    test_probe_ws.py              # WebSocket probe tests
    test_inspect_routes.py        # Inspect endpoint tests
    test_hidden_state_cache.py    # LRU cache tests
  frontend/
    index.html
    package.json
    tsconfig.json
    tsconfig.node.json
    vite.config.ts
    src/
      main.tsx                    # React entry point
      App.tsx                     # Root layout: sidebar + main area
      App.css                     # Global styles, layout grid
      types/
        api.ts                    # TypeScript types matching backend Pydantic schemas
      state/
        store.ts                  # Zustand store: sessions, probes, visualization slices
      hooks/
        useWebSocket.ts           # WebSocket connection manager with cancellation
        useSessions.ts            # REST API hooks for session management
      components/
        SessionSidebar.tsx        # Model loader, session list, surgery panel
        ProbeConfig.tsx           # Prompt input, operation selector, run/cancel
        VisualizationArea.tsx     # Tab container for result panels
        GenerationOutput.tsx      # Token stream display with probability popover
        visualizations/
          LogitLensHeatmap.tsx    # D3 heatmap: layer × position
          LayerInfluence.tsx      # D3 horizontal bar chart
          ABDiff.tsx              # Side-by-side heatmaps with diff overlay
```

---

## Prerequisites

The existing venv is at `testing/.venv` with torch, transformers, etc. The test suite uses `pytest`. The `llm_surgeon` package is imported as `from llm_surgeon import surgery, probe, inspect` etc. Node.js must be available for the frontend build.

**Tool rules (include in every subagent prompt):**
- Use Read (not cat), Edit (not Bash(sed/awk/cat)), Grep (not Bash(grep/rg/awk)), Glob (not find) for all file operations
- For git ops in other directories, use `git -C <path>` instead of `cd <path> && git ...`
- Avoid compound commands unless absolutely necessary
- Use `dangerouslyDisableSandbox: true` for any commands that need GPU/CUDA access

---

## Task 1: Backend Scaffolding + SessionManager

**Files:**
- Create: `testing/gui/__init__.py`
- Create: `testing/gui/backend/__init__.py`
- Create: `testing/gui/backend/app.py`
- Create: `testing/gui/backend/sessions.py`
- Create: `testing/gui/backend/routes/__init__.py`
- Create: `testing/gui/tests/__init__.py`
- Create: `testing/gui/tests/conftest.py`
- Create: `testing/gui/tests/test_sessions.py`

- [ ] **Step 1: Install backend dependencies**

Run: `source testing/.venv/bin/activate && pip install fastapi 'uvicorn[standard]'`

- [ ] **Step 2: Write SessionManager tests**

Create `testing/gui/tests/conftest.py`:

```python
import sys
from pathlib import Path
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

@pytest.fixture
def tiny_config():
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=128,
    )

@pytest.fixture
def tiny_model(tiny_config):
    model = LlamaForCausalLM(tiny_config)
    model.eval()
    return model

@pytest.fixture
def tiny_tokenizer(tiny_config):
    vocab = {f"tok{i}": i for i in range(tiny_config.vocab_size)}
    tok_model = models.WordLevel(vocab=vocab, unk_token="tok0")
    tokenizer = Tokenizer(tok_model)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    with TemporaryDirectory() as d:
        path = Path(d) / "tokenizer.json"
        tokenizer.save(str(path))
        yield PreTrainedTokenizerFast(tokenizer_file=str(path))
```

Create `testing/gui/tests/test_sessions.py`:

```python
import pytest
from gui.backend.sessions import SessionManager, SessionInfo

class TestSessionManager:
    def test_register_and_list(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("test-model", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        sessions = mgr.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].name == "test-model"
        assert sessions[0].model_id == "test/tiny"

    def test_get_session(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        info = mgr.get("s1")
        assert info.model is tiny_model
        assert info.tokenizer is tiny_tokenizer

    def test_get_missing_raises(self):
        mgr = SessionManager()
        with pytest.raises(KeyError, match="s1"):
            mgr.get("s1")

    def test_delete(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        mgr.delete("s1")
        assert mgr.list_sessions() == []

    def test_duplicate_name_raises(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        with pytest.raises(ValueError, match="already exists"):
            mgr.register("s1", tiny_model, tiny_tokenizer,
                          model_id="test/tiny", mode="eval")

    def test_validate_slug_rejects_invalid(self):
        mgr = SessionManager()
        with pytest.raises(ValueError, match="slug"):
            mgr.validate_name("Has Spaces")
        with pytest.raises(ValueError, match="slug"):
            mgr.validate_name("UPPERCASE")
        with pytest.raises(ValueError, match="slug"):
            mgr.validate_name("")

    def test_validate_slug_accepts_valid(self):
        mgr = SessionManager()
        mgr.validate_name("my-model-1")
        mgr.validate_name("baseline")

    def test_snapshot_and_undo(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        original_weight = tiny_model.model.layers[0].mlp.gate_proj.weight.clone()
        mgr.snapshot("s1")
        tiny_model.model.layers[0].mlp.gate_proj.weight.data.zero_()
        mgr.undo("s1")
        restored = tiny_model.model.layers[0].mlp.gate_proj.weight
        assert torch.allclose(restored, original_weight)

    def test_undo_without_snapshot_raises(self, tiny_model, tiny_tokenizer):
        mgr = SessionManager()
        mgr.register("s1", tiny_model, tiny_tokenizer,
                      model_id="test/tiny", mode="eval")
        with pytest.raises(ValueError, match="No snapshot"):
            mgr.undo("s1")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -m pytest testing/gui/tests/test_sessions.py -v`
Expected: ImportError — `gui.backend.sessions` does not exist yet.

- [ ] **Step 4: Implement SessionManager**

Create `testing/gui/__init__.py` (empty file).

Create `testing/gui/backend/__init__.py` (empty file).

Create `testing/gui/backend/sessions.py`:

```python
import re
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch

@dataclass
class SessionInfo:
    name: str
    model: object
    tokenizer: object
    model_id: str
    mode: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _snapshot: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)

    @property
    def has_snapshot(self) -> bool:
        return self._snapshot is not None

    @property
    def snapshot_size_mb(self) -> float:
        if self._snapshot is None:
            return 0.0
        return sum(t.nelement() * t.element_size() for t in self._snapshot.values()) / 1e6

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{0,63}$")

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SessionInfo] = {}

    def validate_name(self, name: str) -> None:
        if not _SLUG_RE.match(name):
            raise ValueError(
                f"Invalid slug: '{name}'. Must be lowercase alphanumeric + hyphens, 1-64 chars."
            )

    def register(self, name: str, model, tokenizer, *, model_id: str, mode: str) -> SessionInfo:
        self.validate_name(name)
        if name in self._sessions:
            raise ValueError(f"Session '{name}' already exists")
        info = SessionInfo(
            name=name, model=model, tokenizer=tokenizer,
            model_id=model_id, mode=mode,
        )
        self._sessions[name] = info
        return info

    def get(self, name: str) -> SessionInfo:
        if name not in self._sessions:
            raise KeyError(f"Session '{name}' not found")
        return self._sessions[name]

    def list_sessions(self) -> list:
        return list(self._sessions.values())

    def delete(self, name: str) -> None:
        info = self.get(name)
        del self._sessions[name]
        del info.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def snapshot(self, name: str) -> None:
        info = self.get(name)
        info._snapshot = {k: v.cpu().clone() for k, v in info.model.state_dict().items()}

    def undo(self, name: str) -> None:
        info = self.get(name)
        if info._snapshot is None:
            raise ValueError(f"No snapshot available for session '{name}'")
        device = next(info.model.parameters()).device
        state = {k: v.to(device) for k, v in info._snapshot.items()}
        info.model.load_state_dict(state)
        info._snapshot = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -m pytest testing/gui/tests/test_sessions.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 6: Create FastAPI app shell**

Create `testing/gui/backend/routes/__init__.py` (empty file).

Create `testing/gui/backend/app.py`:

```python
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .sessions import SessionManager

manager = SessionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for name in list(manager._sessions.keys()):
        manager.delete(name)

app = FastAPI(title="LLM Surgeon GUI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

dist_dir = Path(__file__).parent.parent / "frontend" / "dist"
if dist_dir.is_dir():
    from .routes import api_router, ws_router
    app.include_router(api_router)
    app.include_router(ws_router)
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="static")
else:
    from .routes import api_router, ws_router
    app.include_router(api_router)
    app.include_router(ws_router)
```

- [ ] **Step 7: Create route stubs**

Create `testing/gui/backend/routes/__init__.py`:

```python
from fastapi import APIRouter

api_router = APIRouter(prefix="/api")
ws_router = APIRouter(prefix="/ws")
```

- [ ] **Step 8: Verify app imports cleanly**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -c "from gui.backend.app import app; print(app.title)"`
Expected: `LLM Surgeon GUI`

- [ ] **Step 9: Commit**

```bash
git add testing/gui/
git commit -m "feat(gui): add backend scaffolding and SessionManager with tests"
```

---

## Task 2: Session REST Routes

**Files:**
- Create: `testing/gui/backend/routes/sessions.py`
- Modify: `testing/gui/backend/routes/__init__.py`
- Create: `testing/gui/tests/test_session_routes.py`

- [ ] **Step 1: Write session route tests**

Create `testing/gui/tests/test_session_routes.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from gui.backend.app import app, manager

@pytest.fixture(autouse=True)
def clear_sessions():
    manager._sessions.clear()
    yield
    manager._sessions.clear()

@pytest.fixture
def register_tiny(tiny_model, tiny_tokenizer):
    def _register(name="baseline"):
        manager.register(name, tiny_model, tiny_tokenizer,
                         model_id="test/tiny", mode="eval")
    return _register

@pytest.mark.asyncio
async def test_list_sessions_empty():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/sessions")
    assert resp.status_code == 200
    assert resp.json() == []

@pytest.mark.asyncio
async def test_list_sessions_with_model(register_tiny):
    register_tiny("baseline")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/sessions")
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "baseline"
    assert data[0]["model_id"] == "test/tiny"
    assert "num_layers" in data[0]

@pytest.mark.asyncio
async def test_get_session_info(register_tiny):
    register_tiny("baseline")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/sessions/baseline/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_layers"] == 4
    assert "num_heads" in data
    assert "hidden_size" in data

@pytest.mark.asyncio
async def test_get_session_info_missing():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/sessions/nonexistent/info")
    assert resp.status_code == 404

@pytest.mark.asyncio
async def test_delete_session(register_tiny):
    register_tiny("baseline")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/sessions/baseline")
    assert resp.status_code == 200
    assert manager.list_sessions() == []

@pytest.mark.asyncio
async def test_delete_session_missing():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/sessions/nonexistent")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && pip install httpx pytest-asyncio && python -m pytest testing/gui/tests/test_session_routes.py -v`
Expected: FAIL — routes not implemented, 404 for all endpoints.

- [ ] **Step 3: Implement session routes**

Create `testing/gui/backend/routes/sessions.py`:

```python
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
import re

from ..sessions import SessionManager

router = APIRouter(tags=["sessions"])

def get_manager() -> SessionManager:
    from ..app import manager
    return manager

class SessionSummary(BaseModel):
    name: str
    model_id: str
    mode: str
    num_layers: int
    has_snapshot: bool
    snapshot_size_mb: float

class SessionInfoResponse(BaseModel):
    num_layers: int
    num_heads: int
    hidden_size: int
    total_params: int

class SurgeryRequest(BaseModel):
    operation: str
    params: dict

class SurgeryResponse(BaseModel):
    operations: list
    info: SessionInfoResponse

class CloneRequest(BaseModel):
    target_name: str

    @field_validator("target_name")
    @classmethod
    def validate_slug(cls, v):
        if not re.match(r"^[a-z0-9][a-z0-9\-]{0,63}$", v):
            raise ValueError("Must be a valid slug")
        return v

def _session_summary(info) -> dict:
    config = info.model.config
    return SessionSummary(
        name=info.name,
        model_id=info.model_id,
        mode=info.mode,
        num_layers=config.num_hidden_layers,
        has_snapshot=info.has_snapshot,
        snapshot_size_mb=info.snapshot_size_mb,
    ).model_dump()

def _session_info(info) -> dict:
    config = info.model.config
    total_params = sum(p.numel() for p in info.model.parameters())
    return SessionInfoResponse(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        hidden_size=config.hidden_size,
        total_params=total_params,
    ).model_dump()

@router.get("/sessions", response_model=List[dict])
async def list_sessions():
    mgr = get_manager()
    return [_session_summary(s) for s in mgr.list_sessions()]

@router.get("/sessions/{name}/info")
async def get_session_info(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    return _session_info(info)

@router.delete("/sessions/{name}")
async def delete_session(name: str):
    mgr = get_manager()
    try:
        mgr.delete(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    return {"deleted": name}

SURGERY_OPS = [
    {"name": "remove_layers", "params": {"layer_indices": {"type": "array", "items": "int"}}, "description": "Remove specific layers"},
    {"name": "keep_layers", "params": {"layer_indices": {"type": "array", "items": "int"}}, "description": "Keep only specified layers"},
    {"name": "zero_heads", "params": {"layer": "int", "heads": {"type": "array", "items": "int"}}, "description": "Zero out attention heads"},
    {"name": "scale_heads", "params": {"layer": "int", "heads": {"type": "array", "items": "int"}, "factor": "float"}, "description": "Scale attention head contributions"},
    {"name": "swap_layers", "params": {"i": "int", "j": "int"}, "description": "Swap two layers"},
    {"name": "duplicate_layer", "params": {"src": "int", "dst": "int"}, "description": "Deep-copy layer to new position"},
    {"name": "zero_mlp", "params": {"layer": "int"}, "description": "Zero out MLP contribution"},
    {"name": "zero_attention", "params": {"layer": "int"}, "description": "Zero out attention contribution"},
    {"name": "swap_heads", "params": {"layer": "int", "h1": "int", "h2": "int"}, "description": "Swap two attention heads"},
    {"name": "reorder_layers", "params": {"new_order": {"type": "array", "items": "int"}}, "description": "Rearrange layer order"},
]

@router.get("/surgery/operations")
async def surgery_operations():
    return SURGERY_OPS

@router.post("/sessions/{name}/surgery")
async def apply_surgery(name: str, req: SurgeryRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon import surgery

    op_map = {
        "remove_layers": lambda m, p: surgery.remove_layers(m, p["layer_indices"]),
        "keep_layers": lambda m, p: surgery.keep_layers(m, p["layer_indices"]),
        "zero_heads": lambda m, p: surgery.zero_heads(m, p["layer"], p["heads"]),
        "scale_heads": lambda m, p: surgery.scale_heads(m, p["layer"], p["heads"], p["factor"]),
        "swap_layers": lambda m, p: surgery.swap_layers(m, p["i"], p["j"]),
        "duplicate_layer": lambda m, p: surgery.duplicate_layer(m, p["src"], p["dst"]),
        "zero_mlp": lambda m, p: surgery.zero_mlp(m, p["layer"]),
        "zero_attention": lambda m, p: surgery.zero_attention(m, p["layer"]),
        "swap_heads": lambda m, p: surgery.swap_heads(m, p["layer"], p["h1"], p["h2"]),
        "reorder_layers": lambda m, p: surgery.reorder_layers(m, p["new_order"]),
    }

    if req.operation not in op_map:
        raise HTTPException(422, f"Unknown surgery operation: '{req.operation}'")

    mgr.snapshot(name)

    try:
        log = op_map[req.operation](info.model, req.params)
    except (IndexError, ValueError) as e:
        mgr.undo(name)
        raise HTTPException(422, str(e))

    return SurgeryResponse(
        operations=[{"operation": op.operation, "description": op.description} for op in log.ops],
        info=SessionInfoResponse(**_session_info(info)),
    ).model_dump()

@router.post("/sessions/{name}/surgery/undo")
async def undo_surgery(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    try:
        mgr.undo(name)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"undone": name, "info": _session_info(info)}

@router.post("/sessions/{name}/clone")
async def clone_session(name: str, req: CloneRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    import copy
    state = {k: v.cpu().clone() for k, v in info.model.state_dict().items()}
    cloned_model = type(info.model)(info.model.config)
    device = next(info.model.parameters()).device
    cloned_model.load_state_dict(state)
    cloned_model.to(device)
    cloned_model.eval()

    mgr.register(req.target_name, cloned_model, info.tokenizer,
                 model_id=info.model_id, mode=info.mode)
    return _session_summary(mgr.get(req.target_name))
```

- [ ] **Step 4: Wire routes into the app**

Update `testing/gui/backend/routes/__init__.py`:

```python
from fastapi import APIRouter
from .sessions import router as sessions_router

api_router = APIRouter(prefix="/api")
api_router.include_router(sessions_router)

ws_router = APIRouter(prefix="/ws")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -m pytest testing/gui/tests/test_session_routes.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 6: Write surgery route tests**

Create `testing/gui/tests/test_surgery_routes.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from gui.backend.app import app, manager

@pytest.fixture(autouse=True)
def clear_sessions():
    manager._sessions.clear()
    yield
    manager._sessions.clear()

@pytest.fixture
def register_tiny(tiny_model, tiny_tokenizer):
    def _register(name="baseline"):
        manager.register(name, tiny_model, tiny_tokenizer,
                         model_id="test/tiny", mode="eval")
    return _register

@pytest.mark.asyncio
async def test_surgery_operations_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/surgery/operations")
    assert resp.status_code == 200
    ops = resp.json()
    names = [o["name"] for o in ops]
    assert "remove_layers" in names
    assert "zero_heads" in names

@pytest.mark.asyncio
async def test_apply_surgery_remove_layer(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions/test-model/surgery", json={
            "operation": "remove_layers",
            "params": {"layer_indices": [0]}
        })
    assert resp.status_code == 200
    data = resp.json()
    assert data["info"]["num_layers"] == 3

@pytest.mark.asyncio
async def test_surgery_creates_snapshot(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/test-model/surgery", json={
            "operation": "zero_mlp",
            "params": {"layer": 0}
        })
        resp = await client.get("/api/sessions")
    data = resp.json()
    assert data[0]["has_snapshot"] is True

@pytest.mark.asyncio
async def test_surgery_undo(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/test-model/surgery", json={
            "operation": "remove_layers",
            "params": {"layer_indices": [0]}
        })
        resp = await client.post("/api/sessions/test-model/surgery/undo")
    assert resp.status_code == 200
    assert resp.json()["info"]["num_layers"] == 4

@pytest.mark.asyncio
async def test_surgery_undo_no_snapshot(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions/test-model/surgery/undo")
    assert resp.status_code == 409

@pytest.mark.asyncio
async def test_surgery_unknown_op(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions/test-model/surgery", json={
            "operation": "nonexistent",
            "params": {}
        })
    assert resp.status_code == 422

@pytest.mark.asyncio
async def test_clone_session(register_tiny):
    register_tiny("baseline")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions/baseline/clone", json={
            "target_name": "variant"
        })
    assert resp.status_code == 200
    assert resp.json()["name"] == "variant"
    assert len(manager.list_sessions()) == 2
```

- [ ] **Step 7: Run surgery tests**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -m pytest testing/gui/tests/test_surgery_routes.py -v`
Expected: All 7 tests PASS (routes already implemented in Step 3).

- [ ] **Step 8: Commit**

```bash
git add testing/gui/backend/routes/ testing/gui/tests/
git commit -m "feat(gui): add session REST routes with surgery, undo, clone, and discovery"
```

---

## Task 3: WebSocket Infrastructure + Logit Lens Endpoint

**Files:**
- Create: `testing/gui/backend/routes/probes.py`
- Modify: `testing/gui/backend/routes/__init__.py`
- Create: `testing/gui/tests/test_probe_ws.py`

- [ ] **Step 1: Write logit lens WebSocket tests**

Create `testing/gui/tests/test_probe_ws.py`:

```python
import json
import pytest
from httpx import AsyncClient, ASGITransport
from httpx_ws import aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport
from gui.backend.app import app, manager

@pytest.fixture(autouse=True)
def clear_sessions():
    manager._sessions.clear()
    yield
    manager._sessions.clear()

@pytest.fixture
def register_tiny(tiny_model, tiny_tokenizer):
    def _register(name="baseline"):
        manager.register(name, tiny_model, tiny_tokenizer,
                         model_id="test/tiny", mode="eval")
    return _register

@pytest.mark.asyncio
async def test_logit_lens_streams_layers(register_tiny):
    register_tiny("baseline")
    async with aconnect_ws(
        "http://test/ws/sessions/baseline/logit-lens",
        transport=ASGIWebSocketTransport(app),
    ) as ws:
        await ws.send_text(json.dumps({"prompt": "tok1 tok2 tok3", "top_k": 5}))
        messages = []
        while True:
            msg = json.loads(await ws.receive_text())
            messages.append(msg)
            if msg["type"] in ("complete", "error"):
                break
    data_msgs = [m for m in messages if m["type"] == "data"]
    assert len(data_msgs) > 0
    assert all("layer" in m for m in data_msgs)
    assert all("sublayer" in m for m in data_msgs)
    assert messages[-1]["type"] == "complete"

@pytest.mark.asyncio
async def test_logit_lens_missing_session():
    async with aconnect_ws(
        "http://test/ws/sessions/nonexistent/logit-lens",
        transport=ASGIWebSocketTransport(app),
    ) as ws:
        await ws.send_text(json.dumps({"prompt": "tok1", "top_k": 5}))
        msg = json.loads(await ws.receive_text())
    assert msg["type"] == "error"
    assert "not found" in msg["message"]
```

- [ ] **Step 2: Install httpx-ws and run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && pip install httpx-ws && python -m pytest testing/gui/tests/test_probe_ws.py -v`
Expected: FAIL — WebSocket route not implemented.

- [ ] **Step 3: Implement probe WebSocket routes**

Create `testing/gui/backend/routes/probes.py`:

```python
import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..sessions import SessionManager

router = APIRouter(tags=["probes"])

def get_manager() -> SessionManager:
    from ..app import manager
    return manager

async def _send_json(ws: WebSocket, data: dict) -> bool:
    try:
        await ws.send_text(json.dumps(data, default=str))
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False

@router.websocket("/sessions/{name}/logit-lens")
async def logit_lens_ws(ws: WebSocket, name: str):
    await ws.accept()
    mgr = get_manager()

    try:
        info = mgr.get(name)
    except KeyError:
        await _send_json(ws, {"type": "error", "message": f"Session '{name}' not found"})
        await ws.close()
        return

    raw = await ws.receive_text()
    config = json.loads(raw)
    prompt = config["prompt"]
    top_k = config.get("top_k", 10)

    cancelled = asyncio.Event()

    async def check_cancel():
        try:
            while not cancelled.is_set():
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.1)
                data = json.loads(msg)
                if data.get("type") == "cancel":
                    cancelled.set()
        except (asyncio.TimeoutError, WebSocketDisconnect, RuntimeError):
            pass

    cancel_task = asyncio.create_task(check_cancel())
    connected = True

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

    from llm_surgeon import probe

    collected_layers = []

    def on_layer(layer_idx, sublayer, data):
        nonlocal connected
        if cancelled.is_set() or not connected:
            return

        predictions = data.get("top_k", [])
        serializable_preds = []
        for pos_preds in predictions:
            serializable_preds.append([
                {"token": p["token"], "prob": float(p["prob"])}
                for p in (pos_preds if isinstance(pos_preds, list) else [pos_preds])
            ])

        msg = {
            "type": "data",
            "layer": layer_idx,
            "sublayer": sublayer,
            "predictions": serializable_preds,
        }
        collected_layers.append(msg)

    try:
        async with info.lock:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: probe.logit_lens(
                    info.model, info.tokenizer, prompt,
                    top_k=top_k, on_layer=on_layer,
                ),
            )

        for msg in collected_layers:
            if not await _send_json(ws, msg):
                connected = False
                break

        if connected and not cancelled.is_set():
            summary = {
                "prompt_tokens": result.prompt_tokens,
                "num_layers": len(result.predictions),
            }
            await _send_json(ws, {"type": "complete", "summary": summary})
        elif cancelled.is_set():
            await _send_json(ws, {"type": "cancelled"})

    except Exception as e:
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        cancel_task.cancel()
        try:
            await ws.close()
        except RuntimeError:
            pass
```

- [ ] **Step 4: Wire probe routes into the app**

Update `testing/gui/backend/routes/__init__.py`:

```python
from fastapi import APIRouter
from .sessions import router as sessions_router
from .probes import router as probes_router

api_router = APIRouter(prefix="/api")
api_router.include_router(sessions_router)

ws_router = APIRouter(prefix="/ws")
ws_router.include_router(probes_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -m pytest testing/gui/tests/test_probe_ws.py -v`
Expected: All 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add testing/gui/backend/routes/ testing/gui/tests/test_probe_ws.py
git commit -m "feat(gui): add logit lens WebSocket endpoint with streaming and cancellation"
```

---

## Task 4: Generate + Intervene WebSocket Endpoints

**Files:**
- Modify: `testing/gui/backend/routes/probes.py`
- Create: `testing/gui/backend/hidden_state_cache.py`
- Create: `testing/gui/tests/test_hidden_state_cache.py`
- Modify: `testing/gui/tests/test_probe_ws.py`

- [ ] **Step 1: Write hidden state cache tests**

Create `testing/gui/tests/test_hidden_state_cache.py`:

```python
import torch
import pytest
from gui.backend.hidden_state_cache import HiddenStateCache

class TestHiddenStateCache:
    def test_put_and_get(self):
        cache = HiddenStateCache(max_bytes=100_000_000)
        tensor = torch.randn(10, 32)
        cache.put("session1", "prompt_hash_1", {"key": tensor})
        result = cache.get("session1", "prompt_hash_1")
        assert result is not None
        assert torch.equal(result["key"], tensor)

    def test_get_missing_returns_none(self):
        cache = HiddenStateCache(max_bytes=100_000_000)
        assert cache.get("s1", "h1") is None

    def test_eviction_on_size_limit(self):
        cache = HiddenStateCache(max_bytes=5000)
        t1 = torch.randn(10, 32)
        t2 = torch.randn(10, 32)
        cache.put("s1", "h1", {"k": t1})
        cache.put("s1", "h2", {"k": t2})
        assert cache.get("s1", "h1") is None
        assert cache.get("s1", "h2") is not None

    def test_invalidate_session(self):
        cache = HiddenStateCache(max_bytes=100_000_000)
        cache.put("s1", "h1", {"k": torch.randn(10, 32)})
        cache.put("s2", "h2", {"k": torch.randn(10, 32)})
        cache.invalidate_session("s1")
        assert cache.get("s1", "h1") is None
        assert cache.get("s2", "h2") is not None

    def test_lru_ordering(self):
        cache = HiddenStateCache(max_bytes=4000)
        t1 = torch.randn(5, 32)
        t2 = torch.randn(5, 32)
        t3 = torch.randn(5, 32)
        cache.put("s1", "h1", {"k": t1})
        cache.put("s1", "h2", {"k": t2})
        cache.get("s1", "h1")
        cache.put("s1", "h3", {"k": t3})
        assert cache.get("s1", "h1") is not None
        assert cache.get("s1", "h2") is None
```

- [ ] **Step 2: Run cache tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -m pytest testing/gui/tests/test_hidden_state_cache.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement hidden state cache**

Create `testing/gui/backend/hidden_state_cache.py`:

```python
from collections import OrderedDict
from typing import Dict, Optional, Tuple
import torch

class HiddenStateCache:
    def __init__(self, max_bytes: int = 500_000_000):
        self._max_bytes = max_bytes
        self._cache: OrderedDict[Tuple[str, str], Dict[str, torch.Tensor]] = OrderedDict()
        self._sizes: Dict[Tuple[str, str], int] = {}
        self._current_bytes = 0

    def _entry_size(self, data: Dict[str, torch.Tensor]) -> int:
        return sum(t.nelement() * t.element_size() for t in data.values())

    def put(self, session: str, prompt_hash: str, data: Dict[str, torch.Tensor]) -> None:
        key = (session, prompt_hash)
        if key in self._cache:
            self._current_bytes -= self._sizes[key]
            del self._cache[key]
            del self._sizes[key]

        size = self._entry_size(data)
        while self._current_bytes + size > self._max_bytes and self._cache:
            evict_key, _ = self._cache.popitem(last=False)
            self._current_bytes -= self._sizes.pop(evict_key)

        self._cache[key] = data
        self._sizes[key] = size
        self._current_bytes += size

    def get(self, session: str, prompt_hash: str) -> Optional[Dict[str, torch.Tensor]]:
        key = (session, prompt_hash)
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def invalidate_session(self, session: str) -> None:
        keys_to_remove = [k for k in self._cache if k[0] == session]
        for key in keys_to_remove:
            del self._cache[key]
            self._current_bytes -= self._sizes.pop(key)
```

- [ ] **Step 4: Run cache tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -m pytest testing/gui/tests/test_hidden_state_cache.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Write generate WebSocket test**

Add to `testing/gui/tests/test_probe_ws.py`:

```python
@pytest.mark.asyncio
async def test_generate_streams_tokens(register_tiny):
    register_tiny("baseline")
    async with aconnect_ws(
        "http://test/ws/sessions/baseline/generate",
        transport=ASGIWebSocketTransport(app),
    ) as ws:
        await ws.send_text(json.dumps({
            "prompt": "tok1 tok2",
            "max_tokens": 5,
            "temperature": 0.0,
            "prob_top_k": 3,
        }))
        messages = []
        while True:
            msg = json.loads(await ws.receive_text())
            messages.append(msg)
            if msg["type"] in ("complete", "error"):
                break
    data_msgs = [m for m in messages if m["type"] == "data"]
    assert len(data_msgs) > 0
    assert all("token" in m for m in data_msgs)
    assert all("top_k" in m for m in data_msgs)

@pytest.mark.asyncio
async def test_intervene_streams_results(register_tiny):
    register_tiny("baseline")
    async with aconnect_ws(
        "http://test/ws/sessions/baseline/intervene",
        transport=ASGIWebSocketTransport(app),
    ) as ws:
        await ws.send_text(json.dumps({
            "prompt": "tok1 tok2 tok3",
            "interventions": [
                {"layer": 0, "sublayer": "ffn", "op": "scale", "params": {"factor": 0.5}}
            ],
            "capture_logit_lens": True,
        }))
        messages = []
        while True:
            msg = json.loads(await ws.receive_text())
            messages.append(msg)
            if msg["type"] in ("complete", "error"):
                break
    assert messages[-1]["type"] == "complete"
```

- [ ] **Step 6: Add generate and intervene endpoints**

Add to `testing/gui/backend/routes/probes.py`:

```python
import hashlib
import torch
from ..hidden_state_cache import HiddenStateCache

_hs_cache = HiddenStateCache(max_bytes=500_000_000)

@router.websocket("/sessions/{name}/generate")
async def generate_ws(ws: WebSocket, name: str):
    await ws.accept()
    mgr = get_manager()

    try:
        info = mgr.get(name)
    except KeyError:
        await _send_json(ws, {"type": "error", "message": f"Session '{name}' not found"})
        await ws.close()
        return

    raw = await ws.receive_text()
    config = json.loads(raw)
    prompt = config["prompt"]
    max_tokens = config.get("max_tokens", 256)
    temperature = config.get("temperature", 1.0)
    top_k_sampling = config.get("top_k", 50)
    top_p = config.get("top_p", 1.0)
    repetition_penalty = config.get("repetition_penalty", 1.0)
    prob_top_k = config.get("prob_top_k", 10)

    cancelled = asyncio.Event()
    connected = True
    generated_tokens = []

    try:
        async with info.lock:
            inputs = info.tokenizer(prompt, return_tensors="pt")
            device = next(info.model.parameters()).device
            input_ids = inputs["input_ids"].to(device)

            for step in range(max_tokens):
                if cancelled.is_set() or not connected:
                    break

                with torch.no_grad():
                    outputs = info.model(input_ids)
                    logits = outputs.logits[:, -1, :]

                if temperature > 0:
                    logits = logits / temperature

                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs[0], min(prob_top_k, probs.shape[-1]))

                if temperature == 0:
                    next_token = top_indices[0:1].unsqueeze(0)
                else:
                    next_token = torch.multinomial(probs, 1)

                token_str = info.tokenizer.decode(next_token[0])
                generated_tokens.append(token_str)

                top_k_list = [
                    {"token": info.tokenizer.decode(top_indices[i:i+1]), "prob": float(top_probs[i])}
                    for i in range(len(top_indices))
                ]

                msg = {
                    "type": "data",
                    "step": step,
                    "token": token_str,
                    "token_id": int(next_token[0, 0]),
                    "top_k": top_k_list,
                }
                if not await _send_json(ws, msg):
                    connected = False
                    break

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if next_token[0, 0] == info.tokenizer.eos_token_id:
                    break

        if connected and not cancelled.is_set():
            await _send_json(ws, {
                "type": "complete",
                "generated_text": "".join(generated_tokens),
                "num_tokens": len(generated_tokens),
            })
        elif cancelled.is_set():
            await _send_json(ws, {"type": "cancelled"})

    except Exception as e:
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass


def _resolve_op(op_name: str, params: dict, mgr, cache):
    from llm_surgeon.probe import ops, extract_hidden_states

    simple_ops = {
        "scale": ops.scale,
        "zero_dims": ops.zero_dims,
        "clamp": ops.clamp,
        "noise": ops.noise,
    }

    if op_name in simple_ops:
        return simple_ops[op_name](**params)

    if op_name in ("replace", "project_out"):
        source = params["source"]
        session_name = source["session"]
        source_info = mgr.get(session_name)
        prompt_hash = hashlib.sha256(source["prompt"].encode()).hexdigest()[:16]

        cached = cache.get(session_name, prompt_hash)
        if cached is None:
            hs = extract_hidden_states(
                source_info.model, source_info.tokenizer, source["prompt"],
                layers=[source["layer"]], sublayers=(source["sublayer"],),
            )
            cached = {}
            for key, tensor in hs.states.items():
                cached[f"{key[0]}_{key[1]}"] = tensor
            cache.put(session_name, prompt_hash, cached)

        state_key = f"{source['layer']}_{source['sublayer']}"
        tensor = cached[state_key]
        position = source["position"]
        vec = tensor[position]

        if op_name == "replace":
            return ops.replace(vec)
        else:
            return ops.project_out(vec)

    raise ValueError(f"Unknown op: '{op_name}'")


@router.websocket("/sessions/{name}/intervene")
async def intervene_ws(ws: WebSocket, name: str):
    await ws.accept()
    mgr = get_manager()

    try:
        info = mgr.get(name)
    except KeyError:
        await _send_json(ws, {"type": "error", "message": f"Session '{name}' not found"})
        await ws.close()
        return

    raw = await ws.receive_text()
    config = json.loads(raw)
    prompt = config["prompt"]
    capture_logit_lens = config.get("capture_logit_lens", False)

    from llm_surgeon.probe import Intervention, intervene

    try:
        interventions = []
        for spec in config["interventions"]:
            fn = _resolve_op(spec["op"], spec.get("params", {}), mgr, _hs_cache)
            interventions.append(Intervention(
                layer=spec["layer"],
                sublayer=spec["sublayer"],
                fn=fn,
            ))
    except (KeyError, ValueError) as e:
        await _send_json(ws, {"type": "error", "message": f"Invalid intervention: {e}"})
        await ws.close()
        return

    collected = []
    connected = True

    def on_layer(layer_idx, sublayer, data):
        nonlocal connected
        if not connected:
            return
        msg = {
            "type": "data",
            "layer": layer_idx,
            "sublayer": sublayer,
            "modified": data.get("modified", False),
        }
        if "top_k" in data:
            preds = []
            for pos_preds in data["top_k"]:
                preds.append([
                    {"token": p["token"], "prob": float(p["prob"])}
                    for p in (pos_preds if isinstance(pos_preds, list) else [pos_preds])
                ])
            msg["predictions"] = preds
        collected.append(msg)

    try:
        async with info.lock:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: intervene(
                    info.model, info.tokenizer, prompt,
                    interventions=interventions,
                    capture_logit_lens=capture_logit_lens,
                    on_layer=on_layer,
                ),
            )

        for msg in collected:
            if not await _send_json(ws, msg):
                connected = False
                break

        if connected:
            applied = [
                {"layer": iv["layer"], "sublayer": iv["sublayer"], "op": iv.get("op", "")}
                for iv in result.interventions_applied
            ]
            await _send_json(ws, {
                "type": "complete",
                "interventions_applied": applied,
            })

    except Exception as e:
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass
```

- [ ] **Step 7: Run all probe tests**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -m pytest testing/gui/tests/test_probe_ws.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add testing/gui/
git commit -m "feat(gui): add generate, intervene WebSocket endpoints and hidden state cache"
```

---

## Task 5: Inspect REST Routes

**Files:**
- Create: `testing/gui/backend/routes/inspect.py`
- Modify: `testing/gui/backend/routes/__init__.py`
- Create: `testing/gui/tests/test_inspect_routes.py`

- [ ] **Step 1: Write inspect route tests**

Create `testing/gui/tests/test_inspect_routes.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from gui.backend.app import app, manager

@pytest.fixture(autouse=True)
def clear_sessions():
    manager._sessions.clear()
    yield
    manager._sessions.clear()

@pytest.fixture
def register_tiny(tiny_model, tiny_tokenizer):
    def _register(name="baseline"):
        manager.register(name, tiny_model, tiny_tokenizer,
                         model_id="test/tiny", mode="eval")
    return _register

@pytest.mark.asyncio
async def test_influence(register_tiny):
    register_tiny("baseline")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions/baseline/inspect/influence", json={
            "prompts": ["tok1 tok2 tok3"]
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "scores" in data
    assert len(data["scores"]) == 4

@pytest.mark.asyncio
async def test_attention_entropy(register_tiny):
    register_tiny("baseline")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions/baseline/inspect/attention", json={
            "prompt": "tok1 tok2 tok3"
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "entropy" in data

@pytest.mark.asyncio
async def test_residual_norms(register_tiny):
    register_tiny("baseline")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions/baseline/inspect/residual-norms", json={
            "prompt": "tok1 tok2 tok3"
        })
    assert resp.status_code == 200
    data = resp.json()
    assert "norms" in data
    assert len(data["norms"]) == 5

@pytest.mark.asyncio
async def test_inspect_missing_session():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions/nonexistent/inspect/influence", json={
            "prompts": ["test"]
        })
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -m pytest testing/gui/tests/test_inspect_routes.py -v`
Expected: FAIL — 404 for all inspect endpoints.

- [ ] **Step 3: Implement inspect routes**

Create `testing/gui/backend/routes/inspect.py`:

```python
import asyncio
import sys
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from ..sessions import SessionManager

router = APIRouter(tags=["inspect"])

def get_manager() -> SessionManager:
    from ..app import manager
    return manager

class InfluenceRequest(BaseModel):
    prompts: List[str]

class AttentionRequest(BaseModel):
    prompt: str

class ResidualNormsRequest(BaseModel):
    prompt: str

@router.post("/sessions/{name}/inspect/influence")
async def inspect_influence(name: str, req: InfluenceRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    from llm_surgeon import inspect as insp

    scores = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: insp.block_influence(info.model, info.tokenizer, req.prompts),
    )
    return {"scores": {str(k): float(v) for k, v in scores.items()}}

@router.post("/sessions/{name}/inspect/attention")
async def inspect_attention(name: str, req: AttentionRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    from llm_surgeon import inspect as insp

    entropy = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: insp.attention_entropy(info.model, info.tokenizer, req.prompt),
    )
    return {"entropy": {str(k): v for k, v in entropy.items()}}

@router.post("/sessions/{name}/inspect/residual-norms")
async def inspect_residual_norms(name: str, req: ResidualNormsRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    from llm_surgeon import inspect as insp

    norms = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: insp.residual_stream_norms(info.model, info.tokenizer, req.prompt),
    )
    return {"norms": [float(n) for n in norms]}
```

- [ ] **Step 4: Wire inspect routes into app**

Update `testing/gui/backend/routes/__init__.py`:

```python
from fastapi import APIRouter
from .sessions import router as sessions_router
from .probes import router as probes_router
from .inspect import router as inspect_router

api_router = APIRouter(prefix="/api")
api_router.include_router(sessions_router)
api_router.include_router(inspect_router)

ws_router = APIRouter(prefix="/ws")
ws_router.include_router(probes_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -m pytest testing/gui/tests/test_inspect_routes.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 6: Run full backend test suite**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -m pytest testing/gui/tests/ -v`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add testing/gui/
git commit -m "feat(gui): add inspect REST routes (influence, attention, residual norms)"
```

---

## Task 6: Frontend Scaffolding + Types + Store

**Files:**
- Create: `testing/gui/frontend/index.html`
- Create: `testing/gui/frontend/package.json`
- Create: `testing/gui/frontend/tsconfig.json`
- Create: `testing/gui/frontend/tsconfig.node.json`
- Create: `testing/gui/frontend/vite.config.ts`
- Create: `testing/gui/frontend/src/main.tsx`
- Create: `testing/gui/frontend/src/App.tsx`
- Create: `testing/gui/frontend/src/App.css`
- Create: `testing/gui/frontend/src/types/api.ts`
- Create: `testing/gui/frontend/src/state/store.ts`

- [ ] **Step 1: Initialize frontend project**

Run: `mkdir -p testing/gui/frontend/src`

Create `testing/gui/frontend/package.json`:

```json
{
  "name": "llm-surgeon-gui",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "d3": "^7.9.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "zustand": "^4.5.0"
  },
  "devDependencies": {
    "@types/d3": "^7.4.3",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0"
  }
}
```

- [ ] **Step 2: Create Vite and TypeScript config**

Create `testing/gui/frontend/vite.config.ts`:

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": {
        target: "ws://127.0.0.1:8000",
        ws: true,
      },
    },
  },
});
```

Create `testing/gui/frontend/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": false,
    "noUnusedParameters": false
  },
  "include": ["src"]
}
```

Create `testing/gui/frontend/tsconfig.node.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2023"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
```

Create `testing/gui/frontend/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>LLM Surgeon</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 3: Create TypeScript types**

Create `testing/gui/frontend/src/types/api.ts`:

```typescript
export interface SessionSummary {
  name: string;
  model_id: string;
  mode: string;
  num_layers: number;
  has_snapshot: boolean;
  snapshot_size_mb: number;
}

export interface SessionInfo {
  num_layers: number;
  num_heads: number;
  hidden_size: number;
  total_params: number;
}

export interface SurgeryOperation {
  name: string;
  params: Record<string, { type?: string; items?: string } | string>;
  description: string;
}

export interface SurgeryRequest {
  operation: string;
  params: Record<string, unknown>;
}

export interface LogitLensData {
  type: "data";
  layer: number;
  sublayer: string;
  predictions: Array<Array<{ token: string; prob: number }>>;
}

export interface GenerateData {
  type: "data";
  step: number;
  token: string;
  token_id: number;
  top_k: Array<{ token: string; prob: number }>;
}

export interface InterveneData {
  type: "data";
  layer: number;
  sublayer: string;
  modified: boolean;
  predictions?: Array<Array<{ token: string; prob: number }>>;
}

export interface WsComplete {
  type: "complete";
  [key: string]: unknown;
}

export interface WsError {
  type: "error";
  message: string;
}

export interface WsCancelled {
  type: "cancelled";
}

export type WsMessage = LogitLensData | GenerateData | InterveneData | WsComplete | WsError | WsCancelled;

export interface InterventionSpec {
  layer: number;
  sublayer: "attn" | "ffn";
  op: string;
  params: Record<string, unknown>;
}

export type ProbeOperation = "logit-lens" | "influence" | "intervene" | "generate";

export interface ProbeResult {
  id: string;
  operation: ProbeOperation;
  sessionName: string;
  prompt: string;
  data: WsMessage[];
  timestamp: number;
}

export interface InfluenceResult {
  scores: Record<string, number>;
}

export interface AttentionResult {
  entropy: Record<string, number[]>;
}

export interface ResidualNormsResult {
  norms: number[];
}
```

- [ ] **Step 4: Create Zustand store**

Create `testing/gui/frontend/src/state/store.ts`:

```typescript
import { create } from "zustand";
import type {
  SessionSummary,
  SessionInfo,
  SurgeryOperation,
  ProbeOperation,
  ProbeResult,
  InterventionSpec,
} from "../types/api";

interface SessionsSlice {
  sessions: SessionSummary[];
  sessionInfo: Record<string, SessionInfo>;
  surgeryOps: SurgeryOperation[];
  fetchSessions: () => Promise<void>;
  fetchSessionInfo: (name: string) => Promise<void>;
  fetchSurgeryOps: () => Promise<void>;
  deleteSession: (name: string) => Promise<void>;
  applySurgery: (name: string, operation: string, params: Record<string, unknown>) => Promise<void>;
  undoSurgery: (name: string) => Promise<void>;
  cloneSession: (name: string, targetName: string) => Promise<void>;
}

interface ProbesSlice {
  prompt: string;
  operation: ProbeOperation;
  targetSession: string;
  targetSessionB: string | null;
  isRunning: boolean;
  results: ProbeResult[];
  setPrompt: (prompt: string) => void;
  setOperation: (op: ProbeOperation) => void;
  setTargetSession: (name: string) => void;
  setTargetSessionB: (name: string | null) => void;
  setRunning: (running: boolean) => void;
  addResult: (result: ProbeResult) => void;
  clearResults: () => void;
}

interface VisualizationSlice {
  activeResultId: string | null;
  setActiveResult: (id: string | null) => void;
}

type StoreState = SessionsSlice & ProbesSlice & VisualizationSlice;

export const useStore = create<StoreState>((set, get) => ({
  sessions: [],
  sessionInfo: {},
  surgeryOps: [],
  prompt: "",
  operation: "logit-lens",
  targetSession: "",
  targetSessionB: null,
  isRunning: false,
  results: [],
  activeResultId: null,

  fetchSessions: async () => {
    const resp = await fetch("/api/sessions");
    const data = await resp.json();
    set({ sessions: data });
  },

  fetchSessionInfo: async (name: string) => {
    const resp = await fetch(`/api/sessions/${name}/info`);
    const data = await resp.json();
    set((s) => ({ sessionInfo: { ...s.sessionInfo, [name]: data } }));
  },

  fetchSurgeryOps: async () => {
    const resp = await fetch("/api/surgery/operations");
    const data = await resp.json();
    set({ surgeryOps: data });
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
    if (!resp.ok) throw new Error((await resp.json()).detail);
    await get().fetchSessions();
    await get().fetchSessionInfo(name);
  },

  undoSurgery: async (name: string) => {
    const resp = await fetch(`/api/sessions/${name}/surgery/undo`, { method: "POST" });
    if (!resp.ok) throw new Error((await resp.json()).detail);
    await get().fetchSessions();
    await get().fetchSessionInfo(name);
  },

  cloneSession: async (name: string, targetName: string) => {
    const resp = await fetch(`/api/sessions/${name}/clone`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_name: targetName }),
    });
    if (!resp.ok) throw new Error((await resp.json()).detail);
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
}));
```

- [ ] **Step 5: Create App shell and entry point**

Create `testing/gui/frontend/src/main.tsx`:

```typescript
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./App.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

Create `testing/gui/frontend/src/App.tsx`:

```typescript
import { useEffect } from "react";
import { useStore } from "./state/store";

export default function App() {
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSurgeryOps = useStore((s) => s.fetchSurgeryOps);

  useEffect(() => {
    fetchSessions();
    fetchSurgeryOps();
  }, [fetchSessions, fetchSurgeryOps]);

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-top">
          <h2>Sessions</h2>
          <p>Session sidebar placeholder</p>
        </div>
        <div className="sidebar-bottom">
          <h2>Probe Config</h2>
          <p>Probe config placeholder</p>
        </div>
      </aside>
      <main className="main-area">
        <div className="visualization-area">
          <h2>Visualization</h2>
          <p>Visualization placeholder</p>
        </div>
        <div className="generation-output">
          <h2>Generation Output</h2>
          <p>Generation output placeholder</p>
        </div>
      </main>
    </div>
  );
}
```

Create `testing/gui/frontend/src/App.css`:

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

.sidebar {
  display: flex;
  flex-direction: column;
  background: #16213e;
  border-right: 1px solid #0f3460;
  overflow-y: auto;
}

.sidebar-top {
  flex: 1;
  padding: 16px;
  border-bottom: 1px solid #0f3460;
}

.sidebar-bottom {
  flex: 1;
  padding: 16px;
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
```

- [ ] **Step 6: Install npm dependencies and verify build**

Run: `cd /home/ai/ai-projects/llm/testing/gui/frontend && npm install && npx tsc --noEmit`
Expected: No TypeScript errors.

- [ ] **Step 7: Commit**

```bash
git add testing/gui/frontend/
git commit -m "feat(gui): scaffold React frontend with Vite, Zustand store, and TypeScript types"
```

---

## Task 7: useWebSocket Hook + Session Sidebar

**Files:**
- Create: `testing/gui/frontend/src/hooks/useWebSocket.ts`
- Create: `testing/gui/frontend/src/hooks/useSessions.ts`
- Create: `testing/gui/frontend/src/components/SessionSidebar.tsx`
- Modify: `testing/gui/frontend/src/App.tsx`

- [ ] **Step 1: Create useWebSocket hook**

Create `testing/gui/frontend/src/hooks/useWebSocket.ts`:

```typescript
import { useRef, useCallback } from "react";
import type { WsMessage } from "../types/api";

interface UseWebSocketOptions {
  onMessage: (msg: WsMessage) => void;
  onComplete: (msg: WsMessage) => void;
  onError: (msg: string) => void;
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(
    (path: string, config: Record<string, unknown>, handlers: UseWebSocketOptions) => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${window.location.host}${path}`);
      wsRef.current = ws;

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
        wsRef.current = null;
      };

      return ws;
    },
    []
  );

  const cancel = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "cancel" }));
    }
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  return { connect, cancel, disconnect };
}
```

- [ ] **Step 2: Create useSessions hook**

Create `testing/gui/frontend/src/hooks/useSessions.ts`:

```typescript
import { useStore } from "../state/store";

export function useSessions() {
  const sessions = useStore((s) => s.sessions);
  const sessionInfo = useStore((s) => s.sessionInfo);
  const surgeryOps = useStore((s) => s.surgeryOps);
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSessionInfo = useStore((s) => s.fetchSessionInfo);
  const deleteSession = useStore((s) => s.deleteSession);
  const applySurgery = useStore((s) => s.applySurgery);
  const undoSurgery = useStore((s) => s.undoSurgery);
  const cloneSession = useStore((s) => s.cloneSession);

  return {
    sessions,
    sessionInfo,
    surgeryOps,
    fetchSessions,
    fetchSessionInfo,
    deleteSession,
    applySurgery,
    undoSurgery,
    cloneSession,
  };
}
```

- [ ] **Step 3: Create SessionSidebar component**

Create `testing/gui/frontend/src/components/SessionSidebar.tsx`:

```typescript
import { useState } from "react";
import { useSessions } from "../hooks/useSessions";
import type { SessionSummary, SurgeryOperation } from "../types/api";

export function SessionSidebar() {
  const {
    sessions, sessionInfo, surgeryOps,
    fetchSessions, fetchSessionInfo,
    deleteSession, applySurgery, undoSurgery, cloneSession,
  } = useSessions();

  const [loadModelId, setLoadModelId] = useState("");
  const [loadName, setLoadName] = useState("");
  const [loadMode, setLoadMode] = useState<"inspect" | "eval">("inspect");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [selectedSession, setSelectedSession] = useState<string | null>(null);
  const [cloneName, setCloneName] = useState("");
  const [surgeryOp, setSurgeryOp] = useState("");
  const [surgeryParams, setSurgeryParams] = useState("{}");

  const handleLoad = async () => {
    setLoading(true);
    setError("");
    try {
      const resp = await fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: loadName, model_id: loadModelId, mode: loadMode }),
      });
      if (!resp.ok) {
        const detail = await resp.json();
        throw new Error(detail.detail || "Failed to load model");
      }
      await fetchSessions();
      setLoadModelId("");
      setLoadName("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleClone = async (name: string) => {
    if (!cloneName) return;
    try {
      await cloneSession(name, cloneName);
      setCloneName("");
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const handleSurgery = async (name: string) => {
    try {
      const params = JSON.parse(surgeryParams);
      await applySurgery(name, surgeryOp, params);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <>
      <div className="sidebar-top">
        <h2>Sessions</h2>

        <div style={{ marginBottom: 12 }}>
          <input
            placeholder="Model ID (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
            value={loadModelId}
            onChange={(e) => setLoadModelId(e.target.value)}
            style={{ marginBottom: 4 }}
          />
          <input
            placeholder="Session name (slug)"
            value={loadName}
            onChange={(e) => setLoadName(e.target.value)}
            style={{ marginBottom: 4 }}
          />
          <div style={{ display: "flex", gap: 4 }}>
            <select value={loadMode} onChange={(e) => setLoadMode(e.target.value as "inspect" | "eval")}>
              <option value="inspect">4-bit (inspect)</option>
              <option value="eval">fp16 (eval)</option>
            </select>
            <button onClick={handleLoad} disabled={loading || !loadModelId || !loadName}>
              {loading ? "Loading..." : "Load"}
            </button>
          </div>
        </div>

        {error && <div style={{ color: "#ff6b6b", fontSize: 12, marginBottom: 8 }}>{error}</div>}

        {sessions.map((s: SessionSummary) => (
          <div
            key={s.name}
            style={{
              padding: 8,
              marginBottom: 4,
              background: selectedSession === s.name ? "#1a5276" : "#0d1b2a",
              borderRadius: 4,
              cursor: "pointer",
            }}
            onClick={() => {
              setSelectedSession(s.name);
              fetchSessionInfo(s.name);
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <strong>{s.name}</strong>
              <button
                onClick={(e) => { e.stopPropagation(); deleteSession(s.name); }}
                style={{ padding: "2px 6px", fontSize: 11 }}
              >
                ✕
              </button>
            </div>
            <div style={{ fontSize: 11, color: "#8888aa" }}>
              {s.model_id} · {s.mode} · {s.num_layers}L
              {s.has_snapshot && ` · undo: ${s.snapshot_size_mb.toFixed(0)}MB`}
            </div>
          </div>
        ))}
      </div>

      <div className="sidebar-bottom">
        {selectedSession && (
          <>
            <h2>Surgery — {selectedSession}</h2>

            <div style={{ marginBottom: 8 }}>
              <input
                placeholder="Clone name"
                value={cloneName}
                onChange={(e) => setCloneName(e.target.value)}
                style={{ marginBottom: 4 }}
              />
              <button onClick={() => handleClone(selectedSession)} disabled={!cloneName}>
                Clone
              </button>
            </div>

            <div style={{ marginBottom: 8 }}>
              <select value={surgeryOp} onChange={(e) => setSurgeryOp(e.target.value)} style={{ marginBottom: 4 }}>
                <option value="">Select operation...</option>
                {surgeryOps.map((op: SurgeryOperation) => (
                  <option key={op.name} value={op.name}>{op.name}</option>
                ))}
              </select>
              <textarea
                placeholder='Params JSON, e.g. {"layer_indices": [0]}'
                value={surgeryParams}
                onChange={(e) => setSurgeryParams(e.target.value)}
                rows={2}
                style={{ marginBottom: 4 }}
              />
              <div style={{ display: "flex", gap: 4 }}>
                <button onClick={() => handleSurgery(selectedSession)} disabled={!surgeryOp}>
                  Apply
                </button>
                <button
                  onClick={() => undoSurgery(selectedSession)}
                  disabled={!sessions.find((s) => s.name === selectedSession)?.has_snapshot}
                >
                  Undo
                </button>
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
          </>
        )}
      </div>
    </>
  );
}
```

- [ ] **Step 4: Wire SessionSidebar into App**

Update `testing/gui/frontend/src/App.tsx`:

```typescript
import { useEffect } from "react";
import { useStore } from "./state/store";
import { SessionSidebar } from "./components/SessionSidebar";

export default function App() {
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSurgeryOps = useStore((s) => s.fetchSurgeryOps);

  useEffect(() => {
    fetchSessions();
    fetchSurgeryOps();
  }, [fetchSessions, fetchSurgeryOps]);

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <SessionSidebar />
      </aside>
      <main className="main-area">
        <div className="visualization-area">
          <h2>Visualization</h2>
          <p style={{ color: "#666" }}>Run a probe to see results here</p>
        </div>
        <div className="generation-output">
          <h2>Generation Output</h2>
          <p style={{ color: "#666" }}>Run generate to see token stream here</p>
        </div>
      </main>
    </div>
  );
}
```

- [ ] **Step 5: Verify TypeScript compiles**

Run: `cd /home/ai/ai-projects/llm/testing/gui/frontend && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add testing/gui/frontend/
git commit -m "feat(gui): add useWebSocket hook, useSessions hook, and SessionSidebar component"
```

---

## Task 8: ProbeConfig Component

**Files:**
- Create: `testing/gui/frontend/src/components/ProbeConfig.tsx`
- Modify: `testing/gui/frontend/src/App.tsx`

- [ ] **Step 1: Create ProbeConfig component**

Create `testing/gui/frontend/src/components/ProbeConfig.tsx`:

```typescript
import { useState } from "react";
import { useStore } from "../state/store";
import { useWebSocket } from "../hooks/useWebSocket";
import type { WsMessage, ProbeResult, ProbeOperation } from "../types/api";

export function ProbeConfig() {
  const {
    prompt, operation, targetSession, targetSessionB, isRunning,
    sessions,
    setPrompt, setOperation, setTargetSession, setTargetSessionB,
    setRunning, addResult,
  } = useStore();

  const { connect, cancel } = useWebSocket();
  const [topK, setTopK] = useState(10);
  const [maxTokens, setMaxTokens] = useState(64);
  const [temperature, setTemperature] = useState(0.0);
  const [error, setError] = useState("");

  const handleRun = () => {
    if (!targetSession || !prompt) return;
    setError("");
    setRunning(true);

    const resultData: WsMessage[] = [];
    const resultId = `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;

    const handlers = {
      onMessage: (msg: WsMessage) => {
        resultData.push(msg);
      },
      onComplete: (msg: WsMessage) => {
        resultData.push(msg);
        addResult({
          id: resultId,
          operation,
          sessionName: targetSession,
          prompt,
          data: resultData,
          timestamp: Date.now(),
        });
        setRunning(false);
      },
      onError: (message: string) => {
        setError(message);
        setRunning(false);
      },
    };

    if (operation === "logit-lens") {
      connect(`/ws/sessions/${targetSession}/logit-lens`, { prompt, top_k: topK }, handlers);
    } else if (operation === "generate") {
      connect(`/ws/sessions/${targetSession}/generate`, {
        prompt, max_tokens: maxTokens, temperature, prob_top_k: topK,
      }, handlers);
    } else if (operation === "intervene") {
      setError("Configure interventions in the surgery panel first");
      setRunning(false);
    } else if (operation === "influence") {
      setRunning(true);
      fetch(`/api/sessions/${targetSession}/inspect/influence`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompts: [prompt] }),
      })
        .then((r) => r.json())
        .then((data) => {
          addResult({
            id: resultId,
            operation: "influence",
            sessionName: targetSession,
            prompt,
            data: [{ type: "complete" as const, scores: data.scores }],
            timestamp: Date.now(),
          });
          setRunning(false);
        })
        .catch((e) => {
          setError((e as Error).message);
          setRunning(false);
        });
    }

    if (targetSessionB && (operation === "logit-lens" || operation === "generate")) {
      const resultIdB = `${resultId}-B`;
      const resultDataB: WsMessage[] = [];
      const handlersB = {
        onMessage: (msg: WsMessage) => { resultDataB.push(msg); },
        onComplete: (msg: WsMessage) => {
          resultDataB.push(msg);
          addResult({
            id: resultIdB,
            operation,
            sessionName: targetSessionB,
            prompt,
            data: resultDataB,
            timestamp: Date.now(),
          });
        },
        onError: (message: string) => { setError(message); },
      };

      if (operation === "logit-lens") {
        connect(`/ws/sessions/${targetSessionB}/logit-lens`, { prompt, top_k: topK }, handlersB);
      } else if (operation === "generate") {
        connect(`/ws/sessions/${targetSessionB}/generate`, {
          prompt, max_tokens: maxTokens, temperature, prob_top_k: topK,
        }, handlersB);
      }
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <h2>Probe Config</h2>

      <textarea
        placeholder="Prompt text..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        rows={3}
      />

      <select value={operation} onChange={(e) => setOperation(e.target.value as ProbeOperation)}>
        <option value="logit-lens">Logit Lens</option>
        <option value="influence">Layer Influence</option>
        <option value="generate">Generate</option>
        <option value="intervene">Intervene</option>
      </select>

      <div style={{ display: "flex", gap: 4 }}>
        <select value={targetSession} onChange={(e) => setTargetSession(e.target.value)}>
          <option value="">Session A...</option>
          {sessions.map((s) => (
            <option key={s.name} value={s.name}>{s.name}</option>
          ))}
        </select>
        <select value={targetSessionB ?? ""} onChange={(e) => setTargetSessionB(e.target.value || null)}>
          <option value="">Session B (A/B)...</option>
          {sessions.map((s) => (
            <option key={s.name} value={s.name}>{s.name}</option>
          ))}
        </select>
      </div>

      {(operation === "logit-lens" || operation === "generate") && (
        <div style={{ display: "flex", gap: 8, fontSize: 12 }}>
          <label>
            top_k: <input type="number" value={topK} onChange={(e) => setTopK(+e.target.value)} style={{ width: 50 }} />
          </label>
          {operation === "generate" && (
            <>
              <label>
                max: <input type="number" value={maxTokens} onChange={(e) => setMaxTokens(+e.target.value)} style={{ width: 50 }} />
              </label>
              <label>
                temp: <input type="number" step="0.1" value={temperature} onChange={(e) => setTemperature(+e.target.value)} style={{ width: 50 }} />
              </label>
            </>
          )}
        </div>
      )}

      <div style={{ display: "flex", gap: 4 }}>
        {!isRunning ? (
          <button onClick={handleRun} disabled={!targetSession || !prompt}>
            Run
          </button>
        ) : (
          <button onClick={cancel} style={{ background: "#6b2020" }}>
            Cancel
          </button>
        )}
      </div>

      {error && <div style={{ color: "#ff6b6b", fontSize: 12 }}>{error}</div>}
    </div>
  );
}
```

- [ ] **Step 2: Wire ProbeConfig into App**

Update `testing/gui/frontend/src/App.tsx`:

```typescript
import { useEffect } from "react";
import { useStore } from "./state/store";
import { SessionSidebar } from "./components/SessionSidebar";
import { ProbeConfig } from "./components/ProbeConfig";

export default function App() {
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSurgeryOps = useStore((s) => s.fetchSurgeryOps);

  useEffect(() => {
    fetchSessions();
    fetchSurgeryOps();
  }, [fetchSessions, fetchSurgeryOps]);

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <SessionSidebar />
        <div style={{ padding: 16, borderTop: "1px solid #0f3460" }}>
          <ProbeConfig />
        </div>
      </aside>
      <main className="main-area">
        <div className="visualization-area">
          <h2>Visualization</h2>
          <p style={{ color: "#666" }}>Run a probe to see results here</p>
        </div>
        <div className="generation-output">
          <h2>Generation Output</h2>
          <p style={{ color: "#666" }}>Run generate to see token stream here</p>
        </div>
      </main>
    </div>
  );
}
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd /home/ai/ai-projects/llm/testing/gui/frontend && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add testing/gui/frontend/src/
git commit -m "feat(gui): add ProbeConfig component with logit lens, influence, and generate support"
```

---

## Task 9: Logit Lens Heatmap Visualization

**Files:**
- Create: `testing/gui/frontend/src/components/visualizations/LogitLensHeatmap.tsx`
- Create: `testing/gui/frontend/src/components/VisualizationArea.tsx`
- Modify: `testing/gui/frontend/src/App.tsx`

- [ ] **Step 1: Create LogitLensHeatmap component**

Create `testing/gui/frontend/src/components/visualizations/LogitLensHeatmap.tsx`:

```typescript
import { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import type { LogitLensData, ProbeResult } from "../../types/api";

interface Props {
  result: ProbeResult;
}

export function LogitLensHeatmap({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    content: string;
  } | null>(null);

  const dataMessages = result.data.filter(
    (m): m is LogitLensData => m.type === "data" && "predictions" in m
  );

  useEffect(() => {
    if (!svgRef.current || dataMessages.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 30, right: 20, bottom: 40, left: 80 };
    const numPositions = dataMessages[0].predictions.length || 1;
    const numRows = dataMessages.length;

    const cellW = Math.max(30, Math.min(60, 600 / numPositions));
    const cellH = 20;
    const width = margin.left + numPositions * cellW + margin.right;
    const height = margin.top + numRows * cellH + margin.bottom;

    svg.attr("width", width).attr("height", height);

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    dataMessages.forEach((msg, rowIdx) => {
      const label = `L${msg.layer}.${msg.sublayer}`;

      g.append("text")
        .attr("x", -4)
        .attr("y", rowIdx * cellH + cellH / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", 10)
        .attr("fill", "#8888aa")
        .text(label);

      msg.predictions.forEach((posPreds, posIdx) => {
        const topPred = posPreds[0];
        if (!topPred) return;

        g.append("rect")
          .attr("x", posIdx * cellW)
          .attr("y", rowIdx * cellH)
          .attr("width", cellW - 1)
          .attr("height", cellH - 1)
          .attr("fill", colorScale(topPred.prob))
          .attr("rx", 2)
          .style("cursor", "pointer")
          .on("mouseenter", (event) => {
            const lines = posPreds
              .slice(0, 5)
              .map((p) => `${p.token}: ${(p.prob * 100).toFixed(1)}%`)
              .join("\n");
            setTooltip({
              x: event.pageX + 10,
              y: event.pageY - 10,
              content: `L${msg.layer}.${msg.sublayer} pos ${posIdx}\n${lines}`,
            });
          })
          .on("mouseleave", () => setTooltip(null));

        g.append("text")
          .attr("x", posIdx * cellW + cellW / 2)
          .attr("y", rowIdx * cellH + cellH / 2)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .attr("font-size", 9)
          .attr("fill", topPred.prob > 0.5 ? "#000" : "#fff")
          .style("pointer-events", "none")
          .text(topPred.token.length > 6 ? topPred.token.slice(0, 5) + "…" : topPred.token);
      });
    });

    // Position axis
    g.append("g")
      .attr("transform", `translate(0,${numRows * cellH + 4})`)
      .selectAll("text")
      .data(d3.range(numPositions))
      .join("text")
      .attr("x", (d) => d * cellW + cellW / 2)
      .attr("text-anchor", "middle")
      .attr("font-size", 10)
      .attr("fill", "#8888aa")
      .text((d) => `${d}`);

  }, [dataMessages]);

  return (
    <div style={{ position: "relative" }}>
      <h3 style={{ fontSize: 13, color: "#a0a0c0", marginBottom: 8 }}>
        Logit Lens — {result.sessionName} — "{result.prompt.slice(0, 40)}"
      </h3>
      <div style={{ overflowX: "auto" }}>
        <svg ref={svgRef} />
      </div>
      {tooltip && (
        <div
          style={{
            position: "fixed",
            left: tooltip.x,
            top: tooltip.y,
            background: "#16213e",
            border: "1px solid #1a5276",
            borderRadius: 4,
            padding: "6px 10px",
            fontSize: 12,
            whiteSpace: "pre",
            pointerEvents: "none",
            zIndex: 100,
          }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Create VisualizationArea component**

Create `testing/gui/frontend/src/components/VisualizationArea.tsx`:

```typescript
import { useStore } from "../state/store";
import { LogitLensHeatmap } from "./visualizations/LogitLensHeatmap";

export function VisualizationArea() {
  const results = useStore((s) => s.results);
  const activeResultId = useStore((s) => s.activeResultId);
  const setActiveResult = useStore((s) => s.setActiveResult);
  const clearResults = useStore((s) => s.clearResults);

  const activeResult = results.find((r) => r.id === activeResultId);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <h2>Visualization</h2>
        {results.length > 0 && (
          <button onClick={clearResults} style={{ fontSize: 11 }}>Clear All</button>
        )}
      </div>

      {results.length > 1 && (
        <div style={{ display: "flex", gap: 4, marginBottom: 8, flexWrap: "wrap" }}>
          {results.map((r) => (
            <button
              key={r.id}
              onClick={() => setActiveResult(r.id)}
              style={{
                fontSize: 11,
                padding: "2px 8px",
                background: r.id === activeResultId ? "#1a5276" : "#0d1b2a",
              }}
            >
              {r.operation} · {r.sessionName}
            </button>
          ))}
        </div>
      )}

      {activeResult ? (
        activeResult.operation === "logit-lens" ? (
          <LogitLensHeatmap result={activeResult} />
        ) : activeResult.operation === "influence" ? (
          <div>
            <h3 style={{ fontSize: 13, color: "#a0a0c0", marginBottom: 8 }}>
              Layer Influence — {activeResult.sessionName}
            </h3>
            <p style={{ color: "#666" }}>Layer influence visualization coming next</p>
          </div>
        ) : (
          <p style={{ color: "#666" }}>Visualization for {activeResult.operation} coming soon</p>
        )
      ) : (
        <p style={{ color: "#666" }}>Run a probe to see results here</p>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Wire VisualizationArea into App**

Update `testing/gui/frontend/src/App.tsx`:

```typescript
import { useEffect } from "react";
import { useStore } from "./state/store";
import { SessionSidebar } from "./components/SessionSidebar";
import { ProbeConfig } from "./components/ProbeConfig";
import { VisualizationArea } from "./components/VisualizationArea";

export default function App() {
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSurgeryOps = useStore((s) => s.fetchSurgeryOps);

  useEffect(() => {
    fetchSessions();
    fetchSurgeryOps();
  }, [fetchSessions, fetchSurgeryOps]);

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <SessionSidebar />
        <div style={{ padding: 16, borderTop: "1px solid #0f3460" }}>
          <ProbeConfig />
        </div>
      </aside>
      <main className="main-area">
        <div className="visualization-area">
          <VisualizationArea />
        </div>
        <div className="generation-output">
          <h2>Generation Output</h2>
          <p style={{ color: "#666" }}>Run generate to see token stream here</p>
        </div>
      </main>
    </div>
  );
}
```

- [ ] **Step 4: Verify TypeScript compiles**

Run: `cd /home/ai/ai-projects/llm/testing/gui/frontend && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add testing/gui/frontend/src/
git commit -m "feat(gui): add LogitLensHeatmap D3 visualization and VisualizationArea"
```

---

## Task 10: Layer Influence + Generation Output

**Files:**
- Create: `testing/gui/frontend/src/components/visualizations/LayerInfluence.tsx`
- Create: `testing/gui/frontend/src/components/GenerationOutput.tsx`
- Modify: `testing/gui/frontend/src/components/VisualizationArea.tsx`
- Modify: `testing/gui/frontend/src/App.tsx`

- [ ] **Step 1: Create LayerInfluence component**

Create `testing/gui/frontend/src/components/visualizations/LayerInfluence.tsx`:

```typescript
import { useRef, useEffect } from "react";
import * as d3 from "d3";
import type { ProbeResult } from "../../types/api";

interface Props {
  result: ProbeResult;
}

export function LayerInfluence({ result }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  const completeMsg = result.data.find((m) => m.type === "complete") as
    | { type: "complete"; scores: Record<string, number> }
    | undefined;

  useEffect(() => {
    if (!svgRef.current || !completeMsg?.scores) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const entries = Object.entries(completeMsg.scores)
      .map(([layer, score]) => ({ layer: parseInt(layer), score }))
      .sort((a, b) => a.layer - b.layer);

    const margin = { top: 20, right: 20, bottom: 30, left: 50 };
    const barHeight = 22;
    const width = 500;
    const height = margin.top + entries.length * barHeight + margin.bottom;

    svg.attr("width", width).attr("height", height);

    const x = d3.scaleLinear()
      .domain([0, d3.max(entries, (d) => d.score) || 1])
      .range([0, width - margin.left - margin.right]);

    const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

    const colorScale = d3.scaleSequential(d3.interpolateYlOrRd)
      .domain([0, d3.max(entries, (d) => d.score) || 1]);

    entries.forEach((d, i) => {
      g.append("rect")
        .attr("x", 0)
        .attr("y", i * barHeight)
        .attr("width", x(d.score))
        .attr("height", barHeight - 2)
        .attr("fill", colorScale(d.score))
        .attr("rx", 2);

      g.append("text")
        .attr("x", -4)
        .attr("y", i * barHeight + barHeight / 2)
        .attr("text-anchor", "end")
        .attr("dominant-baseline", "middle")
        .attr("font-size", 10)
        .attr("fill", "#8888aa")
        .text(`L${d.layer}`);

      g.append("text")
        .attr("x", x(d.score) + 4)
        .attr("y", i * barHeight + barHeight / 2)
        .attr("dominant-baseline", "middle")
        .attr("font-size", 10)
        .attr("fill", "#ccccdd")
        .text(d.score.toFixed(3));
    });

  }, [completeMsg]);

  if (!completeMsg?.scores) {
    return <p style={{ color: "#666" }}>No influence data</p>;
  }

  return (
    <div>
      <h3 style={{ fontSize: 13, color: "#a0a0c0", marginBottom: 8 }}>
        Layer Influence — {result.sessionName}
      </h3>
      <svg ref={svgRef} />
    </div>
  );
}
```

- [ ] **Step 2: Create GenerationOutput component**

Create `testing/gui/frontend/src/components/GenerationOutput.tsx`:

```typescript
import { useState } from "react";
import { useStore } from "../state/store";
import type { GenerateData, ProbeResult } from "../types/api";

export function GenerationOutput() {
  const results = useStore((s) => s.results);
  const [selectedToken, setSelectedToken] = useState<GenerateData | null>(null);

  const genResults = results.filter((r) => r.operation === "generate");
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
        {genResults.slice(0, 2).map((result) => (
          <GenerationPanel
            key={result.id}
            result={result}
            selectedToken={selectedToken}
            onSelectToken={setSelectedToken}
          />
        ))}
      </div>
    </div>
  );
}

function GenerationPanel({
  result,
  selectedToken,
  onSelectToken,
}: {
  result: ProbeResult;
  selectedToken: GenerateData | null;
  onSelectToken: (t: GenerateData | null) => void;
}) {
  const tokens = result.data.filter(
    (m): m is GenerateData => m.type === "data" && "token" in m && "step" in m
  );

  return (
    <div style={{ flex: 1 }}>
      <div style={{ fontSize: 12, color: "#8888aa", marginBottom: 4 }}>
        {result.sessionName} — "{result.prompt.slice(0, 30)}"
      </div>
      <div style={{ fontFamily: "monospace", fontSize: 14, lineHeight: 1.8 }}>
        <span style={{ color: "#6688aa" }}>{result.prompt}</span>
        {tokens.map((tok) => (
          <span
            key={tok.step}
            onClick={() => onSelectToken(selectedToken?.step === tok.step ? null : tok)}
            style={{
              cursor: "pointer",
              background:
                selectedToken?.step === tok.step ? "#1a5276" : "transparent",
              borderRadius: 2,
              padding: "0 1px",
            }}
          >
            {tok.token}
          </span>
        ))}
      </div>

      {selectedToken && (
        <div
          style={{
            marginTop: 8,
            padding: 8,
            background: "#0d1b2a",
            borderRadius: 4,
            fontSize: 12,
          }}
        >
          <div style={{ color: "#a0a0c0", marginBottom: 4 }}>
            Step {selectedToken.step} — top alternatives:
          </div>
          {selectedToken.top_k.map((alt, i) => (
            <div key={i} style={{ display: "flex", gap: 8 }}>
              <span style={{ color: i === 0 ? "#4ecdc4" : "#888" }}>
                {alt.token}
              </span>
              <span style={{ color: "#666" }}>
                {(alt.prob * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Update VisualizationArea to include LayerInfluence**

Update `testing/gui/frontend/src/components/VisualizationArea.tsx`:

```typescript
import { useStore } from "../state/store";
import { LogitLensHeatmap } from "./visualizations/LogitLensHeatmap";
import { LayerInfluence } from "./visualizations/LayerInfluence";

export function VisualizationArea() {
  const results = useStore((s) => s.results);
  const activeResultId = useStore((s) => s.activeResultId);
  const setActiveResult = useStore((s) => s.setActiveResult);
  const clearResults = useStore((s) => s.clearResults);

  const vizResults = results.filter((r) => r.operation !== "generate");
  const activeResult = vizResults.find((r) => r.id === activeResultId);

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <h2>Visualization</h2>
        {vizResults.length > 0 && (
          <button onClick={clearResults} style={{ fontSize: 11 }}>Clear All</button>
        )}
      </div>

      {vizResults.length > 1 && (
        <div style={{ display: "flex", gap: 4, marginBottom: 8, flexWrap: "wrap" }}>
          {vizResults.map((r) => (
            <button
              key={r.id}
              onClick={() => setActiveResult(r.id)}
              style={{
                fontSize: 11,
                padding: "2px 8px",
                background: r.id === activeResultId ? "#1a5276" : "#0d1b2a",
              }}
            >
              {r.operation} · {r.sessionName}
            </button>
          ))}
        </div>
      )}

      {activeResult ? (
        activeResult.operation === "logit-lens" ? (
          <LogitLensHeatmap result={activeResult} />
        ) : activeResult.operation === "influence" ? (
          <LayerInfluence result={activeResult} />
        ) : (
          <p style={{ color: "#666" }}>Visualization for {activeResult.operation} coming soon</p>
        )
      ) : (
        <p style={{ color: "#666" }}>Run a probe to see results here</p>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Wire GenerationOutput into App**

Update `testing/gui/frontend/src/App.tsx`:

```typescript
import { useEffect } from "react";
import { useStore } from "./state/store";
import { SessionSidebar } from "./components/SessionSidebar";
import { ProbeConfig } from "./components/ProbeConfig";
import { VisualizationArea } from "./components/VisualizationArea";
import { GenerationOutput } from "./components/GenerationOutput";

export default function App() {
  const fetchSessions = useStore((s) => s.fetchSessions);
  const fetchSurgeryOps = useStore((s) => s.fetchSurgeryOps);

  useEffect(() => {
    fetchSessions();
    fetchSurgeryOps();
  }, [fetchSessions, fetchSurgeryOps]);

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <SessionSidebar />
        <div style={{ padding: 16, borderTop: "1px solid #0f3460" }}>
          <ProbeConfig />
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

- [ ] **Step 5: Verify TypeScript compiles**

Run: `cd /home/ai/ai-projects/llm/testing/gui/frontend && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add testing/gui/frontend/src/
git commit -m "feat(gui): add LayerInfluence bars, GenerationOutput with token probability popover"
```

---

## Task 11: A/B Diff Visualization

**Files:**
- Create: `testing/gui/frontend/src/components/visualizations/ABDiff.tsx`
- Modify: `testing/gui/frontend/src/components/VisualizationArea.tsx`

- [ ] **Step 1: Create ABDiff component**

Create `testing/gui/frontend/src/components/visualizations/ABDiff.tsx`:

```typescript
import { useRef, useEffect, useState } from "react";
import * as d3 from "d3";
import type { LogitLensData, ProbeResult } from "../../types/api";

interface Props {
  resultA: ProbeResult;
  resultB: ProbeResult;
}

export function ABDiff({ resultA, resultB }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [showDiff, setShowDiff] = useState(false);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    content: string;
  } | null>(null);

  const dataA = resultA.data.filter(
    (m): m is LogitLensData => m.type === "data" && "predictions" in m
  );
  const dataB = resultB.data.filter(
    (m): m is LogitLensData => m.type === "data" && "predictions" in m
  );

  useEffect(() => {
    if (!svgRef.current || dataA.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const numPositions = dataA[0]?.predictions.length || 1;
    const numRows = Math.max(dataA.length, dataB.length);
    const cellW = Math.max(24, Math.min(40, 400 / numPositions));
    const cellH = 18;
    const margin = { top: 30, right: 20, bottom: 30, left: 70 };
    const panelWidth = margin.left + numPositions * cellW + margin.right;
    const gap = 40;
    const totalWidth = showDiff ? panelWidth * 3 + gap * 2 : panelWidth * 2 + gap;
    const height = margin.top + numRows * cellH + margin.bottom;

    svg.attr("width", totalWidth).attr("height", height);

    const colorScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);
    const diffScale = d3.scaleDiverging(d3.interpolateRdBu).domain([-1, 0, 1]);

    function drawPanel(
      data: LogitLensData[],
      offsetX: number,
      label: string
    ) {
      const g = svg.append("g").attr("transform", `translate(${offsetX + margin.left},${margin.top})`);

      svg.append("text")
        .attr("x", offsetX + margin.left + (numPositions * cellW) / 2)
        .attr("y", 16)
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .attr("fill", "#a0a0c0")
        .text(label);

      data.forEach((msg, rowIdx) => {
        g.append("text")
          .attr("x", -4)
          .attr("y", rowIdx * cellH + cellH / 2)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("font-size", 9)
          .attr("fill", "#8888aa")
          .text(`L${msg.layer}.${msg.sublayer}`);

        msg.predictions.forEach((posPreds, posIdx) => {
          const topPred = posPreds[0];
          if (!topPred) return;
          g.append("rect")
            .attr("x", posIdx * cellW)
            .attr("y", rowIdx * cellH)
            .attr("width", cellW - 1)
            .attr("height", cellH - 1)
            .attr("fill", colorScale(topPred.prob))
            .attr("rx", 2);
        });
      });
    }

    drawPanel(dataA, 0, resultA.sessionName);
    drawPanel(dataB, panelWidth + gap, resultB.sessionName);

    if (showDiff && dataA.length === dataB.length) {
      const diffOffset = (panelWidth + gap) * 2;
      const g = svg.append("g").attr("transform", `translate(${diffOffset + margin.left},${margin.top})`);

      svg.append("text")
        .attr("x", diffOffset + margin.left + (numPositions * cellW) / 2)
        .attr("y", 16)
        .attr("text-anchor", "middle")
        .attr("font-size", 12)
        .attr("fill", "#a0a0c0")
        .text("Diff (A - B)");

      dataA.forEach((msgA, rowIdx) => {
        const msgB = dataB[rowIdx];
        if (!msgB) return;

        g.append("text")
          .attr("x", -4)
          .attr("y", rowIdx * cellH + cellH / 2)
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .attr("font-size", 9)
          .attr("fill", "#8888aa")
          .text(`L${msgA.layer}`);

        msgA.predictions.forEach((predsA, posIdx) => {
          const predsB = msgB.predictions[posIdx];
          if (!predsA?.[0] || !predsB?.[0]) return;
          const diff = predsA[0].prob - predsB[0].prob;
          const tokensDiffer = predsA[0].token !== predsB[0].token;

          g.append("rect")
            .attr("x", posIdx * cellW)
            .attr("y", rowIdx * cellH)
            .attr("width", cellW - 1)
            .attr("height", cellH - 1)
            .attr("fill", diffScale(diff))
            .attr("rx", 2)
            .attr("stroke", tokensDiffer ? "#ff6b6b" : "none")
            .attr("stroke-width", tokensDiffer ? 1.5 : 0)
            .style("cursor", "pointer")
            .on("mouseenter", (event) => {
              setTooltip({
                x: event.pageX + 10,
                y: event.pageY - 10,
                content: `L${msgA.layer} pos ${posIdx}\nA: ${predsA[0].token} (${(predsA[0].prob * 100).toFixed(1)}%)\nB: ${predsB[0].token} (${(predsB[0].prob * 100).toFixed(1)}%)\nΔ: ${(diff * 100).toFixed(1)}%`,
              });
            })
            .on("mouseleave", () => setTooltip(null));
        });
      });
    }

  }, [dataA, dataB, showDiff, resultA.sessionName, resultB.sessionName]);

  return (
    <div style={{ position: "relative" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
        <h3 style={{ fontSize: 13, color: "#a0a0c0" }}>
          A/B: {resultA.sessionName} vs {resultB.sessionName}
        </h3>
        <button onClick={() => setShowDiff(!showDiff)} style={{ fontSize: 11 }}>
          {showDiff ? "Hide Diff" : "Show Diff"}
        </button>
      </div>
      <div style={{ overflowX: "auto" }}>
        <svg ref={svgRef} />
      </div>
      {tooltip && (
        <div
          style={{
            position: "fixed",
            left: tooltip.x,
            top: tooltip.y,
            background: "#16213e",
            border: "1px solid #1a5276",
            borderRadius: 4,
            padding: "6px 10px",
            fontSize: 12,
            whiteSpace: "pre",
            pointerEvents: "none",
            zIndex: 100,
          }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Update VisualizationArea for A/B diff detection**

Update `testing/gui/frontend/src/components/VisualizationArea.tsx`:

```typescript
import { useStore } from "../state/store";
import { LogitLensHeatmap } from "./visualizations/LogitLensHeatmap";
import { LayerInfluence } from "./visualizations/LayerInfluence";
import { ABDiff } from "./visualizations/ABDiff";

export function VisualizationArea() {
  const results = useStore((s) => s.results);
  const activeResultId = useStore((s) => s.activeResultId);
  const setActiveResult = useStore((s) => s.setActiveResult);
  const clearResults = useStore((s) => s.clearResults);

  const vizResults = results.filter((r) => r.operation !== "generate");
  const activeResult = vizResults.find((r) => r.id === activeResultId);

  const abPair = activeResult && activeResult.id.includes("-B")
    ? null
    : activeResult && vizResults.find(
        (r) => r.id === `${activeResult.id}-B`
      );

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <h2>Visualization</h2>
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
                fontSize: 11,
                padding: "2px 8px",
                background: r.id === activeResultId ? "#1a5276" : "#0d1b2a",
              }}
            >
              {r.operation} · {r.sessionName}
              {vizResults.find((b) => b.id === `${r.id}-B`) ? " (A/B)" : ""}
            </button>
          ))}
        </div>
      )}

      {activeResult ? (
        abPair && activeResult.operation === "logit-lens" ? (
          <ABDiff resultA={activeResult} resultB={abPair} />
        ) : activeResult.operation === "logit-lens" ? (
          <LogitLensHeatmap result={activeResult} />
        ) : activeResult.operation === "influence" ? (
          <LayerInfluence result={activeResult} />
        ) : (
          <p style={{ color: "#666" }}>Visualization for {activeResult.operation} coming soon</p>
        )
      ) : (
        <p style={{ color: "#666" }}>Run a probe to see results here</p>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd /home/ai/ai-projects/llm/testing/gui/frontend && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add testing/gui/frontend/src/
git commit -m "feat(gui): add A/B diff visualization with side-by-side heatmaps and diff overlay"
```

---

## Task 12: Run Script + Load Endpoint + Integration

**Files:**
- Create: `testing/gui/run.sh`
- Modify: `testing/gui/backend/routes/sessions.py` (add POST /sessions load endpoint)
- Modify: `testing/gui/backend/app.py`

- [ ] **Step 1: Add POST /sessions load endpoint**

Add to `testing/gui/backend/routes/sessions.py`, before the existing routes:

```python
class LoadRequest(BaseModel):
    name: str
    model_id: str
    mode: str = "inspect"

    @field_validator("name")
    @classmethod
    def validate_slug(cls, v):
        if not re.match(r"^[a-z0-9][a-z0-9\-]{0,63}$", v):
            raise ValueError("Must be a valid slug")
        return v

@router.post("/sessions")
async def load_session(req: LoadRequest):
    mgr = get_manager()
    try:
        mgr.validate_name(req.name)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if req.name in mgr._sessions:
        raise HTTPException(409, f"Session '{req.name}' already exists")

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon import surgery

    try:
        model, tokenizer = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: surgery.load_model(req.model_id, mode=req.mode),
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")

    mgr.register(req.name, model, tokenizer,
                 model_id=req.model_id, mode=req.mode)
    return _session_summary(mgr.get(req.name))
```

Add the missing `asyncio` import at the top of the file:

```python
import asyncio
```

- [ ] **Step 2: Create run.sh**

Create `testing/gui/run.sh`:

```bash
#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "Starting backend..."
source testing/.venv/bin/activate
PYTHONPATH="$PROJECT_ROOT/testing" uvicorn gui.backend.app:app \
  --host 127.0.0.1 --port 8000 --reload \
  --reload-dir testing/gui/backend &
BACKEND_PID=$!

echo "Starting frontend..."
cd testing/gui/frontend
npm run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT

echo ""
echo "==================================="
echo "  LLM Surgeon GUI"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://127.0.0.1:8000"
echo "==================================="
echo ""

wait
```

- [ ] **Step 3: Make run.sh executable**

Run: `chmod +x /home/ai/ai-projects/llm/testing/gui/run.sh`

- [ ] **Step 4: Write integration test for load endpoint**

Add to `testing/gui/tests/test_session_routes.py`:

```python
@pytest.mark.asyncio
async def test_load_session_invalid_slug():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions", json={
            "name": "INVALID NAME",
            "model_id": "test/tiny",
        })
    assert resp.status_code == 422
```

- [ ] **Step 5: Run full backend test suite**

Run: `cd /home/ai/ai-projects/llm && source testing/.venv/bin/activate && python -m pytest testing/gui/tests/ -v`
Expected: All tests PASS.

- [ ] **Step 6: Verify frontend builds**

Run: `cd /home/ai/ai-projects/llm/testing/gui/frontend && npm run build`
Expected: Build completes, output in `dist/`.

- [ ] **Step 7: Commit**

```bash
git add testing/gui/
git commit -m "feat(gui): add model load endpoint, run.sh, and frontend build"
```

---

## Task 13: End-to-End Manual Verification

**Files:** None (verification only)

- [ ] **Step 1: Start the application**

Run: `cd /home/ai/ai-projects/llm && ./testing/gui/run.sh`

- [ ] **Step 2: Open browser and verify layout**

Open `http://localhost:5173` in Firefox. Verify:
- Four-panel layout renders (sidebar, probe config, visualization area, generation output)
- Dark theme applied
- No console errors

- [ ] **Step 3: Load a model**

In the sidebar:
- Enter model ID: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Enter session name: `baseline`
- Select mode: `4-bit (inspect)`
- Click Load
- Verify session appears in the list with layer count

- [ ] **Step 4: Run logit lens**

In probe config:
- Enter prompt: `The capital of France is`
- Select operation: `Logit Lens`
- Select session A: `baseline`
- Click Run
- Verify heatmap appears in visualization area with layers building

- [ ] **Step 5: Run layer influence**

- Select operation: `Layer Influence`
- Click Run
- Verify horizontal bar chart appears

- [ ] **Step 6: Run generation**

- Select operation: `Generate`
- Set max tokens to 20
- Click Run
- Verify tokens stream in the generation output panel
- Click a token to verify probability popover

- [ ] **Step 7: Test surgery + A/B**

- Clone `baseline` as `variant`
- Select `variant`, apply surgery: `remove_layers` with `{"layer_indices": [10]}`
- In probe config, select Session A: `baseline`, Session B: `variant`
- Run logit lens
- Verify A/B diff visualization with side-by-side heatmaps

- [ ] **Step 8: Test undo**

- Click Undo on `variant`
- Verify layer count returns to original

- [ ] **Step 9: Final commit**

If any fixes were needed during verification:

```bash
git add -A
git commit -m "fix(gui): address issues found during manual verification"
```
