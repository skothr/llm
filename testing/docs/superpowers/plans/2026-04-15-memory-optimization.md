# Memory Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce RAM/VRAM usage for multi-model comparison workflows by making models inference-only, cleaning up GPU memory properly, and replacing snapshot-based undo with a deferred operation queue.

**Architecture:** Three independent changes to the session management layer. Changes 1 and 2 are isolated fixes to existing code. Change 3 replaces the snapshot/undo system with a command-pattern operation queue where surgery ops are staged, committed in batch, and revertible by reloading the clean model.

**Tech Stack:** Python, PyTorch, FastAPI, Pydantic, pytest, httpx (async test client)

**Spec:** `docs/superpowers/specs/2026-04-15-memory-optimization-design.md`

**Test command:** `testing/.venv/bin/python -m pytest testing/gui/tests/ -v`

---

### Task 1: Inference-only model setup

**Files:**
- Modify: `testing/gui/backend/sessions.py:72-85` (register)
- Test: `testing/gui/tests/test_sessions.py`

Note: `surgery.py:load_model()` eval-mode GPU fix was already applied earlier in this branch.

- [ ] **Step 1: Write failing tests for inference-only setup**

Add to `testing/gui/tests/test_sessions.py`:

```python
def test_register_sets_eval_mode(self, tiny_model, tiny_tokenizer):
    mgr = SessionManager()
    tiny_model.train()  # force training mode
    mgr.register("s1", tiny_model, tiny_tokenizer,
                  model_id="test/tiny", mode="eval")
    info = mgr.get("s1")
    assert not info.model.training

def test_register_disables_grad(self, tiny_model, tiny_tokenizer):
    mgr = SessionManager()
    for p in tiny_model.parameters():
        p.requires_grad = True  # force grads on
    mgr.register("s1", tiny_model, tiny_tokenizer,
                  model_id="test/tiny", mode="eval")
    info = mgr.get("s1")
    assert all(not p.requires_grad for p in info.model.parameters())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/test_sessions.py::TestSessionManager::test_register_sets_eval_mode testing/gui/tests/test_sessions.py::TestSessionManager::test_register_disables_grad -v`

Expected: FAIL — `register` doesn't call `.eval()` or `.requires_grad_(False)`

- [ ] **Step 3: Implement inference-only setup in register**

In `testing/gui/backend/sessions.py`, modify `register()` — add two lines after creating `info`, before inserting into `_sessions`:

```python
def register(self, name: str, model, tokenizer, *, model_id: str, mode: str) -> SessionInfo:
    self.validate_name(name)
    if name in self._sessions:
        raise ValueError(f"Session '{name}' already exists")
    model.eval()
    model.requires_grad_(False)
    num_layers = model.config.num_hidden_layers
    info = SessionInfo(
        name=name, model=model, tokenizer=tokenizer,
        model_id=model_id, mode=mode,
        _layer_map=list(range(num_layers)),
        _original_config=copy.deepcopy(model.config),
    )
    self._sessions[name] = info
    log.info("Registered session '%s' (model=%s, mode=%s, layers=%d)",
             name, model_id, mode, num_layers)
    return info
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/test_sessions.py -v`

Expected: ALL PASS (including existing tests — surgery ops use `tensor.data` which works on requires_grad=False tensors)

- [ ] **Step 5: Commit**

```bash
git add testing/gui/backend/sessions.py testing/gui/tests/test_sessions.py
git commit -m "feat(gui): set models to eval mode and disable gradients on register"
```

---

### Task 2: Proper GPU memory cleanup

**Files:**
- Modify: `testing/gui/backend/sessions.py:102-111` (to_cpu), `145-151` (delete)
- Modify: `testing/gui/backend/routes/probes.py:256-264` (generate_ws finally)
- Test: `testing/gui/tests/test_sessions.py`

- [ ] **Step 1: Write failing test for gc.collect in to_cpu**

Add to `testing/gui/tests/test_sessions.py`:

```python
from unittest.mock import patch

def test_to_cpu_calls_gc_collect(self, tiny_model, tiny_tokenizer):
    mgr = SessionManager()
    mgr.register("s1", tiny_model, tiny_tokenizer,
                  model_id="test/tiny", mode="eval")
    # Force model to look like it's on a non-cpu device by mocking
    # We can't actually move to GPU in tests, so test the code path
    # by verifying gc.collect is called when the model is already on cpu
    # (to_cpu early-returns for cpu models, so we test delete instead)
    with patch("gui.backend.sessions.gc.collect") as mock_gc:
        mgr.delete("s1")
        mock_gc.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/test_sessions.py::TestSessionManager::test_to_cpu_calls_gc_collect -v`

Expected: FAIL — `gc` not imported in sessions.py, `gc.collect()` not called

- [ ] **Step 3: Add gc.collect before empty_cache**

In `testing/gui/backend/sessions.py`, add `import gc` at the top (after `import copy`), then add `gc.collect()` before each `torch.cuda.empty_cache()` call:

```python
import gc
```

In `to_cpu()`:
```python
def to_cpu(self, name: str) -> None:
    info = self.get(name)
    if next(info.model.parameters()).device.type == "cpu":
        return
    if self._is_dispatch_model(info):
        return
    log.info("Moving session '%s' to CPU", name)
    info.model = info.model.cpu()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

In `delete()`:
```python
def delete(self, name: str) -> None:
    info = self.get(name)
    log.info("Deleting session '%s'", name)
    del self._sessions[name]
    del info.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/test_sessions.py -v`

Expected: ALL PASS

- [ ] **Step 5: Add to_cpu call in generate_ws finally block**

In `testing/gui/backend/routes/probes.py`, modify the `generate_ws` finally block (around line 256):

```python
    finally:
        log.info("WS generate disconnected (session='%s', tokens=%d, stop=%s)",
                 name, len(generated_tokens), stop_reason or "disconnect")
        try:
            mgr.to_cpu(name)
        except Exception:
            pass
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        try:
            await ws.close()
        except RuntimeError:
            pass
```

- [ ] **Step 6: Run all tests to verify no regressions**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/ -v`

Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add testing/gui/backend/sessions.py testing/gui/backend/routes/probes.py testing/gui/tests/test_sessions.py
git commit -m "feat(gui): gc.collect before empty_cache, release GPU after generate"
```

---

### Task 3: Replace SessionInfo undo_stack with operation queue

**Files:**
- Modify: `testing/gui/backend/sessions.py` (SessionInfo dataclass, remove snapshot/undo methods, add queue methods)
- Test: `testing/gui/tests/test_sessions.py`

- [ ] **Step 1: Write failing tests for the operation queue**

Replace the snapshot/undo tests in `testing/gui/tests/test_sessions.py` with operation queue tests. Remove `test_snapshot_and_undo`, `test_multi_undo`, `test_undo_without_snapshot_raises` and add:

```python
def test_stage_op(self, tiny_model, tiny_tokenizer):
    mgr = SessionManager()
    mgr.register("s1", tiny_model, tiny_tokenizer,
                  model_id="test/tiny", mode="eval")
    info = mgr.get("s1")
    info.stage_op("zero_mlp", {"layer": 0})
    assert len(info.pending_ops) == 1
    assert info.pending_ops[0]["operation"] == "zero_mlp"
    assert info.pending_ops[0]["params"] == {"layer": 0}

def test_undo_op_pops_last(self, tiny_model, tiny_tokenizer):
    mgr = SessionManager()
    mgr.register("s1", tiny_model, tiny_tokenizer,
                  model_id="test/tiny", mode="eval")
    info = mgr.get("s1")
    info.stage_op("zero_mlp", {"layer": 0})
    info.stage_op("remove_layers", {"layer_indices": [1]})
    popped = info.undo_op()
    assert popped["operation"] == "remove_layers"
    assert len(info.pending_ops) == 1

def test_undo_op_empty_raises(self, tiny_model, tiny_tokenizer):
    mgr = SessionManager()
    mgr.register("s1", tiny_model, tiny_tokenizer,
                  model_id="test/tiny", mode="eval")
    info = mgr.get("s1")
    with pytest.raises(ValueError, match="No pending operations"):
        info.undo_op()

def test_clear_pending(self, tiny_model, tiny_tokenizer):
    mgr = SessionManager()
    mgr.register("s1", tiny_model, tiny_tokenizer,
                  model_id="test/tiny", mode="eval")
    info = mgr.get("s1")
    info.stage_op("zero_mlp", {"layer": 0})
    info.stage_op("zero_mlp", {"layer": 1})
    info.clear_pending()
    assert len(info.pending_ops) == 0

def test_record_applied_ops(self, tiny_model, tiny_tokenizer):
    mgr = SessionManager()
    mgr.register("s1", tiny_model, tiny_tokenizer,
                  model_id="test/tiny", mode="eval")
    info = mgr.get("s1")
    ops = [
        {"operation": "zero_mlp", "params": {"layer": 0}},
        {"operation": "remove_layers", "params": {"layer_indices": [1]}},
    ]
    info.record_applied(ops)
    assert len(info.applied_ops) == 2
    assert info.applied_ops[0]["operation"] == "zero_mlp"

def test_revert_moves_applied_to_pending(self, tiny_model, tiny_tokenizer):
    mgr = SessionManager()
    mgr.register("s1", tiny_model, tiny_tokenizer,
                  model_id="test/tiny", mode="eval")
    info = mgr.get("s1")
    ops = [
        {"operation": "zero_mlp", "params": {"layer": 0}},
        {"operation": "remove_layers", "params": {"layer_indices": [1]}},
    ]
    info.record_applied(ops)
    info.revert()
    assert len(info.applied_ops) == 0
    assert len(info.pending_ops) == 2
    assert len(info.op_history) == 1
    assert info.pending_ops[0]["operation"] == "zero_mlp"

def test_revert_empty_raises(self, tiny_model, tiny_tokenizer):
    mgr = SessionManager()
    mgr.register("s1", tiny_model, tiny_tokenizer,
                  model_id="test/tiny", mode="eval")
    info = mgr.get("s1")
    with pytest.raises(ValueError, match="No applied operations"):
        info.revert()

def test_has_pending(self, tiny_model, tiny_tokenizer):
    mgr = SessionManager()
    mgr.register("s1", tiny_model, tiny_tokenizer,
                  model_id="test/tiny", mode="eval")
    info = mgr.get("s1")
    assert not info.has_pending
    info.stage_op("zero_mlp", {"layer": 0})
    assert info.has_pending
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/test_sessions.py -v`

Expected: FAIL — `stage_op`, `undo_op`, `pending_ops`, etc. don't exist

- [ ] **Step 3: Implement the operation queue on SessionInfo**

Replace the undo_stack fields and methods in `testing/gui/backend/sessions.py`. The new `SessionInfo`:

```python
@dataclass
class SessionInfo:
    name: str
    model: object
    tokenizer: object
    model_id: str
    mode: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _pending_ops: list = field(default_factory=list, repr=False)
    _applied_ops: list = field(default_factory=list, repr=False)
    _op_history: list = field(default_factory=list, repr=False)
    _layer_map: list = field(default_factory=list, repr=False)
    _original_config: object = field(default=None, repr=False)

    @property
    def pending_ops(self) -> list:
        return list(self._pending_ops)

    @property
    def applied_ops(self) -> list:
        return list(self._applied_ops)

    @property
    def op_history(self) -> list:
        return list(self._op_history)

    @property
    def has_pending(self) -> bool:
        return len(self._pending_ops) > 0

    def stage_op(self, operation: str, params: dict) -> dict:
        entry = {"operation": operation, "params": params}
        self._pending_ops.append(entry)
        return entry

    def undo_op(self) -> dict:
        if not self._pending_ops:
            raise ValueError("No pending operations to undo")
        return self._pending_ops.pop()

    def clear_pending(self) -> None:
        self._pending_ops.clear()

    def record_applied(self, ops: list) -> None:
        self._applied_ops.extend(ops)

    def revert(self) -> None:
        if not self._applied_ops:
            raise ValueError("No applied operations to revert")
        self._op_history.append(list(self._applied_ops))
        self._pending_ops = list(self._applied_ops) + list(self._pending_ops)
        self._applied_ops.clear()
```

Remove these methods from `SessionManager`:
- `snapshot()`
- `undo()`

Remove these properties from the old `SessionInfo`:
- `has_snapshot`
- `undo_depth`
- `snapshot_size_mb`
- `MAX_UNDO`
- `_undo_stack`

- [ ] **Step 4: Run tests to verify they pass**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/test_sessions.py -v`

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add testing/gui/backend/sessions.py testing/gui/tests/test_sessions.py
git commit -m "feat(gui): replace snapshot undo with operation queue on SessionInfo"
```

---

### Task 4: Update surgery route to stage ops instead of applying

**Files:**
- Modify: `testing/gui/backend/routes/sessions.py:222-266` (apply_surgery → stage_surgery)
- Modify: `testing/gui/backend/routes/sessions.py:268-279` (undo_surgery → undo_staged_op)
- Modify: `testing/gui/backend/routes/sessions.py:86-97` (SessionSummary)
- Test: `testing/gui/tests/test_surgery_routes.py`

- [ ] **Step 1: Write failing tests for staging surgery ops**

Replace tests in `testing/gui/tests/test_surgery_routes.py`:

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
async def test_surgery_stages_op(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions/test-model/surgery", json={
            "operation": "zero_mlp",
            "params": {"layer": 0}
        })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pending"]) == 1
    assert data["pending"][0]["operation"] == "zero_mlp"
    # Model should be unchanged (still 4 layers)
    info = manager.get("test-model")
    assert info.model.config.num_hidden_layers == 4

@pytest.mark.asyncio
async def test_surgery_stages_multiple_ops(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/test-model/surgery", json={
            "operation": "zero_mlp", "params": {"layer": 0}
        })
        resp = await client.post("/api/sessions/test-model/surgery", json={
            "operation": "zero_mlp", "params": {"layer": 1}
        })
    data = resp.json()
    assert len(data["pending"]) == 2

@pytest.mark.asyncio
async def test_surgery_undo_pops_staged(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/test-model/surgery", json={
            "operation": "zero_mlp", "params": {"layer": 0}
        })
        await client.post("/api/sessions/test-model/surgery", json={
            "operation": "remove_layers", "params": {"layer_indices": [1]}
        })
        resp = await client.delete("/api/sessions/test-model/surgery/last")
    assert resp.status_code == 200
    data = resp.json()
    assert data["removed"]["operation"] == "remove_layers"
    assert len(data["pending"]) == 1

@pytest.mark.asyncio
async def test_surgery_undo_empty_returns_409(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/sessions/test-model/surgery/last")
    assert resp.status_code == 409

@pytest.mark.asyncio
async def test_surgery_pending_list(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/test-model/surgery", json={
            "operation": "zero_mlp", "params": {"layer": 0}
        })
        resp = await client.get("/api/sessions/test-model/surgery/pending")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pending"]) == 1

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

- [ ] **Step 2: Run tests to verify they fail**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/test_surgery_routes.py -v`

Expected: FAIL — routes still apply immediately, return wrong shape, wrong HTTP methods

- [ ] **Step 3: Update SessionSummary to remove snapshot fields, add pending_count**

In `testing/gui/backend/routes/sessions.py`, update the response models and `_session_summary`:

```python
class SessionSummary(BaseModel):
    name: str
    model_id: str
    mode: str
    num_layers: int
    pending_count: int
    device: str
```

Update `_session_summary`:

```python
def _session_summary(info) -> dict:
    config = info.model.config
    return SessionSummary(
        name=info.name,
        model_id=info.model_id,
        mode=info.mode,
        num_layers=config.num_hidden_layers,
        pending_count=len(info.pending_ops),
        device=str(next(info.model.parameters()).device),
    ).model_dump()
```

- [ ] **Step 4: Replace surgery route to stage instead of apply**

Replace the `apply_surgery` endpoint:

```python
VALID_OPS = {op["name"] for op in SURGERY_OPS}

@router.post("/sessions/{name}/surgery")
async def stage_surgery(name: str, req: SurgeryRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    if req.operation not in VALID_OPS:
        raise HTTPException(422, f"Unknown surgery operation: '{req.operation}'")

    log.info("Staging op on '%s': %s(%s)", name, req.operation, req.params)
    info.stage_op(req.operation, req.params)
    return {"pending": info.pending_ops}
```

- [ ] **Step 5: Replace undo route with DELETE for last staged op**

Replace the `undo_surgery` endpoint:

```python
@router.delete("/sessions/{name}/surgery/last")
async def undo_staged_op(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    try:
        removed = info.undo_op()
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"removed": removed, "pending": info.pending_ops}
```

Remove the old `POST /sessions/{name}/surgery/undo` route.

- [ ] **Step 6: Add GET endpoint for pending queue**

```python
@router.get("/sessions/{name}/surgery/pending")
async def get_pending_ops(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    return {"pending": info.pending_ops}
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/test_surgery_routes.py -v`

Expected: ALL PASS

- [ ] **Step 8: Run full test suite for regressions**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/ -v`

Expected: ALL PASS (session route tests that reference `has_snapshot`/`undo_depth` may need updating — fix any that reference removed fields in `_session_summary`)

- [ ] **Step 9: Commit**

```bash
git add testing/gui/backend/routes/sessions.py testing/gui/tests/test_surgery_routes.py
git commit -m "feat(gui): surgery endpoint stages ops instead of applying immediately"
```

---

### Task 5: Commit and revert endpoints

**Files:**
- Modify: `testing/gui/backend/routes/sessions.py` (add commit, revert, history endpoints)
- Test: `testing/gui/tests/test_surgery_routes.py`

- [ ] **Step 1: Write failing tests for commit**

Add to `testing/gui/tests/test_surgery_routes.py`:

```python
@pytest.mark.asyncio
async def test_commit_applies_ops(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/test-model/surgery", json={
            "operation": "zero_mlp", "params": {"layer": 0}
        })
        resp = await client.post("/api/sessions/test-model/surgery/commit")
    assert resp.status_code == 200
    data = resp.json()
    assert data["applied_count"] == 1
    assert data["pending"] == []
    info = manager.get("test-model")
    assert len(info.applied_ops) == 1

@pytest.mark.asyncio
async def test_commit_applies_remove_layers(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/test-model/surgery", json={
            "operation": "remove_layers", "params": {"layer_indices": [0]}
        })
        resp = await client.post("/api/sessions/test-model/surgery/commit")
    assert resp.status_code == 200
    info = manager.get("test-model")
    assert info.model.config.num_hidden_layers == 3

@pytest.mark.asyncio
async def test_commit_empty_returns_409(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions/test-model/surgery/commit")
    assert resp.status_code == 409

@pytest.mark.asyncio
async def test_commit_updates_layer_map(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/test-model/surgery", json={
            "operation": "remove_layers", "params": {"layer_indices": [0]}
        })
        await client.post("/api/sessions/test-model/surgery/commit")
        resp = await client.get("/api/sessions/test-model/info")
    data = resp.json()
    assert data["layer_map"] == [1, 2, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/test_surgery_routes.py::test_commit_applies_ops testing/gui/tests/test_surgery_routes.py::test_commit_empty_returns_409 -v`

Expected: FAIL — commit endpoint doesn't exist

- [ ] **Step 3: Implement commit endpoint**

Add to `testing/gui/backend/routes/sessions.py`:

```python
@router.post("/sessions/{name}/surgery/commit")
async def commit_surgery(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    if not info.has_pending:
        raise HTTPException(409, "No pending operations to commit")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon import surgery
    from ..sessions import update_layer_map

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

    pending = info.pending_ops
    applied = []

    for op in pending:
        op_name = op["operation"]
        params = op["params"]
        if op_name not in op_map:
            raise HTTPException(422, f"Unknown operation in queue: '{op_name}'")
        log.info("Committing op on '%s': %s(%s)", name, op_name, params)
        try:
            op_map[op_name](info.model, params)
        except (IndexError, ValueError) as e:
            log.warning("Commit failed on '%s' at op %s: %s", name, op_name, e)
            raise HTTPException(422, f"Operation '{op_name}' failed: {e}")
        info._layer_map = update_layer_map(info._layer_map, op_name, params)
        applied.append(op)

    info.record_applied(applied)
    info.clear_pending()

    return {
        "applied_count": len(applied),
        "pending": info.pending_ops,
        "info": _session_info(info),
    }
```

- [ ] **Step 4: Run commit tests to verify they pass**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/test_surgery_routes.py -k commit -v`

Expected: ALL PASS

- [ ] **Step 5: Write failing tests for revert and history**

Add to `testing/gui/tests/test_surgery_routes.py`:

```python
@pytest.mark.asyncio
async def test_revert_reloads_model(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/test-model/surgery", json={
            "operation": "remove_layers", "params": {"layer_indices": [0]}
        })
        await client.post("/api/sessions/test-model/surgery/commit")
        resp = await client.post("/api/sessions/test-model/surgery/revert")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pending"]) == 1
    assert data["pending"][0]["operation"] == "remove_layers"
    info = manager.get("test-model")
    assert info.model.config.num_hidden_layers == 4

@pytest.mark.asyncio
async def test_revert_empty_returns_409(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions/test-model/surgery/revert")
    assert resp.status_code == 409

@pytest.mark.asyncio
async def test_history_shows_reverted_sequences(register_tiny):
    register_tiny("test-model")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/sessions/test-model/surgery", json={
            "operation": "zero_mlp", "params": {"layer": 0}
        })
        await client.post("/api/sessions/test-model/surgery/commit")
        await client.post("/api/sessions/test-model/surgery/revert")
        resp = await client.get("/api/sessions/test-model/surgery/history")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["history"]) == 1
    assert data["history"][0][0]["operation"] == "zero_mlp"
```

- [ ] **Step 6: Implement revert and history endpoints**

The revert endpoint needs to reload the model. Since tests use tiny models registered directly (not via `load_model`), we need to handle this. In tests, the model_id is `"test/tiny"` which isn't a real path — so revert uses `from_config` + random init for test models, and `load_model` for real ones.

Add to `testing/gui/backend/routes/sessions.py`:

```python
@router.post("/sessions/{name}/surgery/revert")
async def revert_surgery(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    if not info.applied_ops:
        raise HTTPException(409, "No applied operations to revert")

    log.info("Reverting session '%s' — reloading clean model", name)
    info.revert()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon import surgery
    from llm_surgeon.surgery import _snapshot_dir

    try:
        if _snapshot_dir(info.model_id):
            model, _ = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: surgery.load_model(info.model_id, mode=info.mode),
            )
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_config(info._original_config)
        model.eval()
        model.requires_grad_(False)
        if not hasattr(model, "hf_device_map"):
            model = model.cpu()
    except Exception as e:
        log.exception("Revert failed for session '%s'", name)
        raise HTTPException(500, f"Revert failed: {e}")

    info.model = model
    info._layer_map = list(range(info._original_config.num_hidden_layers))

    return {
        "pending": info.pending_ops,
        "info": _session_info(info),
    }

@router.get("/sessions/{name}/surgery/history")
async def get_op_history(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    return {"history": info.op_history}
```

- [ ] **Step 7: Run all tests**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/ -v`

Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add testing/gui/backend/routes/sessions.py testing/gui/tests/test_surgery_routes.py
git commit -m "feat(gui): add commit, revert, and history endpoints for surgery queue"
```

---

### Task 6: Update session summary tests for new schema

**Files:**
- Modify: `testing/gui/tests/test_session_routes.py`
- Modify: `testing/gui/tests/test_surgery_routes.py` (if any reference old fields)

- [ ] **Step 1: Check for references to removed fields**

Search test files for `has_snapshot`, `undo_depth`, `snapshot_size_mb` — these fields were removed from `SessionSummary`. Update any test that checks these fields.

In `testing/gui/tests/test_surgery_routes.py`, the old `test_surgery_creates_snapshot` test was already replaced in Task 4. Verify no remaining references.

In `testing/gui/tests/test_session_routes.py`, check if `test_list_sessions_with_model` references removed fields. Current code checks `name`, `model_id`, `num_layers` — these are all still present. No changes needed unless the test checks `has_snapshot`.

- [ ] **Step 2: Run full test suite**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/ -v`

Expected: ALL PASS. If any test fails due to missing `has_snapshot`/`undo_depth`/`snapshot_size_mb` in the response, update that test to check `pending_count` instead.

- [ ] **Step 3: Commit (if changes were needed)**

```bash
git add testing/gui/tests/
git commit -m "test(gui): update tests for new session summary schema"
```

---

### Task 7: Clean up removed code

**Files:**
- Modify: `testing/gui/backend/sessions.py` (remove update_layer_map if it's only used by the old surgery route)

- [ ] **Step 1: Verify update_layer_map is still used**

`update_layer_map` is used in the commit endpoint (Task 5). It stays. Verify no dead code from the old snapshot system remains in `sessions.py`.

- [ ] **Step 2: Remove SurgeryResponse model if unused**

The old `apply_surgery` returned a `SurgeryResponse` with `operations` and `info` fields. The new `stage_surgery` returns a simple dict. Check if `SurgeryResponse` is still referenced anywhere. If not, remove it from `testing/gui/backend/routes/sessions.py`.

- [ ] **Step 3: Run full test suite**

Run: `testing/.venv/bin/python -m pytest testing/gui/tests/ -v`

Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add testing/gui/backend/routes/sessions.py testing/gui/backend/sessions.py
git commit -m "refactor(gui): remove dead code from snapshot-based undo system"
```
