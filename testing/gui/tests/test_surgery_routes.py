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
