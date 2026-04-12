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
