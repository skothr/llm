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

@pytest.mark.asyncio
async def test_load_session_invalid_slug():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/sessions", json={
            "name": "INVALID NAME",
            "model_id": "test/tiny",
        })
    assert resp.status_code == 422
