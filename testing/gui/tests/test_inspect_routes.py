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
                         model_id="test/tiny", mode="fp16")
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
