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
