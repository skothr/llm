import json
import pytest
from httpx import AsyncClient, ASGITransport
from gui.backend.app import app

@pytest.fixture
def mock_cache(tmp_path):
    m1 = tmp_path / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
    m1.mkdir()
    blobs1 = m1 / "blobs"
    blobs1.mkdir()
    (blobs1 / "weights").write_bytes(b"\x00" * 2048)
    (m1 / "refs").mkdir()
    snap1 = m1 / "snapshots" / "abc123"
    snap1.mkdir(parents=True)
    (snap1 / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        "num_hidden_layers": 22,
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "vocab_size": 32000,
        "intermediate_size": 5632,
        "max_position_embeddings": 2048,
    }))

    m2 = tmp_path / "models--openlm-research--open_llama_3b_v2"
    m2.mkdir()
    blobs2 = m2 / "blobs"
    blobs2.mkdir()
    (blobs2 / "weights").write_bytes(b"\x00" * 4096)
    (m2 / "refs").mkdir()
    snap2 = m2 / "snapshots" / "def456"
    snap2.mkdir(parents=True)
    (snap2 / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "num_hidden_layers": 26,
        "hidden_size": 3200,
        "num_attention_heads": 32,
        "vocab_size": 32000,
    }))

    (tmp_path / "CACHEDIR.TAG").touch()
    (tmp_path / "some_random_dir").mkdir()

    return tmp_path

@pytest.fixture
def mock_ollama(tmp_path):
    lib = tmp_path / "manifests" / "registry.ollama.ai" / "library"
    blobs = tmp_path / "blobs"
    blobs.mkdir(parents=True)

    (lib / "tinyllama").mkdir(parents=True)
    config_digest = "sha256-aabbcc"
    model_digest = "sha256-112233"
    (blobs / config_digest).write_text(json.dumps({
        "model_family": "llama",
        "model_type": "1B",
        "file_type": "Q4_0",
    }))
    (lib / "tinyllama" / "latest").write_text(json.dumps({
        "config": {"digest": config_digest.replace("-", ":", 1), "size": 100},
        "layers": [
            {"mediaType": "application/vnd.ollama.image.model",
             "digest": model_digest.replace("-", ":", 1), "size": 637699456},
        ],
    }))

    (lib / "dolphin-llama3").mkdir()
    config_digest2 = "sha256-ddeeff"
    (blobs / config_digest2).write_text(json.dumps({
        "model_family": "llama",
        "model_type": "8B",
        "file_type": "Q4_K_M",
    }))
    (lib / "dolphin-llama3" / "8b-256k").write_text(json.dumps({
        "config": {"digest": config_digest2.replace("-", ":", 1), "size": 120},
        "layers": [
            {"mediaType": "application/vnd.ollama.image.model",
             "digest": "sha256:445566", "size": 4920000000},
        ],
    }))
    return tmp_path

@pytest.mark.asyncio
async def test_models_available(mock_cache, tmp_path, monkeypatch):
    import gui.backend.routes.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "MODELS_CACHE", mock_cache)
    monkeypatch.setattr(sessions_mod, "OLLAMA_MODELS_DIR", tmp_path / "no-ollama")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models/available")
    assert resp.status_code == 200
    data = resp.json()
    model_ids = [m["model_id"] for m in data]
    assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in model_ids
    assert "openlm-research/open_llama_3b_v2" in model_ids
    assert all(m["source"] == "huggingface" for m in data)
    assert len(model_ids) == 2

@pytest.mark.asyncio
async def test_hf_metadata(mock_cache, tmp_path, monkeypatch):
    import gui.backend.routes.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "MODELS_CACHE", mock_cache)
    monkeypatch.setattr(sessions_mod, "OLLAMA_MODELS_DIR", tmp_path / "no-ollama")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models/available")
    data = resp.json()
    tiny = next(m for m in data if "TinyLlama" in m["model_id"])
    assert tiny["architecture"] == "llama"
    assert tiny["dtype"] == "bfloat16"
    assert tiny["num_layers"] == 22
    assert tiny["hidden_size"] == 2048
    assert tiny["num_heads"] == 32
    assert tiny["num_kv_heads"] == 4
    assert tiny["vocab_size"] == 32000
    assert tiny["intermediate_size"] == 5632
    assert tiny["file_size_bytes"] == 2048

    openllama = next(m for m in data if "open_llama" in m["model_id"])
    assert openllama["architecture"] == "llama"
    assert openllama["num_layers"] == 26
    assert openllama["hidden_size"] == 3200
    assert openllama["file_size_bytes"] == 4096

@pytest.mark.asyncio
async def test_models_available_empty(tmp_path, monkeypatch):
    import gui.backend.routes.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "MODELS_CACHE", tmp_path)
    monkeypatch.setattr(sessions_mod, "OLLAMA_MODELS_DIR", tmp_path / "no-ollama")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models/available")
    assert resp.status_code == 200
    assert resp.json() == []

@pytest.mark.asyncio
async def test_models_available_missing_dir(tmp_path, monkeypatch):
    import gui.backend.routes.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "MODELS_CACHE", tmp_path / "nonexistent")
    monkeypatch.setattr(sessions_mod, "OLLAMA_MODELS_DIR", tmp_path / "no-ollama")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models/available")
    assert resp.status_code == 200
    assert resp.json() == []

@pytest.mark.asyncio
async def test_ollama_models(mock_ollama, tmp_path, monkeypatch):
    import gui.backend.routes.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "MODELS_CACHE", tmp_path / "no-hf")
    monkeypatch.setattr(sessions_mod, "OLLAMA_MODELS_DIR", mock_ollama)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models/available")
    assert resp.status_code == 200
    data = resp.json()
    model_ids = [m["model_id"] for m in data]
    assert "tinyllama:latest" in model_ids
    assert "dolphin-llama3:8b-256k" in model_ids
    assert all(m["source"] == "ollama" for m in data)

@pytest.mark.asyncio
async def test_ollama_metadata(mock_ollama, tmp_path, monkeypatch):
    import gui.backend.routes.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "MODELS_CACHE", tmp_path / "no-hf")
    monkeypatch.setattr(sessions_mod, "OLLAMA_MODELS_DIR", mock_ollama)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models/available")
    data = resp.json()
    tiny = next(m for m in data if m["model_id"] == "tinyllama:latest")
    assert tiny["architecture"] == "llama"
    assert tiny["quantization"] == "Q4_0"
    assert tiny["model_size_label"] == "1B"
    assert tiny["file_size_bytes"] == 637699456

    dolphin = next(m for m in data if m["model_id"] == "dolphin-llama3:8b-256k")
    assert dolphin["architecture"] == "llama"
    assert dolphin["quantization"] == "Q4_K_M"
    assert dolphin["model_size_label"] == "8B"
    assert dolphin["file_size_bytes"] == 4920000000

@pytest.mark.asyncio
async def test_both_sources(mock_cache, mock_ollama, monkeypatch):
    import gui.backend.routes.sessions as sessions_mod
    monkeypatch.setattr(sessions_mod, "MODELS_CACHE", mock_cache)
    monkeypatch.setattr(sessions_mod, "OLLAMA_MODELS_DIR", mock_ollama)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/models/available")
    assert resp.status_code == 200
    data = resp.json()
    hf = [m for m in data if m["source"] == "huggingface"]
    ollama = [m for m in data if m["source"] == "ollama"]
    assert len(hf) == 2
    assert len(ollama) == 2
