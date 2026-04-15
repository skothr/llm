import json
import pytest
from starlette.testclient import TestClient
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

def test_logit_lens_streams_layers(register_tiny):
    register_tiny("baseline")
    client = TestClient(app)
    with client.websocket_connect("/ws/sessions/baseline/logit-lens") as ws:
        ws.send_text(json.dumps({"prompt": "tok1 tok2 tok3", "top_k": 5}))
        messages = []
        while True:
            msg = json.loads(ws.receive_text())
            messages.append(msg)
            if msg["type"] in ("complete", "error"):
                break
    data_msgs = [m for m in messages if m["type"] == "data"]
    assert len(data_msgs) > 0
    assert all("layer" in m for m in data_msgs)
    assert all("sublayer" in m for m in data_msgs)
    assert messages[-1]["type"] == "complete"

def test_logit_lens_missing_session():
    client = TestClient(app)
    with client.websocket_connect("/ws/sessions/nonexistent/logit-lens") as ws:
        ws.send_text(json.dumps({"prompt": "tok1", "top_k": 5}))
        msg = json.loads(ws.receive_text())
    assert msg["type"] == "error"
    assert "not found" in msg["message"]


def test_generate_streams_tokens(register_tiny):
    register_tiny("baseline")
    client = TestClient(app)
    with client.websocket_connect("/ws/sessions/baseline/generate") as ws:
        ws.send_text(json.dumps({
            "prompt": "tok1 tok2",
            "max_tokens": 5,
            "temperature": 0.0,
            "prob_top_k": 3,
        }))
        messages = []
        while True:
            msg = json.loads(ws.receive_text())
            messages.append(msg)
            if msg["type"] in ("complete", "error"):
                break
    data_msgs = [m for m in messages if m["type"] == "data"]
    assert len(data_msgs) > 0
    assert all("token" in m for m in data_msgs)
    assert all("top_k" in m for m in data_msgs)


def test_intervene_streams_results(register_tiny):
    register_tiny("baseline")
    client = TestClient(app)
    with client.websocket_connect("/ws/sessions/baseline/intervene") as ws:
        ws.send_text(json.dumps({
            "prompt": "tok1 tok2 tok3",
            "interventions": [
                {"layer": 0, "sublayer": "ffn", "op": "scale", "params": {"factor": 0.5}}
            ],
            "capture_logit_lens": True,
        }))
        messages = []
        while True:
            msg = json.loads(ws.receive_text())
            messages.append(msg)
            if msg["type"] in ("complete", "error"):
                break
    assert messages[-1]["type"] == "complete"
