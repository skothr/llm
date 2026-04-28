"""Round-trip tests for the session-persistence module.

These don't actually load a model — they inject test callables for
register_one and apply_op so the persistence layer can be exercised in
isolation from torch / surgery imports.
"""

import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gui.backend import persistence


@dataclass
class _StubInfo:
    """Mimics SessionInfo just enough for persistence to work with it."""
    name: str
    model_id: str
    mode: str
    model: Any = None
    llama: Any = None
    dirty: bool = False
    _applied_ops: list = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def applied_ops(self) -> list:
        return list(self._applied_ops)


class _StubManager:
    """Tiny stand-in for SessionManager. list_sessions/list/register only."""
    def __init__(self):
        self._sessions: dict[str, _StubInfo] = {}

    def list_sessions(self) -> list[_StubInfo]:
        return list(self._sessions.values())

    def register_stub(self, info: _StubInfo):
        self._sessions[info.name] = info


def test_persist_writes_expected_schema(tmp_path: Path):
    mgr = _StubManager()
    mgr.register_stub(_StubInfo(name="a", model_id="x/y", mode="fp16"))
    state = tmp_path / "s.json"
    persistence.persist(mgr, path=state)
    data = json.loads(state.read_text())
    assert data["schema"] == persistence.SCHEMA_VERSION
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["name"] == "a"
    assert data["sessions"][0]["model_id"] == "x/y"
    assert data["sessions"][0]["mode"] == "fp16"
    assert data["sessions"][0]["applied_ops"] == []


def test_persist_includes_applied_ops(tmp_path: Path):
    mgr = _StubManager()
    info = _StubInfo(name="a", model_id="x/y", mode="fp16")
    info._applied_ops.append({"operation": "zero_heads", "params": {"layer": 7, "heads": [3]}})
    info._applied_ops.append({"operation": "scale_heads", "params": {"layer": 2, "heads": [1], "factor": 0.5}})
    mgr.register_stub(info)
    state = tmp_path / "s.json"
    persistence.persist(mgr, path=state)
    data = json.loads(state.read_text())
    ops = data["sessions"][0]["applied_ops"]
    assert len(ops) == 2
    assert ops[0]["operation"] == "zero_heads"
    assert ops[1]["params"]["factor"] == 0.5


def test_persist_atomic_write_no_tmp_left(tmp_path: Path):
    """Confirm the temp-file/rename pattern leaves no tmp file behind."""
    mgr = _StubManager()
    mgr.register_stub(_StubInfo(name="a", model_id="x/y", mode="fp16"))
    persistence.persist(mgr, path=tmp_path / "s.json")
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".gui_sessions.") and p.suffix == ".tmp"]
    assert leftovers == []


@pytest.mark.asyncio
async def test_restore_replays_registration_and_ops(tmp_path: Path):
    """Round-trip: persist a session with one applied op, clear the
    manager, restore — assert register_one and apply_op were both
    invoked with the right arguments."""
    # Source state file as if a prior backend wrote it
    state_path = tmp_path / "s.json"
    state_path.write_text(json.dumps({
        "schema": persistence.SCHEMA_VERSION,
        "sessions": [{
            "name": "tl",
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "mode": "fp16",
            "applied_ops": [{"operation": "zero_heads", "params": {"layer": 7, "heads": [3]}}],
        }],
    }))

    mgr = _StubManager()
    register_calls: list[tuple] = []
    apply_calls: list[tuple] = []

    async def mock_register(m, name, model_id, mode):
        register_calls.append((name, model_id, mode))
        info = _StubInfo(name=name, model_id=model_id, mode=mode, model="STUB_MODEL")
        m.register_stub(info)
        return info

    def mock_apply(model, op_name, params):
        apply_calls.append((model, op_name, params))

    await persistence.restore(mgr, mock_register, mock_apply, path=state_path)

    assert register_calls == [("tl", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "fp16")]
    assert apply_calls == [("STUB_MODEL", "zero_heads", {"layer": 7, "heads": [3]})]
    # Session is now in the manager
    assert "tl" in mgr._sessions
    # Applied ops were appended to the new info
    assert mgr._sessions["tl"].applied_ops == [
        {"operation": "zero_heads", "params": {"layer": 7, "heads": [3]}}
    ]


@pytest.mark.asyncio
async def test_restore_skips_session_when_register_fails(tmp_path: Path):
    """If load_model raises (e.g. model gone from cache), the bad
    session is skipped and the rest still restore."""
    state_path = tmp_path / "s.json"
    state_path.write_text(json.dumps({
        "schema": persistence.SCHEMA_VERSION,
        "sessions": [
            {"name": "bad", "model_id": "missing/model", "mode": "fp16", "applied_ops": []},
            {"name": "good", "model_id": "ok/model", "mode": "fp16", "applied_ops": []},
        ],
    }))

    mgr = _StubManager()

    async def mock_register(m, name, model_id, mode):
        if model_id == "missing/model":
            raise RuntimeError("simulated cache miss")
        info = _StubInfo(name=name, model_id=model_id, mode=mode)
        m.register_stub(info)
        return info

    def mock_apply(model, op_name, params):
        pass

    await persistence.restore(mgr, mock_register, mock_apply, path=state_path)

    assert "bad" not in mgr._sessions
    assert "good" in mgr._sessions


@pytest.mark.asyncio
async def test_restore_no_state_file_starts_clean(tmp_path: Path):
    """Missing state file is the normal first-boot case — restore
    must be a no-op, not an error."""
    mgr = _StubManager()
    await persistence.restore(mgr, lambda *a, **k: None, lambda *a, **k: None,
                              path=tmp_path / "absent.json")
    assert mgr._sessions == {}


@pytest.mark.asyncio
async def test_restore_ignores_wrong_schema(tmp_path: Path):
    """Future schema bumps must be loud-skip, not silent-load (which
    would corrupt new state on the next persist)."""
    state_path = tmp_path / "s.json"
    state_path.write_text(json.dumps({
        "schema": 999,
        "sessions": [{"name": "x", "model_id": "y", "mode": "fp16"}],
    }))

    mgr = _StubManager()
    register_calls: list[tuple] = []

    async def mock_register(m, name, model_id, mode):
        register_calls.append((name, model_id, mode))
        return _StubInfo(name=name, model_id=model_id, mode=mode)

    await persistence.restore(mgr, mock_register, lambda *a: None, path=state_path)
    assert register_calls == []
    assert mgr._sessions == {}


def test_persist_resilient_to_unwritable_path(tmp_path: Path):
    """Best-effort: if write fails, persist logs and returns rather than
    raising into the route handler that triggered it."""
    mgr = _StubManager()
    mgr.register_stub(_StubInfo(name="a", model_id="x/y", mode="fp16"))
    # /proc/sys/kernel/cap_last_cap is read-only across the whole tree
    # on Linux; writing under it always fails. Use an unwritable target
    # to simulate disk failure.
    bad_path = Path("/proc/sys/kernel/cap_last_cap/should_fail.json")
    persistence.persist(mgr, path=bad_path)
    # If we got here, the function swallowed the exception as designed.
