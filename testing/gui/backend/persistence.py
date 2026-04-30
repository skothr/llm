"""Disk persistence for the SessionManager's session list.

We do not persist model weights — those live on disk already (HF cache or
GGUF blob) and are tens of MB to GB. We persist only the **delta** the
user produced inside the GUI: which sessions are registered, plus the
applied surgery ops on each. On startup we re-load each model from cache
and replay the ops, so a session that was modified before a backend
restart comes back modified.

State file shape (versioned for future migration):

    {
      "schema": 1,
      "sessions": [
        {
          "name": "tl",
          "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
          "mode": "fp16",
          "applied_ops": [
            {"operation": "zero_heads", "params": {"layer": 7, "heads": [3]}},
            ...
          ]
        },
        ...
      ]
    }

Restore is synchronous within the FastAPI lifespan startup phase —
serving doesn't begin until restore returns. Per-session failures (e.g.
model removed from HF cache) log a warning and skip; the rest restore.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger("gui.backend.persistence")

SCHEMA_VERSION = 1

# Configurable via env so tests can point it at a tmp_path.
DEFAULT_STATE_PATH = Path(
    os.environ.get(
        "LLM_SURGEON_GUI_STATE",
        str(Path(__file__).resolve().parents[2] / ".cache" / "gui_sessions.json"),
    )
)


def _entries_for(mgr: Any) -> list[dict[str, Any]]:
    """Build the JSON-safe per-session payload for the current manager."""
    out: list[dict[str, Any]] = []
    for info in mgr.list_sessions():
        # llama.cpp-only sessions also persist via name+model_id+mode; the
        # POST /sessions route's _is_ollama_id check re-routes them on
        # restore. applied_ops is always serializable (str/int/list params
        # only — see commit_surgery's op_map keys).
        out.append({
            "name": info.name,
            "model_id": info.model_id,
            "mode": info.mode,
            "applied_ops": list(info.applied_ops),
        })
    return out


def persist(mgr: Any, path: Path | None = None) -> None:
    """Write the current session list to disk. Best-effort: log + ignore
    on failure so a transient disk error doesn't break the route that
    triggered the write."""
    state_path = path or DEFAULT_STATE_PATH
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": SCHEMA_VERSION, "sessions": _entries_for(mgr)}
        # Atomic write: stage to tmp file, fsync, rename. Avoids leaving a
        # half-written file if the process is killed mid-write.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=state_path.parent, prefix=".gui_sessions.",
            suffix=".tmp", delete=False, encoding="utf-8",
        ) as tmp:
            json.dump(payload, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, state_path)
    except Exception:
        log.exception("Could not persist session state to %s", state_path)


async def restore(
    mgr: Any,
    register_one: Callable[[Any, str, str, str], Any],
    apply_op: Callable[[Any, str, dict[str, Any]], None],
    path: Path | None = None,
) -> None:
    """Load saved sessions from disk and replay them onto ``mgr``.

    Two callable injections keep this module independent of the heavy
    surgery / load_model imports — the route layer wires them in:

      * ``register_one(mgr, name, model_id, mode) -> Awaitable``: load the
        model and call mgr.register / register_llama. Async because the
        load is offloaded to an executor.
      * ``apply_op(model, op_name, params) -> None``: synchronously
        re-apply one committed surgery op to the freshly-loaded model.

    Per-session errors log and continue so one broken session can't
    block the rest from restoring.
    """
    state_path = path or DEFAULT_STATE_PATH
    if not state_path.exists():
        log.info("No saved session state at %s — starting clean", state_path)
        return
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Could not read session state %s — starting clean", state_path)
        return
    if data.get("schema") != SCHEMA_VERSION:
        log.warning(
            "Session state schema mismatch (have %s, expected %s); ignoring",
            data.get("schema"), SCHEMA_VERSION,
        )
        return

    entries = data.get("sessions", [])
    log.info("Restoring %d session(s) from %s", len(entries), state_path)
    loop = asyncio.get_running_loop()

    for entry in entries:
        name = entry.get("name")
        model_id = entry.get("model_id")
        mode = entry.get("mode")
        applied = entry.get("applied_ops", [])
        if not (name and model_id and mode):
            log.warning("Skipping malformed entry: %r", entry)
            continue
        try:
            info = await register_one(mgr, name, model_id, mode)
        except Exception:
            log.exception("Could not restore session '%s' — skipping", name)
            continue

        if not applied:
            continue
        # Replay committed ops on the freshly-loaded model. Mirrors the
        # commit_surgery loop: failure mid-replay leaves the session
        # partially modified but coherent (applied_ops match what's been
        # applied). The user can inspect and decide what to do.
        async with info.lock:
            for op in applied:
                op_name = op.get("operation")
                params = op.get("params", {})
                try:
                    await loop.run_in_executor(
                        None, lambda f=apply_op, m=info.model, n=op_name, p=params: f(m, n, p)
                    )
                    info._applied_ops.append(op)
                except Exception:
                    log.exception(
                        "Could not replay op %s on '%s' — stopping at %d/%d",
                        op_name, name, len(info._applied_ops), len(applied),
                    )
                    break
            if info.llama is not None and info._applied_ops:
                info.dirty = True
        log.info(
            "Restored '%s' with %d/%d applied op(s)",
            name, len(info._applied_ops), len(applied),
        )
