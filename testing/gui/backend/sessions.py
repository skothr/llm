import copy
import re
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch

@dataclass
class SessionInfo:
    name: str
    model: object
    tokenizer: object
    model_id: str
    mode: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _undo_stack: list = field(default_factory=list, repr=False)
    MAX_UNDO: int = 5

    @property
    def has_snapshot(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def undo_depth(self) -> int:
        return len(self._undo_stack)

    @property
    def snapshot_size_mb(self) -> float:
        if not self._undo_stack:
            return 0.0
        return sum(
            sum(t.nelement() * t.element_size() for t in entry["state"].values())
            for entry in self._undo_stack
        ) / 1e6

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._~\-]{0,63}$")

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SessionInfo] = {}

    def validate_name(self, name: str) -> None:
        if not _NAME_RE.match(name):
            raise ValueError(
                f"Invalid name: '{name}'. Must be alphanumeric, hyphens, underscores, periods, or tildes, 1-64 chars."
            )

    def register(self, name: str, model, tokenizer, *, model_id: str, mode: str) -> SessionInfo:
        self.validate_name(name)
        if name in self._sessions:
            raise ValueError(f"Session '{name}' already exists")
        info = SessionInfo(
            name=name, model=model, tokenizer=tokenizer,
            model_id=model_id, mode=mode,
        )
        self._sessions[name] = info
        return info

    def get(self, name: str) -> SessionInfo:
        if name not in self._sessions:
            raise KeyError(f"Session '{name}' not found")
        return self._sessions[name]

    def list_sessions(self) -> list:
        return list(self._sessions.values())

    def delete(self, name: str) -> None:
        info = self.get(name)
        del self._sessions[name]
        del info.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def snapshot(self, name: str) -> None:
        info = self.get(name)
        if len(info._undo_stack) >= info.MAX_UNDO:
            info._undo_stack.pop(0)
        entry = {
            "state": {k: v.cpu().clone() for k, v in info.model.state_dict().items()},
            "config": copy.deepcopy(info.model.config),
        }
        info._undo_stack.append(entry)

    def undo(self, name: str) -> None:
        info = self.get(name)
        if not info._undo_stack:
            raise ValueError(f"No undo history for session '{name}'")
        entry = info._undo_stack.pop()
        device = next(info.model.parameters()).device
        info.model.config = entry["config"]
        from transformers import AutoModelForCausalLM
        restored = AutoModelForCausalLM.from_config(entry["config"])
        restored.load_state_dict(entry["state"])
        restored.to(device)
        restored.eval()
        info.model = restored
