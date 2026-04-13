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
    _snapshot_model: object = field(default=None, repr=False)

    @property
    def has_snapshot(self) -> bool:
        return self._snapshot_model is not None

    @property
    def snapshot_size_mb(self) -> float:
        if self._snapshot_model is None:
            return 0.0
        return sum(p.nelement() * p.element_size() for p in self._snapshot_model.parameters()) / 1e6

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
        info._snapshot_model = copy.deepcopy(info.model)

    def undo(self, name: str) -> None:
        info = self.get(name)
        if info._snapshot_model is None:
            raise ValueError(f"No snapshot available for session '{name}'")
        info.model = info._snapshot_model
        info._snapshot_model = None
