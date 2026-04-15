import copy
import gc
import logging
import re
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch

log = logging.getLogger("gui.backend.sessions")

@dataclass
class SessionInfo:
    name: str
    model: object
    tokenizer: object
    model_id: str
    mode: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _pending_ops: list = field(default_factory=list, repr=False)
    _applied_ops: list = field(default_factory=list, repr=False)
    _op_history: list = field(default_factory=list, repr=False)
    _layer_map: list = field(default_factory=list, repr=False)
    _original_config: object = field(default=None, repr=False)

    @property
    def pending_ops(self) -> list:
        return list(self._pending_ops)

    @property
    def applied_ops(self) -> list:
        return list(self._applied_ops)

    @property
    def op_history(self) -> list:
        return list(self._op_history)

    @property
    def has_pending(self) -> bool:
        return len(self._pending_ops) > 0

    def stage_op(self, operation: str, params: dict) -> dict:
        entry = {"operation": operation, "params": params}
        self._pending_ops.append(entry)
        return entry

    def undo_op(self) -> dict:
        if not self._pending_ops:
            raise ValueError("No pending operations to undo")
        return self._pending_ops.pop()

    def clear_pending(self) -> None:
        self._pending_ops.clear()

    def record_applied(self, ops: list) -> None:
        self._applied_ops.extend(ops)

    def revert(self) -> None:
        if not self._applied_ops:
            raise ValueError("No applied operations to revert")
        self._op_history.append(list(self._applied_ops))
        self._pending_ops = list(self._applied_ops) + list(self._pending_ops)
        self._applied_ops.clear()

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._~\-]{0,63}$")

def update_layer_map(current_map: list, operation: str, params: dict) -> list:
    m = list(current_map)
    if operation == "remove_layers":
        for i in sorted(params["layer_indices"], reverse=True):
            if i < len(m):
                m.pop(i)
    elif operation == "keep_layers":
        m = [current_map[i] for i in params["layer_indices"] if i < len(current_map)]
    elif operation == "swap_layers":
        i, j = params["i"], params["j"]
        if i < len(m) and j < len(m):
            m[i], m[j] = m[j], m[i]
    elif operation == "reorder_layers":
        m = [current_map[i] for i in params["new_order"] if i < len(current_map)]
    elif operation == "duplicate_layer":
        src, dst = params["src"], params["dst"]
        if src < len(current_map):
            m.insert(dst, current_map[src])
    return m

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
        model.eval()
        model.requires_grad_(False)
        num_layers = model.config.num_hidden_layers
        info = SessionInfo(
            name=name, model=model, tokenizer=tokenizer,
            model_id=model_id, mode=mode,
            _layer_map=list(range(num_layers)),
            _original_config=copy.deepcopy(model.config),
        )
        self._sessions[name] = info
        log.info("Registered session '%s' (model=%s, mode=%s, layers=%d)",
                 name, model_id, mode, num_layers)
        return info

    def get(self, name: str) -> SessionInfo:
        if name not in self._sessions:
            raise KeyError(f"Session '{name}' not found")
        return self._sessions[name]

    def list_sessions(self) -> list:
        return list(self._sessions.values())

    def device_of(self, name: str) -> str:
        info = self.get(name)
        return str(next(info.model.parameters()).device)

    def _is_dispatch_model(self, info) -> bool:
        return hasattr(info.model, "hf_device_map")

    def to_cpu(self, name: str) -> None:
        info = self.get(name)
        if next(info.model.parameters()).device.type == "cpu":
            return
        if self._is_dispatch_model(info):
            return
        log.info("Moving session '%s' to CPU", name)
        info.model = info.model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_gpu(self, name: str, device: str = "cuda:0") -> None:
        info = self.get(name)
        if self._is_dispatch_model(info):
            return
        log.info("Moving session '%s' to %s", name, device)
        info.model = info.model.to(device)

    def ensure_on_gpu(self, name: str) -> None:
        info = self.get(name)
        if next(info.model.parameters()).device.type == "cuda":
            return
        if self._is_dispatch_model(info):
            return
        if torch.cuda.is_available():
            model_bytes = sum(p.nelement() * p.element_size() for p in info.model.parameters())
            free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            log.debug("GPU check for '%s': need %.0fMB, free %.0fMB",
                      name, model_bytes / 1e6, free / 1e6)
            if free < model_bytes * 1.3:
                for other_name, other in self._sessions.items():
                    if other_name != name and not self._is_dispatch_model(other) and next(other.model.parameters()).device.type == "cuda":
                        log.info("Evicting '%s' to CPU to make room for '%s'", other_name, name)
                        self.to_cpu(other_name)
                        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                        if free >= model_bytes * 1.3:
                            break
        try:
            info.model = info.model.to("cuda:0")
            log.info("Session '%s' moved to GPU", name)
        except RuntimeError as e:
            log.error("Failed to move '%s' to GPU: %s", name, e)

    def delete(self, name: str) -> None:
        info = self.get(name)
        log.info("Deleting session '%s'", name)
        del self._sessions[name]
        del info.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

