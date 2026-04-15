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
    _layer_map: list = field(default_factory=list, repr=False)
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
        num_layers = model.config.num_hidden_layers
        info = SessionInfo(
            name=name, model=model, tokenizer=tokenizer,
            model_id=model_id, mode=mode,
            _layer_map=list(range(num_layers)),
        )
        self._sessions[name] = info
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
        info.model = info.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_gpu(self, name: str, device: str = "cuda:0") -> None:
        info = self.get(name)
        if self._is_dispatch_model(info):
            return
        info.model = info.model.to(device)

    def ensure_on_gpu(self, name: str) -> None:
        info = self.get(name)
        if next(info.model.parameters()).device.type == "cuda":
            return
        if torch.cuda.is_available():
            model_bytes = sum(p.nelement() * p.element_size() for p in info.model.parameters())
            free = torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)
            if free < model_bytes * 1.3:
                for other in self._sessions.values():
                    if other.name != name and next(other.model.parameters()).device.type == "cuda":
                        other.model = other.model.cpu()
                        torch.cuda.empty_cache()
                        free = torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)
                        if free >= model_bytes * 1.3:
                            break
        info.model = info.model.to("cuda:0" if torch.cuda.is_available() else "cpu")

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
            "layer_map": list(info._layer_map),
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
        info._layer_map = entry.get("layer_map", list(range(entry["config"].num_hidden_layers)))
