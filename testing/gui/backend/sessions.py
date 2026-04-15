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

    @property
    def num_original_layers(self) -> int:
        return self._original_config.num_hidden_layers

    def stage_op(self, operation: str, params: dict) -> dict:
        validate_original_indices(operation, params, self.num_original_layers)
        entry = {"operation": operation, "params": params}
        self._pending_ops.append(entry)
        return entry

    def undo_op(self) -> dict:
        if not self._pending_ops:
            raise ValueError("No pending operations to undo")
        return self._pending_ops.pop()

    def delete_op(self, index: int) -> dict:
        if index < 0 or index >= len(self._pending_ops):
            raise IndexError(f"Pending op index {index} out of range [0, {len(self._pending_ops)})")
        return self._pending_ops.pop(index)

    def build_manifest(self) -> "SurgeryManifest":
        from .manifest import SurgeryManifest
        return SurgeryManifest(
            model_id=self.model_id,
            original_num_layers=self.num_original_layers,
            applied_ops=list(self._applied_ops),
            layer_map=list(self._layer_map),
        )

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

def translate_to_current(operation: str, params: dict, layer_map: list[int]) -> dict:
    """Translate original-index params to current-index params for surgery execution.

    All layer indices in staged ops reference the *original* model topology.
    This function converts them to current indices using the layer map
    (where layer_map[current_idx] = original_idx).
    """
    translated = dict(params)

    def _current(original: int) -> int:
        try:
            return layer_map.index(original)
        except ValueError:
            raise ValueError(
                f"Original layer {original} not found in current model "
                f"(current layer map: {layer_map})"
            )

    if operation in ("remove_layers", "keep_layers"):
        translated["layer_indices"] = [_current(o) for o in params["layer_indices"]]
    elif operation in ("zero_heads", "scale_heads", "zero_mlp", "zero_attention", "swap_heads"):
        translated["layer"] = _current(params["layer"])
    elif operation == "swap_layers":
        translated["i"] = _current(params["i"])
        translated["j"] = _current(params["j"])
    elif operation == "duplicate_layer":
        translated["src"] = _current(params["src"])
    elif operation == "reorder_layers":
        translated["new_order"] = [_current(o) for o in params["new_order"]]

    return translated


def validate_original_indices(operation: str, params: dict, num_original_layers: int) -> None:
    """Validate that all layer indices are valid original model indices."""
    def _check(idx: int):
        if idx < 0 or idx >= num_original_layers:
            raise ValueError(
                f"Layer index {idx} out of range [0, {num_original_layers})"
            )

    if operation in ("remove_layers", "keep_layers"):
        for idx in params["layer_indices"]:
            _check(idx)
    elif operation in ("zero_heads", "scale_heads", "zero_mlp", "zero_attention", "swap_heads"):
        _check(params["layer"])
    elif operation == "swap_layers":
        _check(params["i"])
        _check(params["j"])
    elif operation == "duplicate_layer":
        _check(params["src"])
    elif operation == "reorder_layers":
        for idx in params["new_order"]:
            _check(idx)


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

