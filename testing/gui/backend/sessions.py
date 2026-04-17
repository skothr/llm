import copy
import gc
import logging
import re
import shutil
import tempfile
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from llm_surgeon.llama_engine import LlamaEngine
    from .manifest import SurgeryManifest

log = logging.getLogger("gui.backend.sessions")

@dataclass
class SessionInfo:
    name: str
    model: Any
    tokenizer: Any
    model_id: str
    mode: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _pending_ops: list = field(default_factory=list, repr=False)
    _applied_ops: list = field(default_factory=list, repr=False)
    _op_history: list = field(default_factory=list, repr=False)
    _layer_map: list = field(default_factory=list, repr=False)
    _original_config: Any = field(default=None, repr=False)
    llama: Optional["LlamaEngine"] = field(default=None, repr=False)
    gguf_path: Optional[Path] = field(default=None, repr=False)
    source_gguf_path: Optional[Path] = field(default=None, repr=False)
    # Temp dir we created via re-export; we own cleanup of this one only
    # (source_gguf_path is user-supplied and left alone).
    _owned_export_dir: Optional[Path] = field(default=None, repr=False)
    dirty: bool = field(default=False)

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
        self._check_conflicts(operation, params)
        entry = {"operation": operation, "params": params}
        self._pending_ops.append(entry)
        return entry

    def _check_conflicts(self, operation: str, params: dict) -> None:
        pending_removals: set[int] = set()
        for op in self._pending_ops:
            if op["operation"] == "remove_layers":
                pending_removals.update(op["params"]["layer_indices"])

        if operation == "remove_layers":
            dups = pending_removals & set(params["layer_indices"])
            if dups:
                raise ValueError(
                    f"Layer(s) {sorted(dups)} already staged for removal"
                )

        layer_idx = params.get("layer")
        if layer_idx is not None and layer_idx in pending_removals:
            raise ValueError(
                f"Layer {layer_idx} is staged for removal — "
                f"cannot stage {operation} on it"
            )

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
        # Serializes GPU move operations (ensure_on_gpu / to_cpu / register).
        # Without this, two handlers racing ensure_on_gpu can both decide to
        # evict the same peer, or both try to move onto GPU without accounting
        # for each other's incoming weights. The lock is held only during the
        # move itself (brief), never during generation.
        self._gpu_lock: asyncio.Lock = asyncio.Lock()

    async def ensure_on_gpu_safe(self, name: str) -> None:
        """Coroutine-safe GPU move. Serializes with any other move in flight.

        The caller should already hold ``info.lock`` for ``name``; that
        prevents a peer handler's ensure_on_gpu from evicting us between
        the move and the first forward pass (``ensure_on_gpu`` skips peers
        whose lock is held).
        """
        loop = asyncio.get_running_loop()
        async with self._gpu_lock:
            await loop.run_in_executor(None, lambda: self.ensure_on_gpu(name))

    async def to_cpu_safe(self, name: str) -> None:
        """Coroutine-safe CPU move, serialized against GPU moves."""
        loop = asyncio.get_running_loop()
        async with self._gpu_lock:
            await loop.run_in_executor(None, lambda: self.to_cpu(name))

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

    def register_llama(
        self, name: str, llama_engine, tokenizer, *,
        model_id: str, mode: str, gguf_path,
    ) -> SessionInfo:
        """Register a GGUF session with llama.cpp engine (no PyTorch model yet)."""
        self.validate_name(name)
        if name in self._sessions:
            raise ValueError(f"Session '{name}' already exists")
        info = SessionInfo(
            name=name, model=None, tokenizer=tokenizer,
            model_id=model_id, mode=mode,
            llama=llama_engine,
            gguf_path=gguf_path,
            source_gguf_path=gguf_path,
        )
        self._sessions[name] = info
        log.info("Registered GGUF session '%s' (model=%s, llama.cpp only)", name, model_id)
        return info

    def ensure_pytorch(self, name: str) -> None:
        """Lazy-load the PyTorch model for a GGUF-only session."""
        info = self.get(name)
        if info.model is not None:
            return
        if info.gguf_path is None:
            raise ValueError(f"Session '{name}' has no model and no GGUF path")

        log.info("Lazy-loading PyTorch model for '%s' from %s", name, info.gguf_path)
        from llm_surgeon.surgery import load_model
        model, tokenizer = load_model(info.model_id, mode=info.mode)
        model.eval()
        model.requires_grad_(False)
        info.model = model
        info.tokenizer = tokenizer
        info._layer_map = list(range(model.config.num_hidden_layers))
        info._original_config = copy.deepcopy(model.config)
        log.info("PyTorch model loaded for '%s'", name)

    def get(self, name: str) -> SessionInfo:
        if name not in self._sessions:
            raise KeyError(f"Session '{name}' not found")
        return self._sessions[name]

    def list_sessions(self) -> list:
        return list(self._sessions.values())

    def device_of(self, name: str) -> str:
        info = self.get(name)
        if info.model is None:
            return "llama.cpp" if info.llama is not None else "none"
        return str(next(info.model.parameters()).device)

    def _is_dispatch_model(self, info) -> bool:
        return hasattr(info.model, "hf_device_map")

    @staticmethod
    def _is_bnb_model(info) -> bool:
        try:
            import bitsandbytes as bnb  # pyright: ignore[reportMissingImports]
            return any(isinstance(p, bnb.nn.Params4bit) or isinstance(p, bnb.nn.Int8Params)  # pyright: ignore[reportPrivateImportUsage]
                       for p in info.model.parameters())
        except ImportError:
            return False

    @staticmethod
    def _move_bnb_params(model, device: str) -> None:
        import bitsandbytes as bnb  # pyright: ignore[reportMissingImports]
        for param in model.parameters():
            if isinstance(param, bnb.nn.Params4bit):  # pyright: ignore[reportPrivateImportUsage]
                param.data = param.data.to(device)
                if hasattr(param, "quant_state") and param.quant_state is not None:
                    qs = param.quant_state
                    if hasattr(qs, "absmax") and qs.absmax is not None:
                        qs.absmax = qs.absmax.to(device)
                    if hasattr(qs, "code") and qs.code is not None:
                        qs.code = qs.code.to(device)
            elif isinstance(param, bnb.nn.Int8Params):  # pyright: ignore[reportPrivateImportUsage]
                param.data = param.data.to(device)
                if hasattr(param, "SCB") and param.SCB is not None:
                    param.SCB = param.SCB.to(device)
            else:
                param.data = param.data.to(device)

    def to_cpu(self, name: str) -> None:
        info = self.get(name)
        if info.model is None:
            return
        if next(info.model.parameters()).device.type == "cpu":
            return
        log.info("Moving session '%s' to CPU", name)
        if self._is_bnb_model(info):
            self._move_bnb_params(info.model, "cpu")
        else:
            info.model = info.model.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to_gpu(self, name: str, device: str = "cuda:0") -> None:
        info = self.get(name)
        log.info("Moving session '%s' to %s", name, device)
        if self._is_bnb_model(info):
            self._move_bnb_params(info.model, device)
        else:
            info.model = info.model.to(device)

    def is_on_gpu(self, name: str) -> bool:
        info = self.get(name)
        if info.model is None:
            return False
        try:
            return next(info.model.parameters()).device.type == "cuda"
        except StopIteration:
            return False

    def ensure_on_gpu(self, name: str) -> None:
        info = self.get(name)
        if info.model is None:
            return
        if next(info.model.parameters()).device.type == "cuda":
            return
        if not torch.cuda.is_available():
            return
        model_bytes = sum(p.nelement() * p.element_size() for p in info.model.parameters())
        free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        log.debug("GPU check for '%s': need %.0fMB, free %.0fMB",
                  name, model_bytes / 1e6, free / 1e6)
        if free < model_bytes * 1.3:
            skipped_busy: list[str] = []
            evicted_any = False
            for other_name, other in self._sessions.items():
                if other_name == name or other.model is None:
                    continue
                try:
                    other_device = next(other.model.parameters()).device
                except StopIteration:
                    continue
                if other_device.type != "cuda":
                    continue
                # Skip peers whose lock is held — evicting mid-forward would
                # reassign param.data under running CUDA kernels.
                if other.lock.locked():
                    skipped_busy.append(other_name)
                    continue
                log.info("Evicting '%s' to CPU to make room for '%s'", other_name, name)
                self.to_cpu(other_name)
                evicted_any = True
                free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                if free >= model_bytes * 1.3:
                    break
            if free < model_bytes * 1.3 and not evicted_any and skipped_busy:
                raise RuntimeError(
                    f"All other GPU sessions are busy ({skipped_busy}); "
                    f"cannot free GPU space for '{name}'"
                )
        try:
            if self._is_bnb_model(info):
                self._move_bnb_params(info.model, "cuda:0")
            else:
                info.model = info.model.to("cuda:0")
            log.info("Session '%s' moved to GPU", name)
        except RuntimeError as e:
            log.error("Failed to move '%s' to GPU: %s", name, e)
            if self._is_bnb_model(info):
                self._move_bnb_params(info.model, "cpu")
            else:
                info.model = info.model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            raise

    def delete(self, name: str) -> None:
        info = self.get(name)
        log.info("Deleting session '%s'", name)
        del self._sessions[name]
        if info.llama is not None:
            try:
                info.llama.close()
            except Exception:
                log.exception("llama.close() failed for '%s'", name)
        if info._owned_export_dir is not None:
            shutil.rmtree(info._owned_export_dir, ignore_errors=True)
            info._owned_export_dir = None
        del info.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


async def ensure_fresh_gguf(info: "SessionInfo") -> None:
    """Re-export PyTorch surgery → GGUF and rebuild the llama.cpp engine.

    No-op if the session is not dirty or has no PyTorch model.

    Caller must hold ``info.lock`` — this mutates ``info.llama``,
    ``info.gguf_path``, ``info._owned_export_dir``, ``info.dirty``.

    On failure (export or engine init), the existing engine is left intact
    and the raised exception reaches the caller. The caller may retry or
    fall back to reporting an error to the client.
    """
    if not info.dirty or info.model is None:
        return

    from llm_surgeon.llama_engine import LlamaEngine, export_hf_to_gguf

    export_dir = Path(tempfile.mkdtemp(prefix="llm_surgeon_"))
    export_path = export_dir / "modified.gguf"
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None, lambda: export_hf_to_gguf(info.model, info.tokenizer, export_path)
        )
        new_engine = await loop.run_in_executor(
            None, lambda: LlamaEngine(export_path)
        )
    except Exception:
        shutil.rmtree(export_dir, ignore_errors=True)
        raise

    old_engine = info.llama
    old_owned = info._owned_export_dir
    info.llama = new_engine
    info.gguf_path = export_path
    info._owned_export_dir = export_dir
    info.dirty = False
    if old_engine is not None:
        try:
            old_engine.close()
        except Exception:
            log.exception("old llama.close() failed for '%s'", info.name)
    if old_owned is not None:
        shutil.rmtree(old_owned, ignore_errors=True)

