import asyncio
import json
import logging
import os
import re
import types
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from ..sessions import SessionManager

log = logging.getLogger("gui.backend.routes.sessions")
router = APIRouter(tags=["sessions"])

MODELS_CACHE = Path(__file__).resolve().parent.parent.parent.parent / ".cache" / "models"
OLLAMA_MODELS_DIR = Path(os.environ.get("OLLAMA_MODELS", "/usr/share/ollama/.ollama/models"))


def _hf_cache_dir(model_id: str) -> Path:
    org, name = model_id.split("/", 1)
    return MODELS_CACHE / f"models--{org}--{name}"


def _read_hf_config(model_id: str) -> dict | None:
    cache = _hf_cache_dir(model_id)
    snapshots = cache / "snapshots"
    if not snapshots.exists():
        return None
    for snap in snapshots.iterdir():
        cfg = snap / "config.json"
        if cfg.exists():
            try:
                return json.loads(cfg.read_text())
            except (json.JSONDecodeError, OSError):
                pass
    return None


def _hf_file_size(model_id: str) -> int | None:
    cache = _hf_cache_dir(model_id)
    blobs = cache / "blobs"
    if not blobs.exists():
        return None
    return sum(f.stat().st_size for f in blobs.iterdir() if f.is_file())


def _hf_model_meta(model_id: str) -> dict:
    meta: dict = {"model_id": model_id, "source": "huggingface",
                  "safetensors": _has_safetensors_cached(model_id)}
    cfg = _read_hf_config(model_id)
    if cfg:
        meta["architecture"] = cfg.get("model_type")
        meta["dtype"] = cfg.get("torch_dtype")
        meta["num_layers"] = cfg.get("num_hidden_layers")
        meta["hidden_size"] = cfg.get("hidden_size")
        meta["num_heads"] = cfg.get("num_attention_heads")
        meta["num_kv_heads"] = cfg.get("num_key_value_heads")
        meta["vocab_size"] = cfg.get("vocab_size")
        meta["intermediate_size"] = cfg.get("intermediate_size")
        meta["max_position_embeddings"] = cfg.get("max_position_embeddings")
    size = _hf_file_size(model_id)
    if size is not None:
        meta["file_size_bytes"] = size
    return meta


def _scan_model_cache(cache_dir: Path) -> list:
    if not cache_dir.exists():
        return []
    models = []
    for entry in cache_dir.iterdir():
        if entry.is_dir() and entry.name.startswith("models--"):
            parts = entry.name[len("models--"):].split("--", 1)
            if len(parts) == 2:
                models.append(parts[0] + "/" + parts[1])
    return sorted(models)


def _read_ollama_manifest(models_dir: Path, model_id: str) -> dict | None:
    name, _, tag = model_id.partition(":")
    manifest_path = models_dir / "manifests" / "registry.ollama.ai" / "library" / name / tag
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _read_ollama_config(models_dir: Path, manifest: dict) -> dict | None:
    digest = manifest.get("config", {}).get("digest", "")
    if not digest:
        return None
    blob = models_dir / "blobs" / digest.replace(":", "-")
    if not blob.exists():
        return None
    try:
        return json.loads(blob.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# GGUF header parsing for an Ollama blob is ~120 ms per model on a warm disk
# cache and the blobs are content-addressed (sha256), so metadata can be
# memoized per blob path forever. Invalidating on (size, mtime_ns) is belt-
# and-braces: in practice digest-addressed blobs never mutate.
_gguf_meta_cache: dict[tuple[str, int, int], dict] = {}


def _gguf_meta_cached(blob_path: Path) -> dict:
    try:
        st = blob_path.stat()
    except OSError:
        return {}
    key = (str(blob_path), st.st_size, st.st_mtime_ns)
    cached = _gguf_meta_cache.get(key)
    if cached is not None:
        return cached
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
        from llm_surgeon.gguf_reader import gguf_model_meta
        meta = gguf_model_meta(blob_path) or {}
    except Exception:
        meta = {}
    _gguf_meta_cache[key] = meta
    return meta


def _ollama_model_meta(models_dir: Path, model_id: str) -> dict:
    meta: dict = {"model_id": model_id, "source": "ollama", "safetensors": False}
    manifest = _read_ollama_manifest(models_dir, model_id)
    if not manifest:
        return meta

    blob_path = None
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            meta["file_size_bytes"] = layer["size"]
            digest = layer["digest"].replace("sha256:", "sha256-")
            candidate = models_dir / "blobs" / digest
            if candidate.exists():
                blob_path = candidate
            break

    if blob_path:
        gguf_meta = _gguf_meta_cached(blob_path)
        for k, v in gguf_meta.items():
            if v is not None:
                meta[k] = v

    cfg = _read_ollama_config(models_dir, manifest)
    if cfg:
        meta.setdefault("architecture", cfg.get("model_family"))
        meta.setdefault("quantization", cfg.get("file_type"))
        meta["model_size_label"] = cfg.get("model_type")

    return meta


def _scan_ollama_models(models_dir: Path) -> list:
    manifests = models_dir / "manifests" / "registry.ollama.ai" / "library"
    if not manifests.exists():
        return []
    models = []
    for model_dir in sorted(manifests.iterdir()):
        if not model_dir.is_dir():
            continue
        for tag_file in sorted(model_dir.iterdir()):
            if tag_file.is_file():
                models.append(f"{model_dir.name}:{tag_file.name}")
    return models

def get_manager() -> SessionManager:
    from ..app import manager
    return manager

class SessionSummary(BaseModel):
    name: str
    model_id: str
    mode: str
    num_layers: int
    pending_count: int
    applied_count: int
    device: str
    engine_llama: bool = False
    engine_pytorch: bool = False
    dirty: bool = False

class SessionInfoResponse(BaseModel):
    num_layers: int
    num_heads: int
    num_kv_heads: int | None = None
    hidden_size: int
    intermediate_size: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None
    rope_theta: float | None = None
    total_params: int
    chat_template: str | None = None
    bos_token: str | None = None
    eos_token: str | None = None
    layer_map: list[int] = []
    original_num_layers: int
    pending_ops: list[dict] = []
    applied_ops: list[dict] = []

class SurgeryRequest(BaseModel):
    operation: str
    params: dict

class CloneRequest(BaseModel):
    target_name: str

    @field_validator("target_name")
    @classmethod
    def validate_slug(cls, v):
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._~\-]{0,63}$", v):
            raise ValueError("Must be alphanumeric, hyphens, underscores, periods, or tildes, 1-64 chars")
        return v

_MODEL_ID_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]*(?:/[a-zA-Z0-9][a-zA-Z0-9._-]*)?(?::[a-zA-Z0-9._-]+)?$"
)


def _validate_model_id(v: str) -> str:
    if not isinstance(v, str) or not _MODEL_ID_RE.match(v):
        raise ValueError(
            "model_id must be 'org/name', 'name:tag', or 'org/name:tag' with "
            "alphanumerics, '.', '_', '-' only"
        )
    return v


class LoadRequest(BaseModel):
    name: str
    model_id: str
    mode: str = "nf4"

    @field_validator("name")
    @classmethod
    def validate_slug(cls, v):
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._~\-]{0,63}$", v):
            raise ValueError("Must be alphanumeric, hyphens, underscores, periods, or tildes, 1-64 chars")
        return v

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v):
        return _validate_model_id(v)

def _session_summary(info) -> dict:
    if info.model is not None:
        config = info.model.config
        num_layers = config.num_hidden_layers
        device = str(next(info.model.parameters()).device)
    elif info.gguf_path is not None:
        from llm_surgeon.gguf_reader import gguf_model_meta
        meta = gguf_model_meta(info.gguf_path)
        num_layers = meta.get("num_layers", 0)
        device = "llama.cpp"
    else:
        num_layers = 0
        device = "none"
    return SessionSummary(
        name=info.name,
        model_id=info.model_id,
        mode=info.mode,
        num_layers=num_layers,
        pending_count=len(info.pending_ops),
        applied_count=len(info.applied_ops),
        device=device,
        engine_llama=info.llama is not None,
        engine_pytorch=info.model is not None,
        dirty=info.dirty,
    ).model_dump()

def _session_info(info) -> dict:
    if info.model is not None:
        config = info.model.config
        total_params = sum(p.numel() for p in info.model.parameters())
    elif info.gguf_path is not None:
        from llm_surgeon.gguf_reader import gguf_model_meta
        meta = gguf_model_meta(info.gguf_path)
        config = types.SimpleNamespace(
            num_hidden_layers=meta.get("num_layers", 0),
            num_attention_heads=meta.get("num_heads", 0),
            num_key_value_heads=meta.get("num_kv_heads"),
            hidden_size=meta.get("hidden_size", 0),
            intermediate_size=meta.get("intermediate_size"),
            vocab_size=meta.get("vocab_size"),
            max_position_embeddings=meta.get("max_position_embeddings"),
            rope_theta=meta.get("rope_theta"),
        )
        total_params = meta.get("total_params", 0)
    else:
        return {}
    chat_template = None
    bos_token = None
    eos_token = None
    if info.tokenizer is not None:
        chat_template = getattr(info.tokenizer, "chat_template", None)
        bos_token = getattr(info.tokenizer, "bos_token", None)
        eos_token = getattr(info.tokenizer, "eos_token", None)
    return SessionInfoResponse(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        num_kv_heads=getattr(config, "num_key_value_heads", None),
        hidden_size=config.hidden_size,
        intermediate_size=getattr(config, "intermediate_size", None),
        vocab_size=getattr(config, "vocab_size", None),
        max_position_embeddings=getattr(config, "max_position_embeddings", None),
        rope_theta=getattr(config, "rope_theta", None),
        total_params=total_params,
        chat_template=chat_template,
        bos_token=bos_token,
        eos_token=eos_token,
        layer_map=list(info._layer_map),
        original_num_layers=info._original_config.num_hidden_layers if info._original_config is not None else config.num_hidden_layers,
        pending_ops=info.pending_ops,
        applied_ops=info.applied_ops,
    ).model_dump()

@router.get("/sessions", response_model=List[dict])
async def list_sessions():
    mgr = get_manager()
    return [_session_summary(s) for s in mgr.list_sessions()]

@router.get("/sessions/{name}/info")
async def get_session_info(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    return _session_info(info)

@router.delete("/sessions/{name}")
async def delete_session(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    # Hold info.lock so any in-flight WS handler (generate/logit-lens/intervene)
    # finishes before we tear down info.model. Without this, `del info.model`
    # would race with a concurrent forward pass and crash the WS handler.
    async with info.lock:
        try:
            mgr.delete(name)
        except KeyError:
            raise HTTPException(404, f"Session '{name}' not found")
    # Drop any cached hidden states keyed to this session so memory is freed
    # and a later session with the same name doesn't read stale tensors.
    from .probes import _hs_cache
    _hs_cache.invalidate_session(name)
    return {"deleted": name}

@router.post("/sessions")
async def load_session(req: LoadRequest):
    mgr = get_manager()
    try:
        mgr.validate_name(req.name)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if req.name in mgr._sessions:
        raise HTTPException(409, f"Session '{req.name}' already exists")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

    from llm_surgeon.surgery import _is_ollama_id
    if _is_ollama_id(req.model_id):
        from llm_surgeon.gguf_reader import resolve_ollama_blob, _build_tokenizer, GGUFFile
        from llm_surgeon.llama_engine import LlamaEngine
        blob = resolve_ollama_blob(req.model_id)
        if blob is not None:
            try:
                llama_eng = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: LlamaEngine(blob)
                )
                with GGUFFile(blob) as g:
                    tokenizer = _build_tokenizer(g.metadata)
                mgr.register_llama(
                    req.name, llama_eng, tokenizer,
                    model_id=req.model_id, mode=req.mode,
                    gguf_path=blob,
                )
                return _session_summary(mgr.get(req.name))
            except Exception as e:
                log.exception("Failed to load GGUF model '%s' via llama.cpp", req.model_id)
                raise HTTPException(500, f"Failed to load GGUF model: {e}")

    from llm_surgeon import surgery

    log.info("Loading model '%s' as session '%s' (mode=%s)", req.model_id, req.name, req.mode)
    try:
        model, tokenizer = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: surgery.load_model(req.model_id, mode=req.mode),
        )
    except Exception as e:
        log.exception("Failed to load model '%s'", req.model_id)
        raise HTTPException(500, f"Failed to load model: {e}")

    mgr.register(req.name, model, tokenizer,
                 model_id=req.model_id, mode=req.mode)
    return _session_summary(mgr.get(req.name))

SURGERY_OPS = [
    {"name": "remove_layers", "params": {"layer_indices": {"type": "array", "items": "int"}}, "description": "Remove specific layers"},
    {"name": "keep_layers", "params": {"layer_indices": {"type": "array", "items": "int"}}, "description": "Keep only specified layers"},
    {"name": "zero_heads", "params": {"layer": "int", "heads": {"type": "array", "items": "int"}}, "description": "Zero out attention heads"},
    {"name": "scale_heads", "params": {"layer": "int", "heads": {"type": "array", "items": "int"}, "factor": "float"}, "description": "Scale attention head contributions"},
    {"name": "swap_layers", "params": {"i": "int", "j": "int"}, "description": "Swap two layers"},
    {"name": "duplicate_layer", "params": {"src": "int", "dst": "int"}, "description": "Deep-copy layer to new position"},
    {"name": "zero_mlp", "params": {"layer": "int"}, "description": "Zero out MLP contribution"},
    {"name": "zero_attention", "params": {"layer": "int"}, "description": "Zero out attention contribution"},
    {"name": "swap_heads", "params": {"layer": "int", "h1": "int", "h2": "int"}, "description": "Swap two attention heads"},
    {"name": "reorder_layers", "params": {"new_order": {"type": "array", "items": "int"}}, "description": "Rearrange layer order"},
]

@router.get("/surgery/operations")
async def surgery_operations():
    return SURGERY_OPS

def _collect_available_models() -> list[dict]:
    hf = [_hf_model_meta(m) for m in _scan_model_cache(MODELS_CACHE)]
    ollama = [_ollama_model_meta(OLLAMA_MODELS_DIR, m)
              for m in _scan_ollama_models(OLLAMA_MODELS_DIR)]
    return hf + ollama


@router.get("/models/available")
async def list_available_models():
    # Scan does sync I/O (stat + GGUF header reads on cold cache); keep the
    # event loop responsive by moving it off the main thread. Cheap on warm
    # cache — skipping run_in_executor would save ~0.3 ms per call.
    return await asyncio.get_event_loop().run_in_executor(None, _collect_available_models)

def _has_safetensors_cached(model_id: str) -> bool:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon.surgery import _has_safetensors
    return _has_safetensors(model_id, str(MODELS_CACHE))

class ConvertRequest(BaseModel):
    model_id: str

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v):
        return _validate_model_id(v)

@router.post("/models/convert-safetensors")
async def convert_model_safetensors(req: ConvertRequest):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon.surgery import convert_to_safetensors

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: convert_to_safetensors(req.model_id, str(MODELS_CACHE)),
        )
    except (ValueError, Exception) as e:
        log.exception("Safetensors conversion failed for '%s'", req.model_id)
        raise HTTPException(500, str(e))
    return result

VALID_OPS = {op["name"] for op in SURGERY_OPS}

@router.post("/sessions/{name}/surgery")
async def stage_surgery(name: str, req: SurgeryRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    if req.operation not in VALID_OPS:
        raise HTTPException(422, f"Unknown surgery operation: '{req.operation}'")

    log.info("Staging op on '%s': %s(%s)", name, req.operation, req.params)
    try:
        info.stage_op(req.operation, req.params)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"pending": info.pending_ops}

@router.delete("/sessions/{name}/surgery/last")
async def undo_staged_op(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    try:
        removed = info.undo_op()
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"removed": removed, "pending": info.pending_ops}

@router.delete("/sessions/{name}/surgery/{index}")
async def delete_staged_op(name: str, index: int):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    try:
        removed = info.delete_op(index)
    except IndexError as e:
        raise HTTPException(409, str(e))
    return {"removed": removed, "pending": info.pending_ops}

@router.get("/sessions/{name}/surgery/pending")
async def get_pending_ops(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    return {"pending": info.pending_ops}

@router.post("/sessions/{name}/surgery/commit")
async def commit_surgery(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    if not info.has_pending:
        raise HTTPException(409, "No pending operations to commit")

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon import surgery
    from ..sessions import update_layer_map, translate_to_current

    op_map = {
        "remove_layers": lambda m, p: surgery.remove_layers(m, p["layer_indices"]),
        "keep_layers": lambda m, p: surgery.keep_layers(m, p["layer_indices"]),
        "zero_heads": lambda m, p: surgery.zero_heads(m, p["layer"], p["heads"]),
        "scale_heads": lambda m, p: surgery.scale_heads(m, p["layer"], p["heads"], p["factor"]),
        "swap_layers": lambda m, p: surgery.swap_layers(m, p["i"], p["j"]),
        "duplicate_layer": lambda m, p: surgery.duplicate_layer(m, p["src"], p["dst"]),
        "zero_mlp": lambda m, p: surgery.zero_mlp(m, p["layer"]),
        "zero_attention": lambda m, p: surgery.zero_attention(m, p["layer"]),
        "swap_heads": lambda m, p: surgery.swap_heads(m, p["layer"], p["h1"], p["h2"]),
        "reorder_layers": lambda m, p: surgery.reorder_layers(m, p["new_order"]),
    }

    async with info.lock:
        # Advance ops one-at-a-time from pending → applied so a failure mid-way
        # leaves pending_ops/applied_ops/layer_map/model all mutually consistent.
        # The user can fix the offending op and retry commit on whatever remains.
        applied_count = 0
        loop = asyncio.get_event_loop()
        while info._pending_ops:
            op = info._pending_ops[0]
            op_name = op["operation"]
            params = op["params"]
            if op_name not in op_map:
                raise HTTPException(422, f"Unknown operation in queue: '{op_name}'")
            translated = translate_to_current(op_name, params, info._layer_map)
            log.info("Committing op on '%s': %s(%s) [original: %s]", name, op_name, translated, params)
            try:
                # Surgery ops are synchronous torch — offload to the default
                # executor so we don't stall the event loop while holding the lock.
                op_fn = op_map[op_name]
                await loop.run_in_executor(
                    None, lambda f=op_fn, t=translated: f(info.model, t)
                )
            except (IndexError, ValueError) as e:
                log.warning("Commit failed on '%s' at op %s: %s", name, op_name, e)
                raise HTTPException(422, f"Operation '{op_name}' failed: {e}")
            info._layer_map = update_layer_map(info._layer_map, op_name, translated)
            info._applied_ops.append(op)
            info._pending_ops.pop(0)
            applied_count += 1

        if applied_count > 0 and info.llama is not None:
            info.dirty = True

    return {
        "applied_count": applied_count,
        "pending": info.pending_ops,
        "info": _session_info(info),
    }

@router.post("/sessions/{name}/surgery/revert")
async def revert_surgery(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    if not info.applied_ops:
        raise HTTPException(409, "No applied operations to revert")

    import sys, gc
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon import surgery
    from llm_surgeon.surgery import _snapshot_dir
    import shutil
    import torch as _torch

    async with info.lock:
        log.info("Reverting session '%s' — reloading clean model", name)

        # Build the replacement model BEFORE mutating op lists, so a load
        # failure doesn't leave pending/applied/history in a half-reverted
        # state that can't be recovered.
        loop = asyncio.get_event_loop()
        try:
            if _snapshot_dir(info.model_id):
                new_model, _ = await loop.run_in_executor(
                    None,
                    lambda: surgery.load_model(info.model_id, mode=info.mode),
                )
            else:
                from transformers import AutoModelForCausalLM
                new_model = await loop.run_in_executor(
                    None,
                    lambda: AutoModelForCausalLM.from_config(info._original_config),
                )

            def _finalize_new_model(m):
                m.eval()
                m.requires_grad_(False)
                if not hasattr(m, "hf_device_map"):
                    m = m.cpu()
                return m
            new_model = await loop.run_in_executor(
                None, lambda: _finalize_new_model(new_model)
            )
        except Exception as e:
            log.exception("Revert failed for session '%s'", name)
            raise HTTPException(500, f"Revert failed: {e}")

        # Now atomically swap: mutate op lists, drop the old model, install the new.
        info.revert()
        old_model = info.model
        info.model = new_model
        info._layer_map = list(range(info._original_config.num_hidden_layers))
        del old_model
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        if info.llama is not None and info.gguf_path != info.source_gguf_path:
            from llm_surgeon.llama_engine import LlamaEngine
            old_engine = info.llama
            old_owned = info._owned_export_dir
            assert info.source_gguf_path is not None
            src = info.source_gguf_path
            try:
                info.llama = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: LlamaEngine(src)
                )
            except Exception as e:
                log.exception("Revert: reopening source GGUF failed for '%s'", name)
                raise HTTPException(500, f"Revert failed to reload source GGUF: {e}")
            # Only close/cleanup old resources after the new engine is live.
            try:
                old_engine.close()
            except Exception:
                log.exception("old llama.close() failed for '%s'", name)
            if old_owned is not None:
                shutil.rmtree(old_owned, ignore_errors=True)
                info._owned_export_dir = None
            info.gguf_path = info.source_gguf_path
        info.dirty = False

    return {
        "pending": info.pending_ops,
        "info": _session_info(info),
    }

@router.get("/sessions/{name}/surgery/history")
async def get_op_history(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    return {"history": info.op_history}

class ValidateRequest(BaseModel):
    prompt: str = "The capital of France is"
    top_k: int = 10

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("top_k must be > 0")
        return v


@router.post("/sessions/{name}/validate")
async def validate_session(name: str, req: ValidateRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    if info.llama is None:
        raise HTTPException(409, "Session has no llama.cpp engine (not a GGUF model)")
    if info.model is None:
        raise HTTPException(409, "PyTorch model not loaded yet — run logit lens first to trigger load")

    import numpy as np
    import torch
    from llm_surgeon.llama_engine import compare_logits
    from ..sessions import ensure_fresh_gguf

    async with info.lock:
        # Auto re-export if PyTorch has uncommitted-to-GGUF surgery, so both
        # engines reflect the same weights before we compare.
        if info.dirty:
            log.info("Validate: re-exporting dirty session '%s' before compare", name)
            await ensure_fresh_gguf(info)

        engine = info.llama
        tokens = engine.tokenize(req.prompt)
        native_logits = await asyncio.get_event_loop().run_in_executor(
            None, lambda: engine.logits(tokens)
        )

        was_on_gpu = mgr.is_on_gpu(name)
        loop = asyncio.get_event_loop()
        # ensure_on_gpu moves tensors to device — synchronous torch, offload.
        await loop.run_in_executor(None, lambda: mgr.ensure_on_gpu(name))
        try:
            def _pytorch_forward():
                input_ids = torch.tensor([tokens], dtype=torch.long).to(
                    next(info.model.parameters()).device
                )
                with torch.no_grad():
                    return info.model(input_ids=input_ids).logits[0, -1].float().cpu().numpy()
            pytorch_logits = await loop.run_in_executor(None, _pytorch_forward)
        finally:
            if not was_on_gpu:
                try:
                    await loop.run_in_executor(None, lambda: mgr.to_cpu(name))
                except Exception:
                    log.exception("validate: to_cpu failed for '%s'", name)

    result = compare_logits(native_logits, pytorch_logits, top_k=req.top_k)

    def _top_k_tokens(logits, k):
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        top_idx = np.argsort(logits)[-k:][::-1]
        return [
            {"token": engine.detokenize([int(i)]), "logit": float(logits[i]), "prob": float(probs[i])}
            for i in top_idx
        ]

    result["native_top_k"] = _top_k_tokens(native_logits, req.top_k)
    result["pytorch_top_k"] = _top_k_tokens(pytorch_logits, req.top_k)

    return result

@router.post("/sessions/{name}/clone")
async def clone_session(name: str, req: CloneRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    log.info("Cloning session '%s' -> '%s'", name, req.target_name)

    import copy, sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

    # Remember whether the original was on GPU so we can move it back after
    # briefly evicting it for the clone's fresh load.
    was_on_gpu = False
    try:
        if info.model is not None:
            was_on_gpu = next(info.model.parameters()).device.type == "cuda"
    except StopIteration:
        was_on_gpu = False

    loop = asyncio.get_event_loop()
    try:
        from llm_surgeon.surgery import _snapshot_dir
        if _snapshot_dir(info.model_id):
            await loop.run_in_executor(None, lambda: mgr.to_cpu(name))
            from llm_surgeon import surgery
            cloned_model, _ = await loop.run_in_executor(
                None,
                lambda: surgery.load_model(info.model_id, mode=info.mode),
            )
        else:
            def _deepcopy_and_eval():
                m = copy.deepcopy(info.model)
                m.eval()
                return m
            cloned_model = await loop.run_in_executor(None, _deepcopy_and_eval)
        if not hasattr(cloned_model, "hf_device_map"):
            cloned_model = await loop.run_in_executor(None, lambda: cloned_model.cpu())
    except Exception as e:
        raise HTTPException(500, f"Clone failed: {e}")
    finally:
        if was_on_gpu:
            try:
                await loop.run_in_executor(None, lambda: mgr.ensure_on_gpu(name))
            except Exception:
                log.exception("Clone: failed to restore '%s' to GPU", name)

    try:
        mgr.register(req.target_name, cloned_model, info.tokenizer,
                     model_id=info.model_id, mode=info.mode)
    except ValueError as e:
        del cloned_model
        raise HTTPException(409, str(e))
    return _session_summary(mgr.get(req.target_name))
