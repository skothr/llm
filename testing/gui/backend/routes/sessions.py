import asyncio
import logging
import re
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from ..sessions import SessionManager

log = logging.getLogger("gui.backend.routes.sessions")
router = APIRouter(tags=["sessions"])

MODELS_CACHE = Path(__file__).resolve().parent.parent.parent.parent / ".cache" / "models"

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

class LoadRequest(BaseModel):
    name: str
    model_id: str
    mode: str = "inspect"

    @field_validator("name")
    @classmethod
    def validate_slug(cls, v):
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._~\-]{0,63}$", v):
            raise ValueError("Must be alphanumeric, hyphens, underscores, periods, or tildes, 1-64 chars")
        return v

def _session_summary(info) -> dict:
    config = info.model.config
    return SessionSummary(
        name=info.name,
        model_id=info.model_id,
        mode=info.mode,
        num_layers=config.num_hidden_layers,
        pending_count=len(info.pending_ops),
        applied_count=len(info.applied_ops),
        device=str(next(info.model.parameters()).device),
    ).model_dump()

def _session_info(info) -> dict:
    config = info.model.config
    total_params = sum(p.numel() for p in info.model.parameters())
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
        original_num_layers=info._original_config.num_hidden_layers,
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
        mgr.delete(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
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

@router.get("/models/available")
async def list_available_models():
    return [{"model_id": m, "safetensors": _has_safetensors_cached(m)} for m in _scan_model_cache(MODELS_CACHE)]

def _has_safetensors_cached(model_id: str) -> bool:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon.surgery import _has_safetensors
    return _has_safetensors(model_id, str(MODELS_CACHE))

class ConvertRequest(BaseModel):
    model_id: str

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

    pending = info.pending_ops
    applied = []

    for op in pending:
        op_name = op["operation"]
        params = op["params"]
        if op_name not in op_map:
            raise HTTPException(422, f"Unknown operation in queue: '{op_name}'")
        translated = translate_to_current(op_name, params, info._layer_map)
        log.info("Committing op on '%s': %s(%s) [original: %s]", name, op_name, translated, params)
        try:
            op_map[op_name](info.model, translated)
        except (IndexError, ValueError) as e:
            log.warning("Commit failed on '%s' at op %s: %s", name, op_name, e)
            raise HTTPException(422, f"Operation '{op_name}' failed: {e}")
        info._layer_map = update_layer_map(info._layer_map, op_name, translated)
        applied.append(op)

    info.record_applied(applied)
    info.clear_pending()

    return {
        "applied_count": len(applied),
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

    log.info("Reverting session '%s' — reloading clean model", name)
    info.revert()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon import surgery
    from llm_surgeon.surgery import _snapshot_dir

    try:
        if _snapshot_dir(info.model_id):
            model, _ = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: surgery.load_model(info.model_id, mode=info.mode),
            )
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_config(info._original_config)
        model.eval()
        model.requires_grad_(False)
        if not hasattr(model, "hf_device_map"):
            model = model.cpu()
    except Exception as e:
        log.exception("Revert failed for session '%s'", name)
        raise HTTPException(500, f"Revert failed: {e}")

    info.model = model
    info._layer_map = list(range(info._original_config.num_hidden_layers))

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

    try:
        from llm_surgeon.surgery import _snapshot_dir
        if _snapshot_dir(info.model_id):
            mgr.to_cpu(name)
            from llm_surgeon import surgery
            cloned_model, _ = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: surgery.load_model(info.model_id, mode=info.mode),
            )
        else:
            cloned_model = copy.deepcopy(info.model)
            cloned_model.eval()
        if not hasattr(cloned_model, "hf_device_map"):
            cloned_model = cloned_model.cpu()
    except Exception as e:
        raise HTTPException(500, f"Clone failed: {e}")

    try:
        mgr.register(req.target_name, cloned_model, info.tokenizer,
                     model_id=info.model_id, mode=info.mode)
    except ValueError as e:
        del cloned_model
        raise HTTPException(409, str(e))
    return _session_summary(mgr.get(req.target_name))
