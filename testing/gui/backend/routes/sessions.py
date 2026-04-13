import asyncio
import re
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from ..sessions import SessionManager

router = APIRouter(tags=["sessions"])

def get_manager() -> SessionManager:
    from ..app import manager
    return manager

class SessionSummary(BaseModel):
    name: str
    model_id: str
    mode: str
    num_layers: int
    has_snapshot: bool
    snapshot_size_mb: float

class SessionInfoResponse(BaseModel):
    num_layers: int
    num_heads: int
    hidden_size: int
    total_params: int

class SurgeryRequest(BaseModel):
    operation: str
    params: dict

class SurgeryResponse(BaseModel):
    operations: list
    info: SessionInfoResponse

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
        has_snapshot=info.has_snapshot,
        snapshot_size_mb=info.snapshot_size_mb,
    ).model_dump()

def _session_info(info) -> dict:
    config = info.model.config
    total_params = sum(p.numel() for p in info.model.parameters())
    return SessionInfoResponse(
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        hidden_size=config.hidden_size,
        total_params=total_params,
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
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon import surgery

    try:
        model, tokenizer = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: surgery.load_model(req.model_id, mode=req.mode),
        )
    except Exception as e:
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

@router.post("/sessions/{name}/surgery")
async def apply_surgery(name: str, req: SurgeryRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon import surgery

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

    if req.operation not in op_map:
        raise HTTPException(422, f"Unknown surgery operation: '{req.operation}'")

    mgr.snapshot(name)

    try:
        log = op_map[req.operation](info.model, req.params)
    except (IndexError, ValueError) as e:
        mgr.undo(name)
        raise HTTPException(422, str(e))

    return SurgeryResponse(
        operations=[{"operation": op.operation, "description": op.description} for op in log.ops],
        info=SessionInfoResponse(**_session_info(info)),
    ).model_dump()

@router.post("/sessions/{name}/surgery/undo")
async def undo_surgery(name: str):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")
    try:
        mgr.undo(name)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {"undone": name, "info": _session_info(info)}

@router.post("/sessions/{name}/clone")
async def clone_session(name: str, req: CloneRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    import copy
    try:
        cloned_model = copy.deepcopy(info.model)
    except Exception as e:
        raise HTTPException(500, f"Clone failed: {e}")

    cloned_model.eval()

    mgr.register(req.target_name, cloned_model, info.tokenizer,
                 model_id=info.model_id, mode=info.mode)
    return _session_summary(mgr.get(req.target_name))
