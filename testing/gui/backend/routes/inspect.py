import asyncio
import sys
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from ..sessions import SessionManager

router = APIRouter(tags=["inspect"])

def get_manager() -> SessionManager:
    from ..app import manager
    return manager

class InfluenceRequest(BaseModel):
    prompts: List[str]

class AttentionRequest(BaseModel):
    prompt: str

class ResidualNormsRequest(BaseModel):
    prompt: str

@router.post("/sessions/{name}/inspect/influence")
async def inspect_influence(name: str, req: InfluenceRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    from llm_surgeon import inspect as insp

    scores = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: insp.block_influence(info.model, info.tokenizer, req.prompts),
    )
    return {"scores": {str(k): float(v) for k, v in scores.items()}}

@router.post("/sessions/{name}/inspect/attention")
async def inspect_attention(name: str, req: AttentionRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    from llm_surgeon import inspect as insp

    entropy = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: insp.attention_entropy(info.model, info.tokenizer, req.prompt),
    )
    return {"entropy": {str(k): v for k, v in entropy.items()}}

@router.post("/sessions/{name}/inspect/residual-norms")
async def inspect_residual_norms(name: str, req: ResidualNormsRequest):
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    from llm_surgeon import inspect as insp

    norms = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: insp.residual_stream_norms(info.model, info.tokenizer, req.prompt),
    )
    return {"norms": [float(n) for n in norms]}
