"""Capture endpoints — run a forward pass and serve per-layer inspection data.

POST /api/sessions/{name}/capture
    body: {prompt: str}
    Returns {prompt_hash, layers, heads, seq_len}. Idempotent on same prompt.

GET /api/sessions/{name}/capture/{hash}/tokens
    Returns {tokens: [str, ...]}

GET /api/sessions/{name}/capture/{hash}/attention?layer=L&head=H
    Returns {matrix: [[float, ...], ...]}  (seq x seq, post-softmax)

GET /api/sessions/{name}/capture/{hash}/residual?layer=L
    Returns ResidualSummary fields as floats.

GET /api/sessions/{name}/capture/{hash}/qkv?layer=L&head=H&token=T
    Returns QKVStats fields.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..hidden_state_cache import HiddenStateCache
from ..sessions import SessionManager

log = logging.getLogger("gui.backend.routes.capture")

router = APIRouter(tags=["capture"])

# One shared cache across the capture module.  Imported from probes
# so the same cache backing store is used by both modules — the capture
# route populates it, intervene / patching routes read residual tensors.
# We import it lazily inside the handler to avoid a circular import at
# module load time (probes imports sessions, sessions imports app, …).
def _hs_cache() -> HiddenStateCache:
    from .probes import _hs_cache as c
    return c


def get_manager() -> SessionManager:
    from ..app import manager
    return manager


# ── Per-session capture metadata (prompt_hash → meta dict) ───────────────────
# Stores the shape data and cached computed artefacts (attention matrices,
# QKV stats) for the most-recently captured prompt per session.  Keyed by
# (session_name, prompt_hash).  Lightweight — actual tensor bytes live in
# the shared HiddenStateCache.
_capture_meta: dict[tuple[str, str], dict[str, Any]] = {}


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _attn_cache_key(layer: int, head: int) -> str:
    return f"attn_{layer}_{head}"


def _qkv_cache_key(layer: int, head: int, token: int) -> str:
    return f"qkv_{layer}_{head}_{token}"


def _compute_residual_summary(
    residual_pre: torch.Tensor,   # (seq_len, d_model)
    residual_post: torch.Tensor,  # (seq_len, d_model)
    attn_out: torch.Tensor | None,  # (seq_len, d_model) or None
    mlp_out: torch.Tensor | None,   # (seq_len, d_model) or None
) -> dict[str, float]:
    """Compute ResidualSummary metrics from raw tensors.

    All computations are on CPU in fp32.
    """
    pre  = residual_pre.to(dtype=torch.float32, device="cpu")
    post = residual_post.to(dtype=torch.float32, device="cpu")

    attn_out_norm = (
        float(attn_out.to(dtype=torch.float32, device="cpu").norm(p=2, dim=-1).mean().item())
        if attn_out is not None else float("nan")
    )
    mlp_out_norm = (
        float(mlp_out.to(dtype=torch.float32, device="cpu").norm(p=2, dim=-1).mean().item())
        if mlp_out is not None else float("nan")
    )
    resid_norm  = float(post.norm(p=2, dim=-1).mean().item())
    cos_prev    = float(torch.nn.functional.cosine_similarity(pre, post, dim=-1).mean().item())

    # Kurtosis of the post-residual stream (mean over positions, per-dim)
    post_flat = post.reshape(-1)
    mean = post_flat.mean()
    std  = post_flat.std() + 1e-8
    kurtosis = float(((((post_flat - mean) / std) ** 4).mean() - 3.0).item())

    # Effective rank via entropy of squared singular-value spectrum.
    # Use first position's vector to keep cost O(d_model) instead of
    # O(seq * d_model * d_model).  Full-rank estimate uses shape[1].
    try:
        svd_vec = post[post.shape[0] // 2]  # middle position, d_model
        # Approximate: use the per-dimension magnitude distribution.
        magnitudes = svd_vec.abs()
        magnitudes_sum = magnitudes.sum() + 1e-10
        probs = magnitudes / magnitudes_sum
        probs = probs.clamp(min=1e-10)
        entropy = -float((probs * probs.log()).sum().item())
        rank_eff  = max(1, round(math.exp(entropy)))
        rank_full = post.shape[1]
    except Exception:
        rank_eff  = -1
        rank_full = -1

    return {
        "attn_out_norm": attn_out_norm,
        "mlp_out_norm":  mlp_out_norm,
        "resid_norm":    resid_norm,
        "cos_prev":      cos_prev,
        "kurtosis":      kurtosis,
        "rank_eff":      rank_eff,
        "rank_full":     rank_full,
    }


def _run_capture(model: Any, tokenizer: Any, prompt: str) -> dict[str, Any]:
    """Run a forward pass capturing attention weights and residual states.

    Returns a flat dict of tensors and metadata, intended for storage in
    the HiddenStateCache under the prompt_hash key.  Keys:

        "tokens"           : list[str] — string tokens
        "token_ids"        : list[int]
        "attn_{L}_{H}"     : torch.Tensor (seq, seq) — post-softmax attention
        "resid_pre_{L}"    : torch.Tensor (seq, d_model)
        "resid_post_{L}"   : torch.Tensor (seq, d_model)
        "attn_out_{L}"     : torch.Tensor (seq, d_model)
        "mlp_out_{L}"      : torch.Tensor (seq, d_model)  [not always available]
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    token_strs: list[str] = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_ids_list: list[int] = input_ids[0].tolist()

    num_layers: int = len(model.model.layers)
    num_heads: int = model.config.num_attention_heads

    result: dict[str, Any] = {
        "tokens":    token_strs,
        "token_ids": token_ids_list,
        "num_layers": num_layers,
        "num_heads":  num_heads,
        "seq_len":    len(token_strs),
    }

    # Hooks to capture residual stream states.
    # Keys: resid_pre_{L}, resid_post_{L}, attn_out_{L}, mlp_out_{L}
    hooks: list = []
    layer_inputs: dict[int, torch.Tensor]  = {}   # pre-attn residual
    attn_outs:    dict[int, torch.Tensor]  = {}   # raw attn output (delta)
    mlp_outs:     dict[int, torch.Tensor]  = {}   # mlp output (delta)
    layer_outputs: dict[int, torch.Tensor] = {}   # post-layer residual

    def make_block_pre_hook(idx: int):
        def hook(_mod, args: tuple) -> None:
            layer_inputs[idx] = args[0].detach()
        return hook

    def make_attn_out_hook(idx: int):
        def hook(_mod, _inp, out: Any) -> None:
            # self_attn returns (attn_out_tensor, ...) or just a tensor
            t = out[0] if isinstance(out, tuple) else out
            attn_outs[idx] = t.detach()
        return hook

    def make_mlp_hook(idx: int):
        def hook(_mod, _inp, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            mlp_outs[idx] = t.detach()
        return hook

    def make_block_hook(idx: int):
        def hook(_mod, _inp, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            layer_outputs[idx] = t.detach()
        return hook

    for i in range(num_layers):
        layer = model.model.layers[i]
        hooks.append(layer.register_forward_pre_hook(make_block_pre_hook(i)))
        hooks.append(layer.self_attn.register_forward_hook(make_attn_out_hook(i)))
        if hasattr(layer, "mlp"):
            hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(i)))
        hooks.append(layer.register_forward_hook(make_block_hook(i)))

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True)
    finally:
        for h in hooks:
            h.remove()

    # output_attentions=True adds attentions to model output.
    # shape: tuple of (batch, num_heads, seq, seq) per layer
    attention_all = out.attentions  # tuple[Tensor(1,H,S,S), ...] or None

    # Package residual states.
    for i in range(num_layers):
        pre  = layer_inputs.get(i)
        post = layer_outputs.get(i)
        attn = attn_outs.get(i)
        mlp  = mlp_outs.get(i)

        if pre is not None:
            result[f"resid_pre_{i}"]  = pre[0].cpu()   # (seq, d)
        if post is not None:
            result[f"resid_post_{i}"] = post[0].cpu()
        if attn is not None:
            result[f"attn_out_{i}"]   = attn[0].cpu()
        if mlp is not None:
            result[f"mlp_out_{i}"]    = mlp[0].cpu()

    # Package attention matrices from output_attentions.
    if attention_all is not None:
        for layer_idx, attn_tensor in enumerate(attention_all):
            # attn_tensor: (1, num_heads, seq, seq)
            attn_2d = attn_tensor[0]  # (num_heads, seq, seq)
            for h in range(attn_2d.shape[0]):
                result[_attn_cache_key(layer_idx, h)] = attn_2d[h].cpu()
    else:
        # Model doesn't support output_attentions — no attention matrices.
        log.warning("model did not return attentions; attention endpoints will be empty")

    return result


class CaptureRequest(BaseModel):
    prompt: str


@router.post("/sessions/{name}/capture")
async def run_capture(name: str, req: CaptureRequest) -> dict:
    """Run a forward pass for prompt, populate cache, return shape metadata.

    Idempotent: if the same prompt is already cached, returns immediately.
    """
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    if info.model is None:
        raise HTTPException(409, "Session has no PyTorch model loaded (required for capture)")
    if info.tokenizer is None:
        raise HTTPException(409, "Session has no tokenizer loaded")

    phash = _prompt_hash(req.prompt)
    cache = _hs_cache()

    # Check if already cached (idempotent).
    cached = cache.get(name, phash)
    if cached is not None and "num_layers" in cached:
        meta = _capture_meta.get((name, phash), {})
        return {
            "prompt_hash": phash,
            "layers":  cached.get("num_layers", -1),
            "heads":   cached.get("num_heads",  -1),
            "seq_len": cached.get("seq_len",    -1),
            **{k: v for k, v in meta.items() if k not in ("prompt",)},
        }

    # Run capture under the session lock to avoid racing with surgery/generate.
    async with info.lock:
        # Re-check after acquiring lock — another request may have filled it.
        cached = cache.get(name, phash)
        if cached is not None and "num_layers" in cached:
            return {
                "prompt_hash": phash,
                "layers":  cached.get("num_layers", -1),
                "heads":   cached.get("num_heads",  -1),
                "seq_len": cached.get("seq_len",    -1),
            }

        model = info.model
        tokenizer = info.tokenizer
        loop = asyncio.get_running_loop()
        try:
            data: dict = await loop.run_in_executor(
                None, lambda: _run_capture(model, tokenizer, req.prompt)
            )
        except Exception as e:
            log.exception("Capture failed for session '%s'", name)
            raise HTTPException(500, f"Capture failed: {e}")

    # Store tensor-valued entries under the prompt hash; keep metadata dict.
    tensor_data: dict[str, torch.Tensor] = {
        k: v for k, v in data.items() if isinstance(v, torch.Tensor)
    }
    meta: dict[str, Any] = {
        k: v for k, v in data.items() if not isinstance(v, torch.Tensor)
    }
    cache.put(name, phash, tensor_data)
    _capture_meta[(name, phash)] = meta

    return {
        "prompt_hash": phash,
        "layers":      data.get("num_layers", -1),
        "heads":       data.get("num_heads",  -1),
        "seq_len":     data.get("seq_len",    -1),
    }


@router.get("/sessions/{name}/capture/{hash}/tokens")
async def get_capture_tokens(name: str, hash: str) -> dict:
    """Return the token strings for a previously captured prompt."""
    mgr = get_manager()
    try:
        mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    meta = _capture_meta.get((name, hash))
    if meta is None:
        raise HTTPException(404, f"No capture found for hash '{hash}' in session '{name}'")

    return {"tokens": meta.get("tokens", [])}


@router.get("/sessions/{name}/capture/{hash}/attention")
async def get_capture_attention(
    name: str, hash: str, layer: int, head: int
) -> dict:
    """Return the post-softmax attention matrix [seq x seq] for (layer, head).

    The first call fetches from the tensor cache; result is a JSON array.
    """
    mgr = get_manager()
    try:
        mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    cache = _hs_cache()
    data = cache.get(name, hash)
    if data is None:
        raise HTTPException(
            404,
            f"No capture found for hash '{hash}'. Run POST /api/sessions/{name}/capture first.",
        )

    key = _attn_cache_key(layer, head)
    if key not in data:
        raise HTTPException(
            404,
            f"Attention not captured for layer={layer} head={head}. "
            "Model may not support output_attentions.",
        )

    matrix_t: torch.Tensor = data[key]  # (seq, seq)
    matrix: list[list[float]] = matrix_t.float().tolist()
    return {"matrix": matrix}


@router.get("/sessions/{name}/capture/{hash}/residual")
async def get_capture_residual(name: str, hash: str, layer: int) -> dict:
    """Return ResidualSummary metrics for a layer."""
    mgr = get_manager()
    try:
        mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    cache = _hs_cache()
    data = cache.get(name, hash)
    if data is None:
        raise HTTPException(
            404,
            f"No capture found for hash '{hash}'. Run POST /api/sessions/{name}/capture first.",
        )

    pre_key  = f"resid_pre_{layer}"
    post_key = f"resid_post_{layer}"
    if pre_key not in data or post_key not in data:
        raise HTTPException(404, f"Residual not captured for layer={layer}")

    pre       = data[pre_key]
    post      = data[post_key]
    attn_out  = data.get(f"attn_out_{layer}")
    mlp_out   = data.get(f"mlp_out_{layer}")

    summary = _compute_residual_summary(pre, post, attn_out, mlp_out)
    return summary


@router.get("/sessions/{name}/capture/{hash}/qkv")
async def get_capture_qkv(
    name: str, hash: str, layer: int, head: int, token: int
) -> dict:
    """Return QKVStats for (layer, head, token position).

    Computes norms from per-head Q/K/V projections sliced from the attention
    output, and reads attention-to-BOS / self / prev from the cached attention
    matrix.
    """
    mgr = get_manager()
    try:
        info = mgr.get(name)
    except KeyError:
        raise HTTPException(404, f"Session '{name}' not found")

    cache = _hs_cache()
    data  = cache.get(name, hash)
    if data is None:
        raise HTTPException(
            404,
            f"No capture found for hash '{hash}'. Run POST /api/sessions/{name}/capture first.",
        )

    meta = _capture_meta.get((name, hash), {})
    seq_len = meta.get("seq_len", 0)

    if not 0 <= token < seq_len:
        raise HTTPException(
            400,
            f"token position {token} out of range [0, {seq_len})",
        )

    # Compute Q/K/V norms via the model's projection weights + captured residual.
    q_norm = float("nan")
    k_norm = float("nan")
    v_norm = float("nan")

    if info.model is not None and f"resid_pre_{layer}" in data:
        try:
            model = info.model
            num_heads: int = model.config.num_attention_heads
            d_model: int = model.config.hidden_size
            d_head: int = d_model // num_heads

            resid_pre = data[f"resid_pre_{layer}"]  # (seq, d_model)
            vec = resid_pre[token].float()           # (d_model,)

            with torch.no_grad():
                layer_mod = model.model.layers[layer].self_attn
                # Q/K/V projections — LLaMA uses q_proj, k_proj, v_proj
                # (may be fused in some architectures; fall back gracefully).
                if hasattr(layer_mod, "q_proj"):
                    W_q = layer_mod.q_proj.weight.float()  # (d_model, d_model)
                    W_k = layer_mod.k_proj.weight.float()
                    W_v = layer_mod.v_proj.weight.float()
                    q_vec = (W_q @ vec)[head * d_head : (head + 1) * d_head]
                    k_vec = (W_k @ vec)[head * d_head : (head + 1) * d_head]
                    v_vec = (W_v @ vec)[head * d_head : (head + 1) * d_head]
                    q_norm = float(q_vec.norm(p=2).item())
                    k_norm = float(k_vec.norm(p=2).item())
                    v_norm = float(v_vec.norm(p=2).item())
        except Exception as exc:
            log.warning("QKV norm computation failed for L=%d H=%d T=%d: %s", layer, head, token, exc)

    # Read attention weights for BOS/self/prev from the cached attention matrix.
    attn_to_bos  = float("nan")
    attn_to_self = float("nan")
    attn_to_prev = float("nan")

    attn_key = _attn_cache_key(layer, head)
    if attn_key in data:
        attn_mat: torch.Tensor = data[attn_key].float()  # (seq, seq)
        row = attn_mat[token]                             # (seq,) — distribution over keys
        attn_to_bos  = float(row[0].item())
        attn_to_self = float(row[token].item())
        attn_to_prev = float(row[max(0, token - 1)].item()) if token > 0 else float("nan")

    return {
        "q_norm":       q_norm,
        "k_norm":       k_norm,
        "v_norm":       v_norm,
        "attn_to_bos":  attn_to_bos,
        "attn_to_self": attn_to_self,
        "attn_to_prev": attn_to_prev,
    }
