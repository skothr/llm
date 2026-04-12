import json
import asyncio
import hashlib
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..sessions import SessionManager
from ..hidden_state_cache import HiddenStateCache

_hs_cache = HiddenStateCache(max_bytes=500_000_000)

router = APIRouter(tags=["probes"])

def get_manager() -> SessionManager:
    from ..app import manager
    return manager

async def _send_json(ws: WebSocket, data: dict) -> bool:
    try:
        await ws.send_text(json.dumps(data, default=str))
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False

@router.websocket("/sessions/{name}/logit-lens")
async def logit_lens_ws(ws: WebSocket, name: str):
    await ws.accept()
    mgr = get_manager()

    try:
        info = mgr.get(name)
    except KeyError:
        await _send_json(ws, {"type": "error", "message": f"Session '{name}' not found"})
        await ws.close()
        return

    raw = await ws.receive_text()
    config = json.loads(raw)
    prompt = config["prompt"]
    top_k = config.get("top_k", 10)

    cancelled = asyncio.Event()
    connected = True

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

    from llm_surgeon import probe

    collected_layers = []

    def on_layer(layer_idx, sublayer, data):
        nonlocal connected
        if cancelled.is_set() or not connected:
            return

        predictions = data.get("top_k", [])
        serializable_preds = []
        for pos_pred in predictions:
            if isinstance(pos_pred, tuple):
                serializable_preds.append([{"token": pos_pred[0], "prob": float(pos_pred[1])}])
            elif isinstance(pos_pred, list):
                serializable_preds.append([
                    {"token": p[0], "prob": float(p[1])} if isinstance(p, tuple)
                    else {"token": p["token"], "prob": float(p["prob"])}
                    for p in pos_pred
                ])
            else:
                serializable_preds.append([{"token": str(pos_pred), "prob": 0.0}])

        msg = {
            "type": "data",
            "layer": layer_idx,
            "sublayer": sublayer,
            "predictions": serializable_preds,
        }
        collected_layers.append(msg)

    try:
        async with info.lock:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: probe.logit_lens(
                    info.model, info.tokenizer, prompt,
                    top_k=top_k, on_layer=on_layer,
                ),
            )

        for msg in collected_layers:
            if not await _send_json(ws, msg):
                connected = False
                break

        if connected and not cancelled.is_set():
            summary = {
                "prompt_tokens": result.prompt_tokens,
                "num_layers": len(result.predictions),
            }
            await _send_json(ws, {"type": "complete", "summary": summary})
        elif cancelled.is_set():
            await _send_json(ws, {"type": "cancelled"})

    except Exception as e:
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass


@router.websocket("/sessions/{name}/generate")
async def generate_ws(ws: WebSocket, name: str):
    await ws.accept()
    mgr = get_manager()

    try:
        info = mgr.get(name)
    except KeyError:
        await _send_json(ws, {"type": "error", "message": f"Session '{name}' not found"})
        await ws.close()
        return

    raw = await ws.receive_text()
    config = json.loads(raw)
    prompt = config["prompt"]
    max_tokens = config.get("max_tokens", 256)
    temperature = config.get("temperature", 1.0)
    prob_top_k = config.get("prob_top_k", 10)

    connected = True
    generated_tokens = []

    try:
        async with info.lock:
            inputs = info.tokenizer(prompt, return_tensors="pt")
            device = next(info.model.parameters()).device
            input_ids = inputs["input_ids"].to(device)

            for step in range(max_tokens):
                if not connected:
                    break

                with torch.no_grad():
                    outputs = info.model(input_ids)
                    logits = outputs.logits[:, -1, :]

                if temperature > 0:
                    logits = logits / temperature

                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs[0], min(prob_top_k, probs.shape[-1]))

                if temperature == 0:
                    next_token = top_indices[0:1].unsqueeze(0)
                else:
                    next_token = torch.multinomial(probs, 1)

                token_str = info.tokenizer.decode(next_token[0])
                generated_tokens.append(token_str)

                top_k_list = [
                    {"token": info.tokenizer.decode(top_indices[i:i+1]), "prob": float(top_probs[i])}
                    for i in range(len(top_indices))
                ]

                msg = {
                    "type": "data",
                    "step": step,
                    "token": token_str,
                    "token_id": int(next_token[0, 0]),
                    "top_k": top_k_list,
                }
                if not await _send_json(ws, msg):
                    connected = False
                    break

                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if next_token[0, 0] == info.tokenizer.eos_token_id:
                    break

        if connected:
            await _send_json(ws, {
                "type": "complete",
                "generated_text": "".join(generated_tokens),
                "num_tokens": len(generated_tokens),
            })

    except Exception as e:
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass


def _resolve_op(op_name: str, params: dict, mgr, cache):
    from llm_surgeon.probe import ops, extract_hidden_states

    simple_ops = {
        "scale": ops.scale,
        "zero_dims": ops.zero_dims,
        "clamp": ops.clamp,
        "noise": ops.noise,
    }

    if op_name in simple_ops:
        return simple_ops[op_name](**params)

    if op_name in ("replace", "project_out"):
        source = params["source"]
        session_name = source["session"]
        source_info = mgr.get(session_name)
        prompt_hash = hashlib.sha256(source["prompt"].encode()).hexdigest()[:16]

        cached = cache.get(session_name, prompt_hash)
        if cached is None:
            hs = extract_hidden_states(
                source_info.model, source_info.tokenizer, source["prompt"],
                layers=[source["layer"]], sublayers=(source["sublayer"],),
            )
            cached = {}
            for key, tensor in hs.states.items():
                cached[f"{key[0]}_{key[1]}"] = tensor
            cache.put(session_name, prompt_hash, cached)

        state_key = f"{source['layer']}_{source['sublayer']}"
        tensor = cached[state_key]
        position = source["position"]
        vec = tensor[position]

        if op_name == "replace":
            return ops.replace(vec)
        else:
            return ops.project_out(vec)

    raise ValueError(f"Unknown op: '{op_name}'")


@router.websocket("/sessions/{name}/intervene")
async def intervene_ws(ws: WebSocket, name: str):
    await ws.accept()
    mgr = get_manager()

    try:
        info = mgr.get(name)
    except KeyError:
        await _send_json(ws, {"type": "error", "message": f"Session '{name}' not found"})
        await ws.close()
        return

    raw = await ws.receive_text()
    config = json.loads(raw)
    prompt = config["prompt"]
    capture_logit_lens = config.get("capture_logit_lens", False)

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

    from llm_surgeon.probe import Intervention, intervene

    try:
        interventions = []
        for spec in config["interventions"]:
            fn = _resolve_op(spec["op"], spec.get("params", {}), mgr, _hs_cache)
            interventions.append(Intervention(
                layer=spec["layer"],
                sublayer=spec["sublayer"],
                fn=fn,
            ))
    except (KeyError, ValueError) as e:
        await _send_json(ws, {"type": "error", "message": f"Invalid intervention: {e}"})
        await ws.close()
        return

    collected = []
    connected = True

    def on_layer(layer_idx, sublayer, data):
        nonlocal connected
        if not connected:
            return
        msg = {
            "type": "data",
            "layer": layer_idx,
            "sublayer": sublayer,
            "modified": data.get("modified", False),
        }
        if data.get("top_k") is not None:
            preds = []
            for pos_pred in data["top_k"]:
                if isinstance(pos_pred, tuple):
                    preds.append([{"token": pos_pred[0], "prob": float(pos_pred[1])}])
                elif isinstance(pos_pred, list):
                    preds.append([
                        {"token": p[0], "prob": float(p[1])} if isinstance(p, tuple)
                        else {"token": p["token"], "prob": float(p["prob"])}
                        for p in pos_pred
                    ])
                else:
                    preds.append([{"token": str(pos_pred), "prob": 0.0}])
            msg["predictions"] = preds
        collected.append(msg)

    try:
        async with info.lock:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: intervene(
                    info.model, info.tokenizer, prompt,
                    interventions=interventions,
                    capture_logit_lens=capture_logit_lens,
                    on_layer=on_layer,
                ),
            )

        for msg in collected:
            if not await _send_json(ws, msg):
                connected = False
                break

        if connected:
            applied = result.interventions_applied or []
            await _send_json(ws, {
                "type": "complete",
                "interventions_applied": len(applied),
            })

    except Exception as e:
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass
