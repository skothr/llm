import json
import asyncio
import hashlib
import logging
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..sessions import SessionManager
from ..hidden_state_cache import HiddenStateCache

log = logging.getLogger("gui.backend.routes.probes")

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
    log.info("WS logit-lens connected (session='%s')", name)
    mgr = get_manager()

    try:
        info = mgr.get(name)
    except KeyError:
        log.warning("WS logit-lens: session '%s' not found", name)
        await _send_json(ws, {"type": "error", "message": f"Session '{name}' not found"})
        await ws.close()
        return

    raw = await ws.receive_text()
    config = json.loads(raw)
    prompt = config["prompt"]
    top_k = config.get("top_k", 10)
    log.debug("WS logit-lens config: prompt=%r, top_k=%d", prompt[:80], top_k)

    connected = True

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

    from llm_surgeon import probe

    collected_layers = []

    def on_layer(layer_idx, sublayer, data):
        nonlocal connected
        if not connected:
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
            "original_layer": info._layer_map[layer_idx] if layer_idx < len(info._layer_map) else layer_idx,
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

        if connected:
            summary = {
                "prompt_tokens": result.prompt_tokens,
                "num_layers": len(result.predictions),
            }
            await _send_json(ws, {"type": "complete", "summary": summary})

    except Exception as e:
        log.exception("WS logit-lens error (session='%s')", name)
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        log.info("WS logit-lens disconnected (session='%s')", name)
        try:
            await ws.close()
        except RuntimeError:
            pass


@router.websocket("/sessions/{name}/generate")
async def generate_ws(ws: WebSocket, name: str):
    await ws.accept()
    log.info("WS generate connected (session='%s')", name)
    mgr = get_manager()

    try:
        info = mgr.get(name)
    except KeyError:
        log.warning("WS generate: session '%s' not found", name)
        await _send_json(ws, {"type": "error", "message": f"Session '{name}' not found"})
        await ws.close()
        return

    try:
        mgr.ensure_on_gpu(name)
    except Exception as e:
        log.error("WS generate: GPU error for '%s': %s", name, e)
        await _send_json(ws, {"type": "error", "message": f"GPU error: {e}"})
        await ws.close()
        return

    raw = await ws.receive_text()
    config = json.loads(raw)
    log.debug("WS generate config: %s", {k: v for k, v in config.items() if k != "prompt"})
    prompt = config["prompt"]
    max_tokens = config.get("max_tokens", 256)
    temperature = config.get("temperature", 1.0)
    prob_top_k = config.get("prob_top_k", 10)
    repetition_penalty = config.get("repetition_penalty", 1.0)
    stop_sequences = config.get("stop_sequences", [])

    connected = True
    generated_tokens = []
    stop_reason = None

    def _tok_display(tokenizer, tid):
        """Convert token ID to display string using raw vocab lookup."""
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        if tok is None:
            return ""
        if tok in ("</s>", "<eos>"):
            return "<eos>"
        if tok == "<s>":
            return "<bos>"
        import re as _re
        tok = _re.sub(r"<0x([0-9A-Fa-f]{2})>", lambda m: chr(int(m.group(1), 16)), tok)
        return tok.replace("\u2581", " ")

    try:
        async with info.lock:
            inputs = info.tokenizer(prompt, return_tensors="pt")
            device = next(info.model.parameters()).device
            input_ids = inputs["input_ids"].to(device)
            prev_text = info.tokenizer.decode(input_ids[0], skip_special_tokens=True)

            for step in range(max_tokens):
                if not connected:
                    break

                try:
                    with torch.no_grad():
                        outputs = info.model(input_ids)
                        logits = outputs.logits[:, -1, :]
                except torch.OutOfMemoryError:
                    log.warning("OOM during generate on '%s' at step %d — returning partial output", name, step)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    stop_reason = "oom"
                    break

                if repetition_penalty != 1.0:
                    for token_id in set(input_ids[0].tolist()):
                        if logits[0, token_id] > 0:
                            logits[0, token_id] /= repetition_penalty
                        else:
                            logits[0, token_id] *= repetition_penalty

                if temperature > 0:
                    logits = logits / temperature

                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs[0], min(prob_top_k, probs.shape[-1]))

                if temperature == 0:
                    next_token = top_indices[0:1].unsqueeze(0)
                else:
                    next_token = torch.multinomial(probs, 1)

                if next_token[0, 0] == info.tokenizer.eos_token_id:
                    top_k_list = [
                        {"token": _tok_display(info.tokenizer, top_indices[i]), "prob": float(top_probs[i])}
                        for i in range(len(top_indices))
                    ]
                    msg = {
                        "type": "data",
                        "step": step,
                        "token": "<eos>",
                        "token_id": int(next_token[0, 0]),
                        "top_k": top_k_list,
                    }
                    await _send_json(ws, msg)
                    stop_reason = "eos"
                    break

                input_ids = torch.cat([input_ids, next_token], dim=-1)
                new_text = info.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                token_str = new_text[len(prev_text):]
                if not token_str:
                    token_str = _tok_display(info.tokenizer, next_token[0, 0])
                prev_text = new_text
                generated_tokens.append(token_str)

                top_k_list = [
                    {"token": _tok_display(info.tokenizer, top_indices[i]), "prob": float(top_probs[i])}
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

                gen_text = "".join(generated_tokens)
                if stop_sequences and any(s in gen_text for s in stop_sequences):
                    stop_reason = "stop_sequence"
                    break

        if connected:
            await _send_json(ws, {
                "type": "complete",
                "generated_text": "".join(generated_tokens),
                "num_tokens": len(generated_tokens),
                "stop_reason": stop_reason or "max_tokens",
            })

    except Exception as e:
        log.exception("WS generate error (session='%s')", name)
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        log.info("WS generate disconnected (session='%s', tokens=%d, stop=%s)",
                 name, len(generated_tokens), stop_reason or "disconnect")
        try:
            mgr.to_cpu(name)
        except Exception:
            pass
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
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
    log.info("WS intervene connected (session='%s')", name)
    mgr = get_manager()

    try:
        info = mgr.get(name)
    except KeyError:
        log.warning("WS intervene: session '%s' not found", name)
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
        log.info("WS intervene: %d intervention(s) on '%s'", len(interventions), name)
    except (KeyError, ValueError) as e:
        log.warning("WS intervene: invalid intervention spec: %s", e)
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
        log.exception("WS intervene error (session='%s')", name)
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        log.info("WS intervene disconnected (session='%s')", name)
        try:
            await ws.close()
        except RuntimeError:
            pass
