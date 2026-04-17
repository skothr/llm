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

    try:
        mgr.ensure_pytorch(name)
    except Exception as e:
        log.exception("WS logit-lens: ensure_pytorch failed for '%s'", name)
        await _send_json(ws, {"type": "error", "message": f"Failed to load PyTorch model: {e}"})
        await ws.close()
        return

    try:
        raw = await ws.receive_text()
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        await _send_json(ws, {"type": "error", "message": f"Invalid JSON config: {e}"})
        await ws.close()
        return
    except WebSocketDisconnect:
        return
    prompt = config.get("prompt", "")
    top_k = config.get("top_k", 10)
    log.debug("WS logit-lens config: prompt=%r, top_k=%d", prompt[:80], top_k)

    connected = True

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

    from llm_surgeon import probe

    loop = asyncio.get_running_loop()

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
            "metrics": data.get("metrics", []),
        }
        fut = asyncio.run_coroutine_threadsafe(_send_json(ws, msg), loop)
        try:
            ok = fut.result(timeout=10)
        except Exception:
            ok = False
        if not ok:
            connected = False

    # Hint to the client that we may be waiting on a re-export in another
    # handler. Racy (dirty may flip before we acquire the lock) but harmless:
    # worst case is a spurious status message.
    if info.dirty:
        await _send_json(ws, {"type": "status", "message": "Waiting for model export..."})

    try:
        async with info.lock:
            result = await loop.run_in_executor(
                None,
                lambda: probe.logit_lens(
                    info.model, info.tokenizer, prompt,
                    top_k=top_k, on_layer=on_layer,
                ),
            )

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


@router.websocket("/sessions/{name}/compare-logit-lens")
async def compare_logit_lens_ws(ws: WebSocket, name: str):
    """Exact A/B logit-lens comparison. Config must include `with_session` (the
    B-side session name). Streams one frame per aligned (original_layer, sublayer)
    with per-position A/B top-k, per-side metrics, and pairwise KL/JS/cosine/delta.
    """
    await ws.accept()
    log.info("WS compare-logit-lens connected (A='%s')", name)
    mgr = get_manager()

    try:
        raw = await ws.receive_text()
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        await _send_json(ws, {"type": "error", "message": f"Invalid JSON config: {e}"})
        await ws.close()
        return
    except WebSocketDisconnect:
        return

    name_b = config.get("with_session")
    if not name_b or not isinstance(name_b, str):
        await _send_json(ws, {"type": "error", "message": "config.with_session is required"})
        await ws.close()
        return
    if name_b == name:
        await _send_json(ws, {"type": "error", "message": "with_session must differ from the primary session"})
        await ws.close()
        return

    prompt = config.get("prompt", "")
    top_k = config.get("top_k", 10)

    try:
        info_a = mgr.get(name)
        info_b = mgr.get(name_b)
    except KeyError as e:
        await _send_json(ws, {"type": "error", "message": f"Session not found: {e}"})
        await ws.close()
        return

    try:
        mgr.ensure_pytorch(name)
        mgr.ensure_pytorch(name_b)
    except Exception as e:
        log.exception("WS compare-logit-lens: ensure_pytorch failed")
        await _send_json(ws, {"type": "error", "message": f"Failed to load PyTorch model: {e}"})
        await ws.close()
        return

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from llm_surgeon import probe

    connected = True
    loop = asyncio.get_running_loop()

    def on_layer(orig_layer, sublayer, data):
        nonlocal connected
        if not connected:
            return
        msg = {
            "type": "data",
            "original_layer": orig_layer,
            "sublayer": sublayer,
            "cells": data.get("cells", []),
        }
        fut = asyncio.run_coroutine_threadsafe(_send_json(ws, msg), loop)
        try:
            ok = fut.result(timeout=10)
        except Exception:
            ok = False
        if not ok:
            connected = False

    # Any dirty session needs its pytorch model to be up-to-date. ensure_pytorch
    # already rebuilt weights from the manifest, so dirty only signals a GGUF
    # re-export is pending — which doesn't affect logit lens (PyTorch path).

    # Order locks by session name to prevent deadlock with any other handler
    # that might also hold two session locks.
    first, second = sorted([(name, info_a), (name_b, info_b)], key=lambda x: x[0])

    try:
        async with first[1].lock:
            async with second[1].lock:
                result = await loop.run_in_executor(
                    None,
                    lambda: probe.compare_logit_lens(
                        info_a.model, info_b.model, info_a.tokenizer, prompt,
                        top_k=top_k, on_layer=on_layer,
                        layer_map_a=list(info_a._layer_map),
                        layer_map_b=list(info_b._layer_map),
                    ),
                )

        if connected:
            await _send_json(ws, {
                "type": "complete",
                "summary": {
                    "prompt_tokens": result.prompt_tokens,
                    "aligned_keys": [list(k) for k in result.aligned_keys],
                    "num_aligned": len(result.aligned_keys),
                },
            })

    except Exception as e:
        log.exception("WS compare-logit-lens error")
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        log.info("WS compare-logit-lens disconnected (A='%s', B='%s')", name, name_b)
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

    was_on_gpu_before = mgr.is_on_gpu(name)
    try:
        # ensure_on_gpu moves tensors (sync torch); offload to keep the loop hot.
        await asyncio.get_event_loop().run_in_executor(None, lambda: mgr.ensure_on_gpu(name))
    except Exception as e:
        log.error("WS generate: GPU error for '%s': %s", name, e)
        await _send_json(ws, {"type": "error", "message": f"GPU error: {e}"})
        await ws.close()
        return

    try:
        raw = await ws.receive_text()
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        await _send_json(ws, {"type": "error", "message": f"Invalid JSON config: {e}"})
        await ws.close()
        return
    except WebSocketDisconnect:
        return
    log.debug("WS generate config: %s", {k: v for k, v in config.items() if k != "prompt"})
    prompt = config.get("prompt", "")
    max_tokens = config.get("max_tokens", 256)
    temperature = config.get("temperature", 1.0)
    prob_top_k = config.get("prob_top_k", 10)
    repetition_penalty = max(0.01, float(config.get("repetition_penalty", 1.0)))
    stop_sequences = config.get("stop_sequences", [])

    connected = True
    generated_tokens = []
    stop_reason = None

    if info.llama is not None:
        import numpy as np
        from ..sessions import ensure_fresh_gguf

        # Re-export (if dirty) happens under the lock; streaming then runs
        # on a captured engine reference so surgery/revert can't yank it.
        try:
            async with info.lock:
                if info.dirty and info.model is not None:
                    await _send_json(ws, {"type": "status", "message": "Re-exporting modified model to GGUF..."})
                    await ensure_fresh_gguf(info)
                engine = info.llama
        except Exception as e:
            log.exception("WS generate: re-export failed for '%s'", name)
            await _send_json(ws, {"type": "error", "message": f"Re-export failed: {e}"})
            try:
                await ws.close()
            except RuntimeError:
                pass
            return

        try:
            tokens = engine.tokenize(prompt)
            step_num = 0
            for step in engine.generate(
                tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                stop_sequences=stop_sequences,
                emit_logits=(prob_top_k > 0),
            ):
                if not connected:
                    break
                generated_tokens.append(step.token_str)

                top_k_list = []
                if step.logits is not None:
                    probs = np.exp(step.logits - step.logits.max())
                    probs /= probs.sum()
                    top_idx = np.argsort(probs)[-prob_top_k:][::-1]
                    top_k_list = [
                        {"token": engine.detokenize([int(i)]), "prob": float(probs[i])}
                        for i in top_idx
                    ]

                msg = {
                    "type": "data",
                    "step": step_num,
                    "token": step.token_str,
                    "token_id": step.token_id,
                    "top_k": top_k_list,
                    "engine": "llama.cpp",
                }
                if not await _send_json(ws, msg):
                    connected = False
                    break

                gen_text = "".join(generated_tokens)
                if stop_sequences and any(s in gen_text for s in stop_sequences):
                    stop_reason = "stop_sequence"
                    break

                step_num += 1

            if connected:
                stop_reason = stop_reason or "max_tokens"
                await _send_json(ws, {
                    "type": "complete",
                    "generated_text": "".join(generated_tokens),
                    "num_tokens": len(generated_tokens),
                    "stop_reason": stop_reason,
                    "engine": "llama.cpp",
                })
        except Exception as e:
            log.exception("WS generate error via llama.cpp (session='%s')", name)
            await _send_json(ws, {"type": "error", "message": str(e)})
        finally:
            try:
                await ws.close()
            except RuntimeError:
                pass
        return

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
            loop = asyncio.get_event_loop()
            inputs = info.tokenizer(prompt, return_tensors="pt")
            device = next(info.model.parameters()).device
            input_ids = inputs["input_ids"].to(device)
            all_token_ids = input_ids[0].tolist()
            prev_text = info.tokenizer.decode(all_token_ids, skip_special_tokens=True)
            past_key_values = None

            for step in range(max_tokens):
                if not connected:
                    break

                try:
                    def _forward(ids=input_ids, pkv=past_key_values):
                        with torch.no_grad():
                            return info.model(ids, past_key_values=pkv, use_cache=True)
                    outputs = await loop.run_in_executor(None, _forward)
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                except torch.OutOfMemoryError:  # pyright: ignore[reportAttributeAccessIssue]
                    log.warning("OOM during generate on '%s' at step %d — returning partial output", name, step)
                    past_key_values = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    stop_reason = "oom"
                    break

                if repetition_penalty != 1.0:
                    for token_id in set(all_token_ids):
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

                all_token_ids.append(int(next_token[0, 0]))
                new_text = info.tokenizer.decode(all_token_ids, skip_special_tokens=True)
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

                input_ids = next_token

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
        past_key_values = None
        log.info("WS generate disconnected (session='%s', tokens=%d, stop=%s)",
                 name, len(generated_tokens), stop_reason or "disconnect")
        if not was_on_gpu_before:
            try:
                await asyncio.get_event_loop().run_in_executor(None, lambda: mgr.to_cpu(name))
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

    try:
        mgr.ensure_pytorch(name)
    except Exception as e:
        log.exception("WS intervene: ensure_pytorch failed for '%s'", name)
        await _send_json(ws, {"type": "error", "message": f"Failed to load PyTorch model: {e}"})
        await ws.close()
        return

    try:
        raw = await ws.receive_text()
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        await _send_json(ws, {"type": "error", "message": f"Invalid JSON config: {e}"})
        await ws.close()
        return
    except WebSocketDisconnect:
        return
    prompt = config.get("prompt", "")
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

    connected = True
    loop = asyncio.get_running_loop()

    def on_layer(layer_idx, sublayer, data):
        nonlocal connected
        if not connected:
            return
        msg: dict = {
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
        fut = asyncio.run_coroutine_threadsafe(_send_json(ws, msg), loop)
        try:
            ok = fut.result(timeout=10)
        except Exception:
            ok = False
        if not ok:
            connected = False

    if info.dirty:
        await _send_json(ws, {"type": "status", "message": "Waiting for model export..."})

    try:
        async with info.lock:
            result = await loop.run_in_executor(
                None,
                lambda: intervene(
                    info.model, info.tokenizer, prompt,
                    interventions=interventions,
                    capture_logit_lens=capture_logit_lens,
                    on_layer=on_layer,
                ),
            )

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
