import base64
import json
import asyncio
import hashlib
import logging
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect


def _encode_hidden_state(tensor: torch.Tensor) -> dict:
    """Serialize a (seq_len, hidden_size) tensor as base64-encoded float32 bytes.

    Used to ship hidden states to the frontend for click-to-pin heatmap rendering.
    Base64 float32 keeps the payload ~4/3 × raw bytes vs ~15× for JSON float arrays.
    """
    arr = tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()
    return {
        "shape": list(arr.shape),
        "b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }

from ..sessions import SessionManager
from ..hidden_state_cache import HiddenStateCache
from llm_surgeon.llama_engine import _sample as _numpy_sample

log = logging.getLogger("gui.backend.routes.probes")

_hs_cache = HiddenStateCache(max_bytes=500_000_000)

router = APIRouter(tags=["probes"])


def _auto_pick_ap_pair(
    clean_logits: torch.Tensor, corrupted_logits: torch.Tensor,
) -> tuple[int, int]:
    """Pick (correct, incorrect) token IDs that maximize the divergence
    between the two baseline distributions at the measurement position.

    The naive approach — clean.argmax() vs corrupted.argmax() — collapses
    when both prompts produce the same top-1 token (common for small
    models on weak factual tasks like "The capital of France is" vs
    "...Italy is" — both can predict " a" or another generic filler).
    With identical correct == incorrect, every logit_diff is zero and
    AP's recovery metric divides by zero.

    Diff-based picking is the canonical AP framing (Heimersheim & Nanda
    2024): "correct" is the token clean uniquely promotes vs corrupted,
    "incorrect" is the token corrupted uniquely promotes vs clean. This
    is well-defined whenever the two logit vectors aren't bit-identical,
    which is essentially always for distinct prompts.

    Raises ValueError if the two logit vectors are exactly equal (would
    still hit divide-by-zero — the caller should surface a clear error
    instructing the user to fill in correct/incorrect manually).
    """
    diff = clean_logits - corrupted_logits
    if not (diff.max() > 0 or diff.min() < 0):
        raise ValueError(
            "clean and corrupted produce identical logits at the "
            "measurement position; cannot auto-pick a contrastive pair. "
            "Either choose more-distinct prompts or fill in correct/"
            "incorrect tokens manually."
        )
    return int(diff.argmax().item()), int(diff.argmin().item())


def _check_stop(
    accumulated: str,
    new_chunk: str,
    stops: list[str],
) -> tuple[str, bool, str]:
    """Decide what portion of ``new_chunk`` to stream given stop sequences.

    Mirrors EOS semantics: the stop sequence itself never appears in the
    returned visible text, so stops are lossless markers rather than tokens
    that leak into output.

    Returns ``(visible, stop_hit, matched)``:
      * ``visible``: substring of new_chunk to emit (empty if the stop starts
        inside already-accumulated text, e.g. when a multi-token stop only
        completes now).
      * ``stop_hit``: True if any stop matched in accumulated+new_chunk.
      * ``matched``: the stop string that fired (earliest wins).
    """
    if not stops:
        return (new_chunk, False, "")
    combined = accumulated + new_chunk
    earliest = -1
    matched = ""
    for s in stops:
        idx = combined.find(s)
        if idx < 0:
            continue
        if earliest < 0 or idx < earliest:
            earliest = idx
            matched = s
    if earliest < 0:
        return (new_chunk, False, "")
    start = len(accumulated)
    visible = combined[start:earliest] if earliest > start else ""
    return (visible, True, matched)

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
            "original_layer": info.original_layer(layer_idx),
            "sublayer": sublayer,
            "predictions": serializable_preds,
            "metrics": data.get("metrics", []),
        }
        hs = data.get("hidden_state")
        if isinstance(hs, torch.Tensor):
            msg["hidden_state"] = _encode_hidden_state(hs)
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
        hs_a = data.get("hidden_state_a")
        hs_b = data.get("hidden_state_b")
        if isinstance(hs_a, torch.Tensor):
            msg["hidden_state_a"] = _encode_hidden_state(hs_a)
        if isinstance(hs_b, torch.Tensor):
            msg["hidden_state_b"] = _encode_hidden_state(hs_b)
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
    # display_top_k: how many token probs to stream per step (UI/display only).
    # prob_top_k kept as a legacy alias so older clients keep working.
    display_top_k = int(config.get("display_top_k", config.get("prob_top_k", 10)))
    # sampling_top_k, top_p, min_p: actual sampler cutoffs. Applied in
    # temperature>0 branch only (greedy ignores them).
    sampling_top_k = int(config.get("sampling_top_k", 0))
    top_p = float(config.get("top_p", 1.0))
    min_p = float(config.get("min_p", 0.0))
    seed_cfg = config.get("seed")
    seed = int(seed_cfg) if seed_cfg is not None else None
    repetition_penalty = max(0.01, float(config.get("repetition_penalty", 1.0)))
    stop_sequences = [s for s in config.get("stop_sequences", []) if s]

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
            accumulated = ""
            matched_stop = ""
            # We pass stop_sequences=None to the engine and own truncation here
            # so both engines share identical stop-at-match semantics.
            for step in engine.generate(
                tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=sampling_top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                stop_sequences=None,
                emit_logits=(display_top_k > 0),
                seed=seed,
            ):
                if not connected:
                    break

                visible, hit, matched = _check_stop(accumulated, step.token_str, stop_sequences)

                top_k_list = []
                if step.logits is not None and display_top_k > 0:
                    # Match the PyTorch branch: apply temperature before softmax
                    # so the displayed probabilities agree across engines for
                    # A/B compare. At temperature == 0 the column collapses to a
                    # one-hot argmax — we skip the scaling to avoid inf.
                    if temperature > 0:
                        scaled = step.logits / temperature
                    else:
                        scaled = step.logits
                    probs = np.exp(scaled - scaled.max())
                    probs /= probs.sum()
                    top_idx = np.argsort(probs)[-display_top_k:][::-1]
                    top_k_list = [
                        {"token": engine.detokenize([int(i)]), "prob": float(probs[i])}
                        for i in top_idx
                    ]

                if visible:
                    generated_tokens.append(visible)
                    accumulated += visible
                    msg = {
                        "type": "data",
                        "step": step_num,
                        "token": visible,
                        "token_id": step.token_id,
                        "top_k": top_k_list,
                        "engine": "llama.cpp",
                    }
                    if not await _send_json(ws, msg):
                        connected = False
                        break
                    step_num += 1

                if hit:
                    matched_stop = matched
                    await _send_json(ws, {
                        "type": "data",
                        "step": step_num,
                        "token": "<stop>",
                        "token_id": step.token_id,
                        "top_k": top_k_list,
                        "engine": "llama.cpp",
                        "stop_match": matched,
                    })
                    stop_reason = "stop_sequence"
                    break

            if connected:
                stop_reason = stop_reason or "max_tokens"
                complete_msg: dict = {
                    "type": "complete",
                    "generated_text": accumulated,
                    "num_tokens": len(generated_tokens),
                    "stop_reason": stop_reason,
                    "engine": "llama.cpp",
                }
                if matched_stop:
                    complete_msg["stop_match"] = matched_stop
                await _send_json(ws, complete_msg)
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
        import numpy as np
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        matched_stop = ""
        gen_accumulated = ""

        async with info.lock:
            # ensure_on_gpu must happen inside info.lock so a peer handler
            # cannot evict us between the move and our first forward pass.
            # The manager's gpu lock (inside ensure_on_gpu_safe) additionally
            # serializes concurrent GPU moves from other handlers.
            try:
                await mgr.ensure_on_gpu_safe(name)
            except Exception as e:
                log.error("WS generate: GPU error for '%s': %s", name, e)
                await _send_json(ws, {"type": "error", "message": f"GPU error: {e}"})
                return
            loop = asyncio.get_running_loop()
            inputs = info.tokenizer(prompt, return_tensors="pt")
            device = next(info.model.parameters()).device
            input_ids = inputs["input_ids"].to(device)
            all_token_ids = input_ids[0].tolist()
            # Track generated tokens separately so the per-step decode is O(k)
            # over generated length, not O(n) over prompt+generated. A 1000-
            # token prompt + 256 max_tokens was paying ~290k decode ops under
            # the old all_token_ids slice; now it's ~33k.
            prompt_token_count = len(all_token_ids)
            prev_gen_text = ""
            past_key_values = None
            # Decoded form of the last prompt token. Used as the slicing
            # baseline so the first generated token retains its leading
            # SentencePiece space (see the per-step decode comment below).
            # Falls back to "" when the prompt is empty — in that case the
            # decode buffer starts at position 0 and the leading-space
            # quirk doesn't apply (no prior token to bridge from).
            boundary_text = (
                info.tokenizer.decode(
                    all_token_ids[prompt_token_count - 1:prompt_token_count],
                    skip_special_tokens=True,
                )
                if prompt_token_count > 0
                else ""
            )

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
                except (torch.OutOfMemoryError, RuntimeError) as e:
                    # torch.OutOfMemoryError catches the common CUDA path. Some
                    # backends (MPS, older CPU allocators, XPU) surface OOM as
                    # a plain RuntimeError with "out of memory" in the message
                    # — promote those to the same handler so the user sees a
                    # graceful partial output rather than a 500.
                    if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
                        raise
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

                # Scaled logits feed both display (post-temp softmax) and the
                # sampler; llama.cpp applies temperature before any filters, so
                # matching its order here keeps the two engines comparable.
                if temperature > 0:
                    scaled = logits / temperature
                else:
                    scaled = logits
                probs_disp = torch.softmax(scaled, dim=-1)
                display_k = min(display_top_k, probs_disp.shape[-1]) if display_top_k > 0 else 0
                if display_k > 0:
                    top_probs, top_indices = torch.topk(probs_disp[0], display_k)
                else:
                    top_probs = torch.empty(0, device=probs_disp.device)
                    top_indices = torch.empty(0, dtype=torch.long, device=probs_disp.device)

                if temperature == 0:
                    next_id_int = int(torch.argmax(logits[0]).item())
                else:
                    # Route sampling through the shared numpy helper so top_k /
                    # top_p / min_p behave identically across the llama.cpp and
                    # PyTorch backends. Cost is tiny: one vocab-sized CPU copy.
                    logits_np = logits[0].detach().to(dtype=torch.float32, device="cpu").numpy()
                    next_id_int = _numpy_sample(
                        logits_np,
                        temperature=temperature,
                        top_k=sampling_top_k,
                        top_p=top_p,
                        min_p=min_p,
                        rng=rng,
                    )
                next_token = torch.tensor([[next_id_int]], device=logits.device, dtype=torch.long)

                top_k_list = [
                    {"token": _tok_display(info.tokenizer, top_indices[i]), "prob": float(top_probs[i])}
                    for i in range(len(top_indices))
                ]

                if next_id_int == info.tokenizer.eos_token_id:
                    msg = {
                        "type": "data",
                        "step": step,
                        "token": "<eos>",
                        "token_id": next_id_int,
                        "top_k": top_k_list,
                    }
                    await _send_json(ws, msg)
                    stop_reason = "eos"
                    break

                all_token_ids.append(next_id_int)
                # Decode `[last_prompt_token, *gen_tokens]` rather than
                # the bare gen-tokens slice, then strip the boundary
                # token's own decoded prefix. SentencePiece-based
                # tokenizers (Llama family) drop the leading ▁ marker
                # whenever decode() runs on a buffer that begins at
                # position 0 — without this, the very first generated
                # token loses its leading space, so " Paris" arrives as
                # "Paris" and the panel displays "isParis" once
                # concatenated with the prompt. Including one prompt
                # token of context costs O(1) extra per step and keeps
                # the slicing baseline stable across the loop.
                decode_start = prompt_token_count - 1 if prompt_token_count > 0 else 0
                new_gen_text = info.tokenizer.decode(
                    all_token_ids[decode_start:],
                    skip_special_tokens=True,
                )[len(boundary_text):]
                token_str = new_gen_text[len(prev_gen_text):]
                if not token_str:
                    token_str = _tok_display(info.tokenizer, next_token[0, 0])
                prev_gen_text = new_gen_text

                visible, hit, matched = _check_stop(gen_accumulated, token_str, stop_sequences)

                if visible:
                    generated_tokens.append(visible)
                    gen_accumulated += visible
                    msg = {
                        "type": "data",
                        "step": step,
                        "token": visible,
                        "token_id": next_id_int,
                        "top_k": top_k_list,
                    }
                    if not await _send_json(ws, msg):
                        connected = False
                        break

                if hit:
                    matched_stop = matched
                    await _send_json(ws, {
                        "type": "data",
                        "step": step,
                        "token": "<stop>",
                        "token_id": next_id_int,
                        "top_k": top_k_list,
                        "stop_match": matched,
                    })
                    stop_reason = "stop_sequence"
                    break

                input_ids = next_token

        if connected:
            complete_msg: dict = {
                "type": "complete",
                "generated_text": gen_accumulated,
                "num_tokens": len(generated_tokens),
                "stop_reason": stop_reason or "max_tokens",
            }
            if matched_stop:
                complete_msg["stop_match"] = matched_stop
            await _send_json(ws, complete_msg)

    except Exception as e:
        log.exception("WS generate error (session='%s')", name)
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        past_key_values = None
        log.info("WS generate disconnected (session='%s', tokens=%d, stop=%s)",
                 name, len(generated_tokens), stop_reason or "disconnect")
        if not was_on_gpu_before:
            try:
                await mgr.to_cpu_safe(name)
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


@router.websocket("/sessions/{name}/activation-patching")
async def activation_patching_ws(ws: WebSocket, name: str):
    """Streaming activation-patching: one frame per (layer, sublayer, position) cell.

    Config (first JSON message):
      {
        "clean_prompt": str,
        "corrupted_prompt": str,
        "direction": "denoise" | "noise",     # default "denoise"
        "measurement_position": int,           # default -1
        "positions": [int] | null,             # default null (= all)
        "sublayers": ["attn","ffn"] subset,    # default ["attn","ffn"]
        "layers": [int] | null,                # default null (= all)
        "correct_token": str (optional),       # for manual logit-diff token pair
        "incorrect_token": str (optional)
      }

    Frames: status → data (N) → baselines → complete | error.
    """
    await ws.accept()
    log.info("WS activation-patching connected (session='%s')", name)
    mgr = get_manager()

    try:
        info = mgr.get(name)
    except KeyError:
        log.warning("WS activation-patching: session '%s' not found", name)
        await _send_json(ws, {"type": "error", "message": f"Session '{name}' not found"})
        await ws.close()
        return

    try:
        mgr.ensure_pytorch(name)
    except Exception as e:
        log.exception("WS activation-patching: ensure_pytorch failed for '%s'", name)
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

    clean_prompt = config.get("clean_prompt", "")
    corrupted_prompt = config.get("corrupted_prompt", "")
    direction = config.get("direction", "denoise")
    measurement_position = int(config.get("measurement_position", -1))
    positions = config.get("positions")
    sublayers = tuple(config.get("sublayers", ["attn", "ffn"]))
    layers = config.get("layers")
    correct_token = config.get("correct_token")
    incorrect_token = config.get("incorrect_token")

    from llm_surgeon import probe
    from llm_surgeon.probe import attribution_patch, attribution_patch_per_head, attribution_patch_per_neuron

    mode = config.get("mode", "exact")
    if mode not in ("exact", "approx", "approx_head", "edge", "circuit", "approx_neuron"):
        await _send_json(ws, {"type": "error",
                              "message": f"mode must be 'exact', 'approx', 'approx_head', 'edge', 'circuit', or 'approx_neuron', got {mode!r}"})
        await ws.close()
        return

    correct_token_id: int | None = None
    incorrect_token_id: int | None = None
    if correct_token is not None or incorrect_token is not None:
        if correct_token is None or incorrect_token is None:
            await _send_json(ws, {"type": "error",
                                  "message": "correct_token and incorrect_token must both be provided or both omitted"})
            await ws.close()
            return
        try:
            c_ids = info.tokenizer(correct_token, add_special_tokens=False)["input_ids"]
            i_ids = info.tokenizer(incorrect_token, add_special_tokens=False)["input_ids"]
            if len(c_ids) != 1 or len(i_ids) != 1:
                raise ValueError(
                    f"correct_token/incorrect_token must tokenize to exactly one token "
                    f"(got {len(c_ids)} and {len(i_ids)})"
                )
            correct_token_id = int(c_ids[0])
            incorrect_token_id = int(i_ids[0])
        except Exception as e:
            await _send_json(ws, {"type": "error", "message": str(e)})
            await ws.close()
            return

    top_k_edges = int(config.get("top_k_edges", 200))
    tau = float(config.get("tau", 0.02))
    top_k_candidates = int(config.get("top_k_candidates", 2000))
    top_k_neurons = int(config.get("top_k_neurons", 200))
    n_steps_raw = config.get("n_steps", 1)
    try:
        n_steps = int(n_steps_raw)
    except (TypeError, ValueError):
        n_steps = 1
    if n_steps < 1 or n_steps > 50:
        await ws.send_json({"type": "error", "message": f"n_steps must be int in [1, 50], got {n_steps_raw!r}"})
        await ws.close()
        return

    # approx/approx_head/edge/circuit/approx_neuron mode requires token IDs before the call (backward needs a scalar metric).
    # Auto-pick: do a quick no_grad forward on the clean prompt to argmax the target pair.
    if mode in ("approx", "approx_head", "edge", "circuit", "approx_neuron") and correct_token_id is None:
        try:
            def _auto_pick_ids():
                device = next(info.model.parameters()).device
                clean_ids = info.tokenizer(clean_prompt, return_tensors="pt")["input_ids"].to(device)
                corr_ids = info.tokenizer(corrupted_prompt, return_tensors="pt")["input_ids"].to(device)
                with torch.no_grad():
                    clean_logits = info.model(clean_ids).logits[0, measurement_position]
                    corr_logits = info.model(corr_ids).logits[0, measurement_position]
                return _auto_pick_ap_pair(clean_logits, corr_logits)
            loop = asyncio.get_running_loop()
            async with info.lock:
                correct_token_id, incorrect_token_id = await loop.run_in_executor(
                    None, _auto_pick_ids
                )
        except Exception as e:
            await _send_json(ws, {"type": "error", "message": f"Auto-pick token IDs failed: {e}"})
            await ws.close()
            return

    connected = True
    loop = asyncio.get_running_loop()

    def on_cell(layer_idx: int, unit_or_sub: str, position: int, cell: dict) -> None:
        nonlocal connected
        if not connected:
            return
        msg: dict = {
            "type": "data",
            "layer": layer_idx,
            "original_layer": info.original_layer(layer_idx),
            "position": position,
        }
        if mode == "approx_head":
            msg["unit"] = cell.get("unit", unit_or_sub)
        else:
            msg["sublayer"] = unit_or_sub
        if "patched_logits" in cell:
            msg["patched_logits"] = _encode_hidden_state(cell["patched_logits"])
        if "ap_recovery" in cell:
            msg["ap_recovery"] = cell["ap_recovery"]
        fut = asyncio.run_coroutine_threadsafe(_send_json(ws, msg), loop)
        try:
            ok = fut.result(timeout=10)
        except Exception:
            ok = False
        if not ok:
            connected = False

    def on_cell_edge(cell: dict) -> None:
        nonlocal connected
        if not connected:
            return
        msg: dict = {
            "type": "data",
            "writer_layer": cell["writer_layer"],
            "writer_unit": cell["writer_unit"],
            "reader_layer": cell["reader_layer"],
            "reader_unit": cell["reader_unit"],
            "position": cell["position"],
            "ap_recovery": cell["ap_recovery"],
        }
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
            await _send_json(ws, {"type": "status", "message": "Capturing activations..."})
            if mode == "exact":
                result = await loop.run_in_executor(
                    None,
                    lambda: probe.activation_patch(
                        info.model, info.tokenizer,
                        clean_prompt=clean_prompt,
                        corrupted_prompt=corrupted_prompt,
                        direction=direction,
                        measurement_position=measurement_position,
                        positions=positions,
                        sublayers=sublayers,
                        layers=layers,
                        on_cell=on_cell,
                    ),
                )
            elif mode == "approx_head":
                assert correct_token_id is not None and incorrect_token_id is not None
                _cid: int = correct_token_id
                _iid: int = incorrect_token_id
                _n_steps_head: int = n_steps
                result = await loop.run_in_executor(
                    None,
                    lambda: attribution_patch_per_head(
                        info.model, info.tokenizer,
                        clean_prompt=clean_prompt,
                        corrupted_prompt=corrupted_prompt,
                        correct_token_id=_cid,
                        incorrect_token_id=_iid,
                        direction=direction,
                        measurement_position=measurement_position,
                        positions=positions,
                        layers=layers,
                        n_steps=_n_steps_head,
                        on_cell=on_cell,
                    ),
                )
            elif mode == "edge":
                from llm_surgeon.probe import edge_attribution_patch
                assert correct_token_id is not None and incorrect_token_id is not None
                _cid2: int = correct_token_id
                _iid2: int = incorrect_token_id
                _topk: int = top_k_edges
                _n_steps_edge: int = n_steps
                result = await loop.run_in_executor(
                    None,
                    lambda: edge_attribution_patch(
                        info.model, info.tokenizer,
                        clean_prompt=clean_prompt,
                        corrupted_prompt=corrupted_prompt,
                        correct_token_id=_cid2,
                        incorrect_token_id=_iid2,
                        direction=direction,
                        measurement_position=measurement_position,
                        positions=positions,
                        layers=layers,
                        top_k_edges=_topk,
                        n_steps=_n_steps_edge,
                        on_cell=on_cell_edge,
                    ),
                )
            elif mode == "circuit":
                from llm_surgeon.probe import extract_circuit
                assert correct_token_id is not None and incorrect_token_id is not None
                if tau < 0.0:
                    await _send_json(ws, {"type": "error", "message": "tau must be >= 0"})
                    await ws.close()
                    return
                if top_k_candidates < 1:
                    await _send_json(ws, {"type": "error", "message": "top_k_candidates must be >= 1"})
                    await ws.close()
                    return

                def on_cell_circuit(cell: dict) -> None:
                    nonlocal connected
                    if not connected:
                        return
                    msg: dict = {
                        "type": "data",
                        "writer_layer": cell["writer_layer"],
                        "writer_unit": cell["writer_unit"],
                        "reader_layer": cell["reader_layer"],
                        "reader_unit": cell["reader_unit"],
                        "position": cell["position"],
                        "ap_recovery": cell["ap_recovery"],
                        "in_circuit": cell.get("in_circuit", False),
                    }
                    fut = asyncio.run_coroutine_threadsafe(_send_json(ws, msg), loop)
                    try:
                        ok = fut.result(timeout=10)
                    except Exception:
                        ok = False
                    if not ok:
                        connected = False

                _cid3: int = correct_token_id
                _iid3: int = incorrect_token_id
                _tau: float = tau
                _topkc: int = top_k_candidates
                _n_steps_circuit: int = n_steps
                result = await asyncio.to_thread(
                    extract_circuit,
                    info.model,
                    info.tokenizer,
                    clean_prompt,
                    corrupted_prompt,
                    correct_token_id=_cid3,
                    incorrect_token_id=_iid3,
                    direction=direction,
                    measurement_position=measurement_position,
                    positions=positions,
                    layers=layers,
                    tau=_tau,
                    top_k_candidates=_topkc,
                    n_steps=_n_steps_circuit,
                    on_cell=on_cell_circuit,
                )
            elif mode == "approx_neuron":
                if top_k_neurons < 1:
                    await _send_json(ws, {"type": "error", "message": "top_k_neurons must be >= 1"})
                    await ws.close()
                    return
                assert correct_token_id is not None and incorrect_token_id is not None
                _cid4: int = correct_token_id
                _iid4: int = incorrect_token_id
                _topkn: int = top_k_neurons

                def on_cell_neuron(cell: dict) -> None:
                    nonlocal connected
                    if not connected:
                        return
                    msg: dict = {"type": "data", **cell}
                    fut = asyncio.run_coroutine_threadsafe(_send_json(ws, msg), loop)
                    try:
                        ok = fut.result(timeout=10)
                    except Exception:
                        ok = False
                    if not ok:
                        connected = False

                _n_steps_neuron: int = n_steps
                result = await asyncio.to_thread(
                    attribution_patch_per_neuron,
                    info.model,
                    info.tokenizer,
                    clean_prompt,
                    corrupted_prompt,
                    correct_token_id=_cid4,
                    incorrect_token_id=_iid4,
                    direction=direction,
                    measurement_position=measurement_position,
                    positions=positions,
                    layers=layers,
                    top_k_neurons=_topkn,
                    n_steps=_n_steps_neuron,
                    on_cell=on_cell_neuron,
                )
            else:
                assert correct_token_id is not None and incorrect_token_id is not None
                _cid = correct_token_id
                _iid = incorrect_token_id
                _n_steps = n_steps
                result = await loop.run_in_executor(
                    None,
                    lambda: attribution_patch(
                        info.model, info.tokenizer,
                        clean_prompt=clean_prompt,
                        corrupted_prompt=corrupted_prompt,
                        correct_token_id=_cid,
                        incorrect_token_id=_iid,
                        direction=direction,
                        measurement_position=measurement_position,
                        positions=positions,
                        sublayers=sublayers,
                        layers=layers,
                        n_steps=_n_steps,
                        on_cell=on_cell,
                    ),
                )

        if connected:
            baselines_msg: dict = {
                "type": "baselines",
                "clean_logits": _encode_hidden_state(result.clean_baseline_logits),
                "corrupted_logits": _encode_hidden_state(result.corrupted_baseline_logits),
                "prompt_tokens_clean": result.prompt_tokens_clean,
                "prompt_tokens_corrupted": result.prompt_tokens_corrupted,
                "measurement_position": result.measurement_position,
            }
            if correct_token_id is None:
                correct_token_id, incorrect_token_id = _auto_pick_ap_pair(
                    result.clean_baseline_logits, result.corrupted_baseline_logits,
                )
            baselines_msg["correct_token_id"] = correct_token_id
            baselines_msg["incorrect_token_id"] = incorrect_token_id
            await _send_json(ws, baselines_msg)

            summary: dict = {
                "num_cells": len(result.cells),
                "direction": result.direction,
                "measurement_position": result.measurement_position,
                "mode": result.mode,
            }
            if result.n_heads is not None:
                summary["n_heads"] = result.n_heads
            if result.n_edges is not None:
                summary["n_edges"] = result.n_edges
            if result.n_edges_in_circuit is not None:
                summary["n_edges_in_circuit"] = result.n_edges_in_circuit
            if result.n_nodes_in_circuit is not None:
                summary["n_nodes_in_circuit"] = result.n_nodes_in_circuit
            if result.tau is not None:
                summary["tau"] = result.tau
            if result.n_neurons is not None:
                summary["n_neurons"] = result.n_neurons
            if mode == "circuit":
                summary["top_k_candidates"] = top_k_candidates
            if mode == "approx_neuron":
                summary["top_k_neurons"] = top_k_neurons
            if result.n_steps is not None:
                summary["n_steps"] = result.n_steps
            await _send_json(ws, {"type": "complete", "summary": summary})

    except ValueError as e:
        log.warning("WS activation-patching validation error: %s", e)
        await _send_json(ws, {"type": "error", "message": str(e)})
    except Exception as e:
        log.exception("WS activation-patching error (session='%s')", name)
        await _send_json(ws, {"type": "error", "message": str(e)})
    finally:
        log.info("WS activation-patching disconnected (session='%s')", name)
        try:
            await ws.close()
        except RuntimeError:
            pass
