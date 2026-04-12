import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..sessions import SessionManager

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
