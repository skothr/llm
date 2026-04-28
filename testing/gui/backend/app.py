import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .log import setup_logging
from .sessions import SessionManager

setup_logging()
log = logging.getLogger("gui.backend.app")

manager = SessionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("LLM Surgeon GUI backend starting")
    # Pre-warm the model-discovery cache in the background so the first
    # /api/models/available hit (typically from the frontend's initial load)
    # doesn't pay the ~1-second cold GGUF-header parse cost.
    import asyncio
    from .routes.sessions import _collect_available_models, _scan_executor

    async def _warm():
        try:
            await asyncio.get_running_loop().run_in_executor(_scan_executor, _collect_available_models)
            log.info("Model cache pre-warmed")
        except Exception:
            log.exception("Model cache pre-warm failed")

    warm_task = asyncio.create_task(_warm())

    # Re-register sessions (and replay surgery deltas) saved before the
    # last shutdown. Synchronous within startup so that by the time we
    # accept requests, mgr.list_sessions() reflects the persisted state.
    # Per-session failures are logged and skipped inside restore() — one
    # broken session can't block the rest.
    from . import persistence
    from .routes.sessions import _restore_register_one, _apply_op_for_replay
    try:
        await persistence.restore(manager, _restore_register_one, _apply_op_for_replay)
    except Exception:
        log.exception("Session restore failed; starting clean")

    try:
        yield
    finally:
        warm_task.cancel()
        log.info("Shutting down — cleaning up %d session(s)", len(manager._sessions))
        for name in list(manager._sessions.keys()):
            # Best-effort acquire info.lock before tearing down: if a WS
            # handler is mid-forward we don't want to rip the model out from
            # under the running CUDA kernels. A short timeout keeps shutdown
            # bounded — worst case we proceed anyway and the handler fails
            # with a torn-down model error, which is fine at shutdown.
            info = manager._sessions[name]
            try:
                await asyncio.wait_for(info.lock.acquire(), timeout=2.0)
                try:
                    manager.delete(name)
                finally:
                    try:
                        info.lock.release()
                    except RuntimeError:
                        # delete() already removed the session; the lock
                        # may be gone or re-ownership is moot here.
                        pass
            except asyncio.TimeoutError:
                log.warning("Shutdown: '%s' still busy after 2s — forcing delete", name)
                manager.delete(name)
        _scan_executor.shutdown(wait=False, cancel_futures=True)

app = FastAPI(title="LLM Surgeon GUI", lifespan=lifespan)

MAX_BODY_LOG = 2048

@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.info("%s %s", request.method, request.url.path)

    if log.isEnabledFor(logging.DEBUG):
        if request.url.query:
            log.debug("  query: %s", request.url.query)
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            # Skip reading the body when it is large — await request.body() buffers
            # the entire stream, so logging a 50MB upload would cost 50MB of RAM
            # per request even though we only preview the first 2KB.
            ct = request.headers.get("content-type", "")
            cl_header = request.headers.get("content-length")
            try:
                cl = int(cl_header) if cl_header is not None else -1
            except ValueError:
                cl = -1
            if cl > MAX_BODY_LOG:
                log.debug("  body (%s, %d bytes) [not logged: exceeds preview cap]", ct, cl)
            else:
                body_bytes = await request.body()
                if body_bytes:
                    preview = body_bytes[:MAX_BODY_LOG].decode("utf-8", errors="replace")
                    truncated = " [truncated]" if len(body_bytes) > MAX_BODY_LOG else ""
                    log.debug("  body (%s, %d bytes%s): %s", ct, len(body_bytes), truncated, preview)

    try:
        response = await call_next(request)
    except Exception:
        log.exception("Unhandled error: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    if response.status_code >= 400:
        log.warning("%s %s -> %d", request.method, request.url.path, response.status_code)
    else:
        log.debug("%s %s -> %d", request.method, request.url.path, response.status_code)
    return response

app.add_middleware(
    CORSMiddleware,
    # Allow both the frontend origins vite can bind to. Browsers treat
    # localhost and 127.0.0.1 as distinct origins, so a user who browses
    # to 127.0.0.1:5173 would otherwise fail CORS on any direct fetch to
    # 127.0.0.1:8000 (proxy routes are fine either way).
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

from .routes import api_router, ws_router
app.include_router(api_router)
app.include_router(ws_router)
