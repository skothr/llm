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
    from .routes.sessions import _collect_available_models

    async def _warm():
        try:
            await asyncio.get_event_loop().run_in_executor(None, _collect_available_models)
            log.info("Model cache pre-warmed")
        except Exception:
            log.exception("Model cache pre-warm failed")

    warm_task = asyncio.create_task(_warm())
    try:
        yield
    finally:
        warm_task.cancel()
        log.info("Shutting down — cleaning up %d session(s)", len(manager._sessions))
        for name in list(manager._sessions.keys()):
            manager.delete(name)

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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from .routes import api_router, ws_router
app.include_router(api_router)
app.include_router(ws_router)
