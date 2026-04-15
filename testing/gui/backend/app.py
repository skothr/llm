import logging
from contextlib import asynccontextmanager
from pathlib import Path
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
    yield
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
            body_bytes = await request.body()
            if body_bytes:
                ct = request.headers.get("content-type", "")
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
