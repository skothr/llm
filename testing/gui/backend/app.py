from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .sessions import SessionManager

manager = SessionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    for name in list(manager._sessions.keys()):
        manager.delete(name)

app = FastAPI(title="LLM Surgeon GUI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from .routes import api_router, ws_router
app.include_router(api_router)
app.include_router(ws_router)
