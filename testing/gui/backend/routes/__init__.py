from fastapi import APIRouter
from .sessions import router as sessions_router
from .probes import router as probes_router
from .inspect import router as inspect_router
from .capture import router as capture_router

api_router = APIRouter(prefix="/api")
api_router.include_router(sessions_router)
api_router.include_router(inspect_router)
api_router.include_router(capture_router)

ws_router = APIRouter(prefix="/ws")
ws_router.include_router(probes_router)
