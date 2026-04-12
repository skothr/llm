from fastapi import APIRouter
from .sessions import router as sessions_router

api_router = APIRouter(prefix="/api")
api_router.include_router(sessions_router)

ws_router = APIRouter(prefix="/ws")
