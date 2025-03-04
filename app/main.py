from fastapi import FastAPI
from app.routers import rag_router, health_router
from app.config import settings

app = FastAPI(title="Student Republic RAG Service")

app.include_router(rag_router.router, prefix="/rag", tags=["RAG"])
app.include_router(health_router.router, prefix="/health", tags=["Health"])