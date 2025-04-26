from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routers import rag_router, health_router
from app.services.rag_service import RAGOrchestrator
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events for the application"""
    # Startup logic
    try:
        logger.info("Initializing services...")
        orchestrator = RAGOrchestrator()
        await orchestrator.initialize()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.critical(f"Failed to initialize services: {str(e)}")

    yield  # Run application
    
    # Shutdown logic
    try:
        logger.info("Shutting down services...")
        if orchestrator:
            await orchestrator.close()
        logger.info("Services shut down successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

app = FastAPI(title="Student Republic RAG Service", lifespan=lifespan)

app.include_router(rag_router.router, prefix="/rag", tags=["RAG"])
app.include_router(health_router.router, prefix="/health", tags=["Health"])