from fastapi import APIRouter
from app.utils.logger import logger

router = APIRouter()

@router.get("/live")
async def liveness_check():
    logger.debug("Liveness check passed")
    return {"status": "alive"}

@router.get("/ready")
async def readiness_check():
    # Add actual readiness checks here
    return {"status": "ready"}