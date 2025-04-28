from fastapi import APIRouter, HTTPException
from app.utils.logger import logger
from app.services.mongo_service import MongoDBClient
from app.services.pinecone_service import PineconeManager
from app.services.openai_service import OpenAIHandler

router = APIRouter()

@router.get("/live")
async def liveness_check():
    logger.debug("Liveness check passed")
    return {"status": "alive"}

@router.get("/ready")
async def readiness_check():
    # 1) MongoDB
    try:
        mongo = MongoDBClient()
        # ping é um comando síncrono, mas o cliente Motor expõe como coroutine
        await mongo.db.command("ping")
        logger.debug("MongoDB ping OK")
    except Exception as e:
        logger.error(f"MongoDB readiness failed: {e}")
        raise HTTPException(status_code=503, detail="MongoDB unavailable")

    # 2) Pinecone
    try:
        pine = PineconeManager()
        # initialize() já faz health check inicial
        await pine.initialize()
        # health check explícito
        healthy = await pine._check_index_health()
        if not healthy:
            raise RuntimeError("Pinecone index not ready")
        logger.debug("Pinecone index healthy")
    except Exception as e:
        logger.error(f"Pinecone readiness failed: {e}")
        raise HTTPException(status_code=503, detail="Pinecone unavailable")

    # 3) OpenAI
    try:
        oa = OpenAIHandler()
        await oa.initialize()
        logger.debug("OpenAI client initialized")
    except Exception as e:
        logger.error(f"OpenAI readiness failed: {e}")
        raise HTTPException(status_code=503, detail="OpenAI unavailable")
    
    return {"status": "ready"}