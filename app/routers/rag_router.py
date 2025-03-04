from fastapi import APIRouter, HTTPException
from app.models.schemas import RAGQuery, RAGResponse
from app.services.rag_service import RAGOrchestrator

router = APIRouter()
rag_orchestrator = RAGOrchestrator()

@router.post("/query", response_model=RAGResponse)
async def handle_rag_query(query: RAGQuery):
    try:
        result = await rag_orchestrator.process_query(
            query.query,
            filter=query.filter
        )
        return RAGResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))