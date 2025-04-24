from fastapi import APIRouter, HTTPException
from app.models.schemas import RAGQuery, RAGResponse
from app.services.rag_service import RAGOrchestrator
from app.utils.logger import logger

router = APIRouter()
rag_orchestrator = RAGOrchestrator()

@router.post("/query", response_model=RAGResponse)
async def handle_rag_query(query: RAGQuery):
    """
    Handle a RAG query by processing the user's input, retrieving relevant context,
    and generating a response.
    """
    try:
        logger.info(f"Received query: {query.query}")
        
        # Process the query using the RAG orchestrator
        result = await rag_orchestrator.process_query(
            query=query.query,
            history=query.history,
            system_message=query.system_message,
        )
        
        logger.info(f"Successfully processed query: {query.query}")
        return RAGResponse(**result)
        
    except HTTPException:
        # Re-raise HTTP exceptions (e.g., from external APIs)
        raise
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again later."
        )