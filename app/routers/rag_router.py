from fastapi import APIRouter, HTTPException
from app.models.schemas import RAGQuery, RAGResponse
from app.services.rag_service import RAGOrchestrator
from app.services.openai_service import user_rate_limiter
from app.utils.logger import logger
import asyncio

router = APIRouter()

@router.post("/query", response_model=RAGResponse)
async def handle_rag_query(query: RAGQuery):
    """
    Handle a RAG query by processing the user's input, retrieving relevant context,
    and generating a response.
    """
    allowed = await user_rate_limiter.check_user_limit(query.user_id)
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many requests, please retry later.")
    
    try:
            # Get or initialize the RAG orchestrator
            rag_orchestrator = RAGOrchestrator()
            
            # Process the query
            logger.info(f"Processing query from user {query.user_id}: {query.query[:50]}...")
            result = await rag_orchestrator.process_query(
                query=query.query,
                history=query.history,
                system_message=query.system_message,
                user_id=query.user_id,
            )
            
            logger.info(f"Successfully processed query for user {query.user_id}: {query.query[:50]}...")
            return RAGResponse(**result)
        
    except asyncio.TimeoutError:
        logger.error(f"Timeout processing query: {query.query[:50]}...")
        raise HTTPException(
            status_code=504,
            detail="Request timed out. The server is experiencing high load."
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (e.g., from external APIs)
        raise
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again later."
        )