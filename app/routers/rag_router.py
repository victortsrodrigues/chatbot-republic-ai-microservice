from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from app.models.schemas import RAGQuery, RAGResponse
from app.services.rag_service import RAGOrchestrator
from app.utils.logger import logger
import asyncio
from typing import Dict, Any

router = APIRouter()

# Global response cache to reduce duplicate processing
response_cache: Dict[str, Dict[str, Any]] = {}
request_locks: Dict[str, asyncio.Lock] = {}
cache_lock = asyncio.Lock()

@router.post("/query", response_model=RAGResponse)
async def handle_rag_query(query: RAGQuery, request: Request, background_tasks: BackgroundTasks):
    """
    Handle a RAG query by processing the user's input, retrieving relevant context,
    and generating a response.
    """
    # Generate a cache key based on query content
    cache_key = f"{hash(query.query)}:{hash(str(query.history[-3:] if query.history else []))}"
    
    try:
        # Check if this exact request is already being processed
        async with cache_lock:
            if cache_key in response_cache:
                logger.info(f"Cache hit for query: {query.query[:50]}...")
                return RAGResponse(**response_cache[cache_key])
                
            # Create a lock for this specific request if needed
            if cache_key not in request_locks:
                request_locks[cache_key] = asyncio.Lock()
        
        # Use the request-specific lock to prevent duplicate processing
        async with request_locks[cache_key]:
            # Double-check cache after acquiring lock
            if cache_key in response_cache:
                return RAGResponse(**response_cache[cache_key])
                
            # Get or initialize the RAG orchestrator
            rag_orchestrator = RAGOrchestrator()
            if not getattr(rag_orchestrator, "_initialized", False):
                await rag_orchestrator.initialize()
            
            # Process the query
            logger.info(f"Processing query: {query.query[:50]}...")
            result = await rag_orchestrator.process_query(
                query=query.query,
                history=query.history,
                system_message=query.system_message,
            )
            
            # Cache the result for future identical requests
            async with cache_lock:
                response_cache[cache_key] = result
                # Schedule cache cleanup to run in background
                background_tasks.add_task(cleanup_cache, cache_key)
            
            logger.info(f"Successfully processed query: {query.query[:50]}...")
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
    finally:
        # Clean up the request lock if it exists
        async with cache_lock:
            if cache_key in request_locks:
                request_locks.pop(cache_key, None)

async def cleanup_cache(cache_key: str, delay: int = 300):
    """Remove items from cache after a delay to avoid memory bloat"""
    await asyncio.sleep(delay)
    async with cache_lock:
        if cache_key in response_cache:
            response_cache.pop(cache_key, None)