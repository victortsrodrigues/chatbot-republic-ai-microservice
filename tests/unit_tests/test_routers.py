import asyncio
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock

from app.models.schemas import RAGQuery, RAGResponse
from app.main import app
from app.routers import rag_router, health_router


@pytest.mark.unit
def test_liveness_check():
    """Test that liveness check endpoint returns correct response."""
    with patch('app.services.pinecone_service.PineconeManager.initialize', new_callable=AsyncMock), \
         patch('app.services.rag_service.RAGOrchestrator.initialize', new_callable=AsyncMock):
        
        with TestClient(app) as client:
            response = client.get("/health/live")
            assert response.status_code == 200
            assert response.json() == {"status": "alive"}


@pytest.mark.unit
def test_readiness_check():
    """Test that readiness check endpoint returns correct response."""
    with patch('app.services.pinecone_service.PineconeManager.initialize', AsyncMock(return_value=True)), \
         patch('app.services.pinecone_service.PineconeManager._check_index_health', AsyncMock(return_value=True)), \
         patch('app.services.rag_service.RAGOrchestrator.initialize', AsyncMock(return_value=True)):
        
        with TestClient(app) as client:
            response = client.get("/health/ready")
            assert response.status_code == 200
            assert response.json() == {"status": "ready"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_rag_query_success():
    """Test successful RAG query handling."""
    # Setup test data
    query = RAGQuery(
        query="Quais quartos estão disponíveis?",
        user_id="test_user",
        history=[],
        system_message="Você é um assistente útil."
    )
    
    # Mock the singleton instance directly
    mock_orchestrator = AsyncMock()
    mock_orchestrator._initialized = True
    mock_orchestrator.initialize = AsyncMock()
    mock_orchestrator.process_query = AsyncMock(return_value={
        "response": "Temos dois quartos disponíveis: A101 e B202.",
        "sources": [{"type": "room", "room_id": "A101"}],
        "requires_action": False
    })
    
    # Mock user rate limiter
    with patch('app.routers.rag_router.user_rate_limiter.check_user_limit', AsyncMock(return_value=True)), \
         patch('app.routers.rag_router.RAGOrchestrator', return_value=mock_orchestrator):
        
        # Call the route function directly
        result = await rag_router.handle_rag_query(query)
        
        # Assert expected outcomes
        assert isinstance(result, RAGResponse)
        assert result.response == "Temos dois quartos disponíveis: A101 e B202."
        assert result.sources == [{"type": "room", "room_id": "A101"}]
        assert result.requires_action == False
        assert result.action_type is None
        assert result.media_list is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_rag_query_rate_limited():
    """Test rate limiting in RAG query handling."""
    # Setup test data
    query = RAGQuery(
        query="Quais quartos estão disponíveis?",
        user_id="test_user",
        history=[],
    )
    
    # Mock user rate limiter to deny the request
    with patch('app.routers.rag_router.user_rate_limiter.check_user_limit', AsyncMock(return_value=False)):
        # Verify that HTTPException is raised with correct status code
        with pytest.raises(HTTPException) as excinfo:
            await rag_router.handle_rag_query(query)
        
        assert excinfo.value.status_code == 429
        assert "Too many requests" in excinfo.value.detail


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_rag_query_timeout():
    """Test timeout handling in RAG query."""
    # Setup test data
    query = RAGQuery(
        query="Quais quartos estão disponíveis?",
        user_id="test_user",
        history=[],
    )
    
    # Mock the singleton instance directly
    mock_orchestrator = AsyncMock()
    mock_orchestrator._initialized = True
    mock_orchestrator.initialize = AsyncMock()
    # Mock the process_query method to raise a timeout error
    mock_orchestrator.process_query = AsyncMock(side_effect=asyncio.TimeoutError("Request timed out"))
    
    # Mock user rate limiter
    with patch('app.routers.rag_router.user_rate_limiter.check_user_limit', AsyncMock(return_value=True)), \
         patch('app.routers.rag_router.RAGOrchestrator', return_value=mock_orchestrator):
        
        # Verify that HTTPException is raised with correct status code
        with pytest.raises(HTTPException) as excinfo:
            await rag_router.handle_rag_query(query)
        
        assert excinfo.value.status_code == 504
        assert "Request timed out" in excinfo.value.detail


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_rag_query_general_error():
    """Test general error handling in RAG query."""
    # Setup test data
    query = RAGQuery(
        query="Quais quartos estão disponíveis?",
        user_id="test_user",
        history=[],
    )
    
    # Mock the singleton instance directly
    mock_orchestrator = AsyncMock()
    mock_orchestrator._initialized = True
    mock_orchestrator.initialize = AsyncMock()
    # Mock the process_query method to raise a general exception
    mock_orchestrator.process_query = AsyncMock(side_effect=Exception("General error"))
    
    # Mock user rate limiter
    with patch('app.routers.rag_router.user_rate_limiter.check_user_limit', AsyncMock(return_value=True)), \
         patch('app.routers.rag_router.RAGOrchestrator', return_value=mock_orchestrator):
        
        # Verify that HTTPException is raised with correct status code
        with pytest.raises(HTTPException) as excinfo:
            await rag_router.handle_rag_query(query)
        
        assert excinfo.value.status_code == 500
        assert "An error occurred" in excinfo.value.detail