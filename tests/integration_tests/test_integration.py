# tests/test_integration.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from app.main import app
from app.models.schemas import RAGQuery, RAGResponse
from app.services.rag_service import RAGOrchestrator


@pytest.fixture
def test_client():
    """Returns a FastAPI TestClient for making HTTP requests."""
    return TestClient(app)


class TestRagEndpoint:
    """Integration tests for the /rag/query endpoint."""
    
    @pytest.mark.asyncio
    async def test_rag_query_success(self, test_client, mock_rag_orchestrator):
        """Test successful RAG query with proper response."""
        # Mock RAGOrchestrator.process_query to return a successful response
        mock_rag_orchestrator.process_query.return_value = {
            "response": "Este é um exemplo de resposta processada.",
            "sources": [
                {"type": "room", "room_id": "A101", "description": "Suite com varanda"},
                {"type": "room", "room_id": "B202", "description": "Quarto duplo com banheiro"}
            ],
            "requires_action": False
        }
        
        # Create a test query
        test_query = {
            "query": "Quais quartos estão disponíveis?",
            "history": [
                {"role": "user", "content": "Olá, estou procurando um quarto."},
                {"role": "assistant", "content": "Olá! Posso ajudar você a encontrar um quarto adequado."}
            ],
            "system_message": "Você é um atendente prestativo da República dos Estudantes.",
            "user_id": "test_user_123"
        }
        
        # Send the request to the endpoint
        with patch('app.routers.rag_router.RAGOrchestrator', return_value=mock_rag_orchestrator):
            with patch('app.routers.rag_router.user_rate_limiter.check_user_limit', return_value=True):
                response = test_client.post("/rag/query", json=test_query)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Este é um exemplo de resposta processada."
        assert len(data["sources"]) == 2
        assert not data["requires_action"]
        
        # Verify the mock was called with the expected arguments
        mock_rag_orchestrator.process_query.assert_called_once()
        call_args = mock_rag_orchestrator.process_query.call_args[1]
        assert call_args["query"] == test_query["query"]
        assert call_args["user_id"] == test_query["user_id"]
    
    @pytest.mark.asyncio
    async def test_rag_query_with_media(self, test_client, mock_rag_orchestrator):
        """Test RAG query that returns media response."""
        # Mock RAGOrchestrator.process_query to return a response with media
        mock_rag_orchestrator.process_query.return_value = {
            "response": "Aqui estão fotos dos quartos disponíveis.",
            "sources": [
                {"type": "room", "room_id": "A101", "description": "Suite com varanda"}
            ],
            "requires_action": True,
            "action_type": "fetch_media",
            "media_list": ["rooms/A101/photo1.jpg", "rooms/B202/photo1.jpg"]
        }
        
        # Create a test query
        test_query = {
            "query": "Você pode mostrar fotos dos quartos?",
            "history": [],
            "system_message": None,
            "user_id": "test_user_456"
        }
        
        # Send the request to the endpoint
        with patch('app.routers.rag_router.RAGOrchestrator', return_value=mock_rag_orchestrator):
            with patch('app.routers.rag_router.user_rate_limiter.check_user_limit', return_value=True):
                response = test_client.post("/rag/query", json=test_query)
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Aqui estão fotos dos quartos disponíveis."
        assert data["requires_action"]
        assert data["action_type"] == "fetch_media"
        assert len(data["media_list"]) == 2
        
    @pytest.mark.asyncio
    async def test_rag_query_rate_limit_exceeded(self, test_client):
        """Test rate limiting for RAG queries."""
        # Mock rate limiter to reject the request
        with patch('app.routers.rag_router.user_rate_limiter.check_user_limit', return_value=False):
            response = test_client.post("/rag/query", json={
                "query": "Quais quartos estão disponíveis?",
                "history": [],
                "system_message": None,
                "user_id": "test_user_789"
            })
        
        # Verify rate limit response
        assert response.status_code == 429
        assert "Too many requests" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_rag_query_timeout(self, test_client, mock_rag_orchestrator):
        """Test handling of timeout errors."""
        # Mock RAGOrchestrator.process_query to raise a timeout
        mock_rag_orchestrator.process_query.side_effect = AsyncMock(side_effect=TimeoutError("Request timed out"))
        
        # Send the request to the endpoint
        with patch('app.routers.rag_router.RAGOrchestrator', return_value=mock_rag_orchestrator):
            with patch('app.routers.rag_router.user_rate_limiter.check_user_limit', return_value=True):
                response = test_client.post("/rag/query", json={
                    "query": "Quais quartos estão disponíveis?",
                    "history": [],
                    "system_message": None,
                    "user_id": "test_user_abc"
                })
        
        # Verify timeout response
        assert response.status_code == 504
        assert "request timed out" in response.json()["detail"].lower()
    
    @pytest.mark.asyncio
    async def test_rag_query_internal_error(self, test_client, mock_rag_orchestrator):
        """Test handling of internal server errors."""
        # Mock RAGOrchestrator.process_query to raise an exception
        mock_rag_orchestrator.process_query.side_effect = Exception("Something went wrong")
        
        # Send the request to the endpoint
        with patch('app.routers.rag_router.RAGOrchestrator', return_value=mock_rag_orchestrator):
            with patch('app.routers.rag_router.user_rate_limiter.check_user_limit', return_value=True):
                response = test_client.post("/rag/query", json={
                    "query": "Quais quartos estão disponíveis?",
                    "history": [],
                    "system_message": None,
                    "user_id": "test_user_def"
                })
        
        # Verify error response
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower()


class TestHealthEndpoints:
    """Integration tests for health check endpoints."""
    
    def test_liveness_check(self, test_client):
        """Test the /health/live endpoint."""
        response = test_client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}
    
    def test_readiness_check(self, test_client):
        """Test the /health/ready endpoint."""
        response = test_client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}


class TestRAGOrchestrator:
    """Integration tests for the RAGOrchestrator service."""
    
    @pytest.mark.asyncio
    async def test_process_query_end_to_end(
        self, 
        mock_openai_handler, 
        mock_pinecone_manager, 
        mock_mongodb_client,
        sample_query,
        sample_history
    ):
        """Test full RAG pipeline with mocked external services."""
        # Create a real RAGOrchestrator but with mocked dependencies
        with patch('app.services.rag_service.OpenAIHandler', return_value=mock_openai_handler):
            with patch('app.services.rag_service.PineconeManager', return_value=mock_pinecone_manager):
                with patch('app.services.rag_service.MongoDBClient', return_value=mock_mongodb_client):
                    # Create real orchestrator with mocked dependencies
                    orchestrator = RAGOrchestrator()
                    await orchestrator.initialize()
                    
                    # Process a query
                    result = await orchestrator.process_query(
                        query=sample_query,
                        history=sample_history,
                        user_id="test_user_xyz"
                    )
                    
                    # Verify the pipeline was executed correctly
                    assert "response" in result
                    assert mock_openai_handler.check_moderation.called
                    assert mock_openai_handler.generate_embedding.called
                    assert mock_pinecone_manager.query_index.called
                    assert mock_mongodb_client.get_all_rooms.called
                    assert mock_openai_handler.generate_chat_completion.called
    
    @pytest.mark.asyncio
    async def test_rag_circuit_breaker(
        self, 
        mock_openai_handler, 
        mock_pinecone_manager, 
        mock_mongodb_client
    ):
        """Test circuit breaker functionality in RAG service."""
        # Configure the mock to trigger circuit breaker
        mock_openai_handler.check_moderation.side_effect = [Exception("Service unavailable")] * 6
        
        with patch('app.services.rag_service.OpenAIHandler', return_value=mock_openai_handler):
            with patch('app.services.rag_service.PineconeManager', return_value=mock_pinecone_manager):
                with patch('app.services.rag_service.MongoDBClient', return_value=mock_mongodb_client):
                    orchestrator = RAGOrchestrator()
                    await orchestrator.initialize()
                    
                    # Generate multiple failures to trigger circuit breaker
                    for _ in range(6):
                        result = await orchestrator.process_query(
                            query="Teste circuit breaker",
                            history=[],
                            user_id="test_user_circuit"
                        )
                    
                    # Verify circuit breaker is open
                    assert "circuit_open" in result
                    assert result.get("circuit_open", False)
    
    @pytest.mark.asyncio
    async def test_content_moderation_flow(
        self, 
        mock_openai_handler, 
        mock_pinecone_manager, 
        mock_mongodb_client
    ):
        """Test content moderation in the RAG pipeline."""
        # Configure moderation to flag inappropriate content
        mock_openai_handler.check_moderation.return_value = True
        
        with patch('app.services.rag_service.OpenAIHandler', return_value=mock_openai_handler):
            with patch('app.services.rag_service.PineconeManager', return_value=mock_pinecone_manager):
                with patch('app.services.rag_service.MongoDBClient', return_value=mock_mongodb_client):
                    orchestrator = RAGOrchestrator()
                    await orchestrator.initialize()
                    
                    # Process a query that should be flagged
                    result = await orchestrator.process_query(
                        query="Conteúdo impróprio para teste",
                        history=[],
                        user_id="test_user_moderation"
                    )
                    
                    # Verify moderation response
                    assert "error" in result
                    assert "policy violation" in result["error"].lower()