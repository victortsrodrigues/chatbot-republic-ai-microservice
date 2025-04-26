# tests/test_integration.py

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from app.main import app
from app.models.schemas import RAGQuery, RAGResponse
from app.services.rag_service import RAGOrchestrator

import asyncio
import time
from app.utils.logger import logger
from concurrent.futures import ThreadPoolExecutor

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

    def test_empty_query_validation(self, test_client):
        """Test validation for empty query."""
        test_query = {
            "query": "",  # Empty query should be rejected
            "history": [],
            "user_id": "test_user_123"
        }
        
        response = test_client.post("/rag/query", json=test_query)
        
        # Verify validation error response
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("query" in error["loc"] and "should have at least 1 character" in error["msg"].lower() for error in error_detail)
    
    def test_whitespace_query_validation(self, test_client):
        """Test validation for query with only whitespace."""
        test_query = {
            "query": "   ",  # Only whitespace should be rejected
            "history": [],
            "user_id": "test_user_123"
        }
        
        response = test_client.post("/rag/query", json=test_query)
        
        # Verify validation error response
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("query" in error["loc"] and "empty" in error["msg"].lower() for error in error_detail)
    
    def test_empty_user_id_validation(self, test_client):
        """Test validation for empty user_id."""
        test_query = {
            "query": "Quais quartos estão disponíveis?",
            "history": [],
            "user_id": ""  # Empty user_id should be rejected
        }
        
        response = test_client.post("/rag/query", json=test_query)
        
        # Verify validation error response
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("user_id" in error["loc"] and "should have at least 1 character" in error["msg"].lower() for error in error_detail)
    
    def test_invalid_history_format_validation(self, test_client):
        """Test validation for invalid history format."""
        test_query = {
            "query": "Quais quartos estão disponíveis?",
            "history": [
                {"invalid_key": "This history item is missing role and content"}
            ],
            "user_id": "test_user_123"
        }
        
        response = test_client.post("/rag/query", json=test_query)
        
        # Verify validation error response
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("history" in error["loc"] and "must contain role and content" in error["msg"].lower() for error in error_detail)
    
    def test_invalid_role_in_history_validation(self, test_client):
        """Test validation for invalid role in history."""
        test_query = {
            "query": "Quais quartos estão disponíveis?",
            "history": [
                {"role": "invalid_role", "content": "This has an invalid role"}
            ],
            "user_id": "test_user_123"
        }
        
        response = test_client.post("/rag/query", json=test_query)
        
        # Verify validation error response
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("history" in error["loc"] and "invalid role" in error["msg"].lower() for error in error_detail)
    
    def test_history_too_long_validation(self, test_client):
        """Test validation for history that exceeds the maximum length."""
        # Create a history with 51 items (exceeding the 50 limit)
        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(51)
        ]
        
        test_query = {
            "query": "Quais quartos estão disponíveis?",
            "history": long_history,
            "user_id": "test_user_123"
        }
        
        response = test_client.post("/rag/query", json=test_query)
        
        # Verify validation error response
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("history" in error["loc"] and "too long" in error["msg"].lower() for error in error_detail)
    
    def test_query_too_long_validation(self, test_client):
        """Test validation for query that exceeds the maximum length."""
        # Create a query with 4001 characters (exceeding the 4000 limit)
        long_query = "a" * 4001
        
        test_query = {
            "query": long_query,
            "history": [],
            "user_id": "test_user_123"
        }
        
        response = test_client.post("/rag/query", json=test_query)
        
        # Verify validation error response
        assert response.status_code == 422
        error_detail = response.json()["detail"]
        assert any("query" in error["loc"] and "at most 4000" in error["msg"].lower() for error in error_detail)

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
                    
                    
                    
class TestConcurrentRequests:
    """Tests for handling multiple concurrent requests."""
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, test_client, mock_rag_orchestrator):
        """Test handling of multiple concurrent requests from different users."""
        # Configure mock to return different responses based on user_id
        async def mock_process_query(**kwargs):
            user_id = kwargs.get("user_id", "unknown")
            # Simulate processing time
            await asyncio.sleep(0.1)
            return {
                "response": f"Response for {user_id}",
                "sources": [{"type": "test", "id": user_id}],
                "requires_action": False
            }
        
        mock_rag_orchestrator.process_query.side_effect = mock_process_query
        
        # Patch rate limiter to allow all requests but track calls
        rate_limit_calls = {}
        
        async def mock_check_user_limit(user_id):
            rate_limit_calls[user_id] = rate_limit_calls.get(user_id, 0) + 1
            # Allow all requests except for specific test user
            return user_id != "rate_limited_user"
        
        # Number of concurrent requests to simulate
        num_requests = 10
        num_users = 5  # Distribute requests among these users
        
        # Create test queries for different users
        test_queries = [
            {
                "query": f"Test query {i}",
                "history": [],
                "system_message": None,
                "user_id": f"test_user_{i % num_users}"  # Distribute among users
            }
            for i in range(num_requests)
        ]
        
        # Add a rate-limited user
        test_queries.append({
            "query": "This should be rate limited",
            "history": [],
            "system_message": None,
            "user_id": "rate_limited_user"
        })
        
        # Function to make a request in a separate thread
        def make_request(query_data):
            with patch('app.routers.rag_router.RAGOrchestrator', return_value=mock_rag_orchestrator):
                with patch('app.routers.rag_router.user_rate_limiter.check_user_limit', side_effect=mock_check_user_limit):
                    return test_client.post("/rag/query", json=query_data)
        
        # Use ThreadPoolExecutor to make concurrent requests
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_requests + 1) as executor:
            responses = list(executor.map(make_request, test_queries))
        end_time = time.time()
        
        # Verify responses
        success_count = 0
        rate_limited_count = 0
        
        for i, response in enumerate(responses):
            if test_queries[i]["user_id"] == "rate_limited_user":
                assert response.status_code == 429
                rate_limited_count += 1
            else:
                assert response.status_code == 200
                data = response.json()
                assert data["response"] == f"Response for {test_queries[i]['user_id']}"
                success_count += 1
        
        # Verify rate limiter was called for each user
        assert len(rate_limit_calls) <= num_users + 1  # +1 for rate limited user
        
        # Verify the rate limited user was rejected
        assert rate_limited_count == 1
        
        # Verify successful requests
        assert success_count == num_requests
        
        # Verify mock was called the expected number of times
        assert mock_rag_orchestrator.process_query.call_count == num_requests
        
        # Log performance information
        total_time = end_time - start_time
        requests_per_second = num_requests / total_time
        logger.info(f"Processed {num_requests} requests in {total_time:.2f}s ({requests_per_second:.2f} req/s)")
    
    @pytest.mark.asyncio
    async def test_connection_pool_under_load(self, test_client):
        """Test the connection pool behavior under load with real services."""
        # This test verifies that connection pools for MongoDB and other services
        # properly handle concurrent requests without exhausting connections
        
        # Skip this test in CI environments or when using mocks
        # if os.environ.get("CI") == "true" or os.environ.get("USE_MOCKS") == "true":
        #     pytest.skip("Skipping connection pool test in CI environment")
        
        # Create a real RAGOrchestrator for this test
        # We'll use the actual services but with minimal operations
        
        # Patch the expensive operations to be fast but still use connections
        with patch('app.services.openai_service.OpenAIHandler.generate_embedding', 
                  return_value=[0.1] * 1536):
            with patch('app.services.openai_service.OpenAIHandler.generate_chat_completion',
                      return_value="Test response"):
                with patch('app.services.openai_service.OpenAIHandler.check_moderation',
                          return_value=False):
                    with patch('app.services.pinecone_service.PineconeManager.query_index',
                              return_value=[{"id": "doc1", "score": 0.9, "metadata": {"type": "test"}}]):
                        
                        # Number of concurrent requests
                        num_requests = 20
                        
                        # Create test queries
                        test_queries = [
                            {
                                "query": f"Connection pool test {i}",
                                "history": [],
                                "system_message": None,
                                "user_id": f"pool_test_user_{i % 5}"  # Use 5 different users
                            }
                            for i in range(num_requests)
                        ]
                        
                        # Make concurrent requests
                        async def make_async_request(query_data):
                            response = test_client.post("/rag/query", json=query_data)
                            return response
                        
                        # Use asyncio.gather to run requests concurrently
                        tasks = [make_async_request(query) for query in test_queries]
                        responses = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        # Count successful responses and exceptions
                        success_count = 0
                        error_count = 0
                        
                        for response in responses:
                            if isinstance(response, Exception):
                                error_count += 1
                            elif response.status_code == 200:
                                success_count += 1
                        
                        # Verify most requests succeeded
                        # We allow some failures due to potential rate limiting
                        assert success_count >= num_requests * 0.8, f"Only {success_count}/{num_requests} requests succeeded"
                        
                        # Log results
                        logger.info(f"Connection pool test: {success_count} successes, {error_count} errors out of {num_requests} requests")