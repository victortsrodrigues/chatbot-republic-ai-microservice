import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from app.config import settings
from app.services.openai_service import OpenAIHandler, OpenAIRateLimiter, UserBasedRateLimiter
from app.services.pinecone_service import PineconeManager
from app.services.mongo_service import MongoDBClient
from app.services.rag_service import RAGOrchestrator

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singleton instances between tests."""
    RAGOrchestrator._instance = None
    PineconeManager._instance = None
    OpenAIHandler._instance = None
    MongoDBClient._instance = None
    yield

@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Reset circuit breaker state between tests."""
    if RAGOrchestrator._instance:
        RAGOrchestrator._instance.circuit_breaker.state = "closed"
        RAGOrchestrator._instance.circuit_breaker.failure_count = 0
    yield

@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiter state between tests."""
    from app.services.openai_service import user_rate_limiter
    user_rate_limiter.user_requests = {}
    yield

@pytest.fixture(scope="function")
def event_loop():
    """Create a new event loop for each test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
async def cleanup_services():
    """Ensure services are properly cleaned up after tests."""
    yield
    # Cleanup after test
    if RAGOrchestrator._instance and hasattr(RAGOrchestrator._instance, "close"):
        await RAGOrchestrator._instance.close()

# General fixtures
@pytest.fixture
def sample_embedding() -> List[float]:
    """Returns a sample embedding vector."""
    return [0.1] * 1536  # Match the dimension in settings


@pytest.fixture
def sample_query() -> str:
    """Returns a sample user query."""
    return "Quais quartos estão disponíveis com preço entre R$500 e R$800?"


@pytest.fixture
def sample_history() -> List[Dict[str, str]]:
    """Returns a sample conversation history."""
    return [
        {"role": "user", "content": "Olá, estou procurando um quarto."},
        {"role": "assistant", "content": "Olá! Posso ajudar você a encontrar um quarto adequado. Que tipo de quarto você está procurando?"}
    ]


# Mock responses
@pytest.fixture
def mock_openai_response() -> Dict[str, Any]:
    """Returns a mock OpenAI chat completion response."""
    class MockResponse:
        class Usage:
            def __init__(self):
                self.total_tokens = 150

        class Choice:
            class Message:
                def __init__(self):
                    self.content = "Este é um exemplo de resposta do assistente."

            def __init__(self):
                self.message = self.Message()

        def __init__(self):
            self.choices = [self.Choice()]
            self.usage = self.Usage()

    return MockResponse()


@pytest.fixture
def mock_embedding_response() -> Dict[str, Any]:
    """Returns a mock OpenAI embedding response."""
    class MockEmbedding:
        def __init__(self):
            self.embedding = [0.1] * 1536

    class MockResponse:
        def __init__(self):
            self.data = [MockEmbedding()]

    return MockResponse()


@pytest.fixture
def mock_moderation_response() -> Dict[str, Any]:
    """Returns a mock OpenAI moderation response."""
    class MockResult:
        def __init__(self, flagged=False):
            self.flagged = flagged

    class MockResponse:
        def __init__(self, flagged=False):
            self.results = [MockResult(flagged)]

    return MockResponse(False)


@pytest.fixture
def mock_pinecone_response() -> Dict[str, Any]:
    """Returns a mock Pinecone query response."""
    class Match:
        def __init__(self, id, score, metadata):
            self.id = id
            self.score = score
            self.metadata = metadata

    class MockResponse:
        def __init__(self):
            self.matches = [
                Match("doc1", 0.92, {"type": "room", "room_id": "A101", "description": "Suite com varanda"}),
                Match("doc2", 0.85, {"type": "room", "room_id": "B202", "description": "Quarto duplo com banheiro"}),
                Match("doc3", 0.78, {"type": "policy", "title": "Regras da República"})
            ]

    return MockResponse()


@pytest.fixture
def mock_mongo_rooms_response() -> List[Dict[str, Any]]:
    """Returns a mock MongoDB rooms response."""
    return [
        {
            "room_id": "A101",
            "description": "Suite com varanda", 
            "price": 750,
            "features": ["suite", "varanda", "ar-condicionado"],
            "availability": True,
            "s3_object_key": "rooms/A101/photo1.jpg"
        },
        {
            "room_id": "B202",
            "description": "Quarto duplo com banheiro", 
            "price": 650,
            "features": ["banheiro", "escrivaninha"],
            "availability": True,
            "s3_object_key": ["rooms/B202/photo1.jpg", "rooms/B202/photo2.jpg"]
        }
    ]


# Service mocks
@pytest.fixture
def mock_openai_handler():
    """Returns a mocked OpenAIHandler."""
    with patch('app.services.openai_service.OpenAIHandler', autospec=True) as mock:
        instance = mock.return_value
        
        instance.initialize = AsyncMock()
        instance.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
        instance.generate_chat_completion = AsyncMock(return_value="Este é um exemplo de resposta do assistente.")
        instance.check_moderation = AsyncMock(return_value=False)
        instance.close = AsyncMock()
        
        yield instance


@pytest.fixture
def mock_pinecone_manager():
    """Returns a mocked PineconeManager."""
    with patch('app.services.pinecone_service.PineconeManager', autospec=True) as mock:
        instance = mock.return_value
        
        instance.initialize = AsyncMock()
        instance.query_index = AsyncMock(return_value=[
            {"id": "doc1", "score": 0.92, "metadata": {"type": "room", "room_id": "A101", "description": "Suite com varanda"}},
            {"id": "doc2", "score": 0.85, "metadata": {"type": "room", "room_id": "B202", "description": "Quarto duplo com banheiro"}},
            {"id": "doc3", "score": 0.78, "metadata": {"type": "policy", "title": "Regras da República"}}
        ])
        instance.close = AsyncMock()
        
        yield instance


@pytest.fixture
def mock_mongodb_client():
    """Returns a mocked MongoDBClient."""
    with patch('app.services.mongo_service.MongoDBClient', autospec=True) as mock:
        instance = mock.return_value
        
        instance.get_all_rooms = AsyncMock(return_value=[
            {
                "room_id": "A101",
                "description": "Suite com varanda", 
                "price": 750,
                "features": ["suite", "varanda", "ar-condicionado"],
                "availability": True,
                "s3_object_key": "rooms/A101/photo1.jpg"
            },
            {
                "room_id": "B202",
                "description": "Quarto duplo com banheiro", 
                "price": 650,
                "features": ["banheiro", "escrivaninha"],
                "availability": True,
                "s3_object_key": ["rooms/B202/photo1.jpg", "rooms/B202/photo2.jpg"]
            }
        ])
        instance.close = AsyncMock()
        instance._build_mongo_query = MagicMock(return_value={"availability": True})
        
        yield instance


@pytest.fixture
def mock_rag_orchestrator(mock_openai_handler, mock_pinecone_manager, mock_mongodb_client):
    """Returns a mocked RAGOrchestrator."""
    with patch('app.services.rag_service.RAGOrchestrator', autospec=True) as mock:
        instance = mock.return_value
        
        # Set attributes
        instance.openai = mock_openai_handler
        instance.pinecone = mock_pinecone_manager
        instance.mongo = mock_mongodb_client
        instance._initialized = True
        
        # Mock methods
        instance.initialize = AsyncMock()
        instance.process_query = AsyncMock(return_value={
            "response": "Este é um exemplo de resposta processada.",
            "requires_action": False
        })
        instance.close = AsyncMock()
        
        yield instance


# Add event loop for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()