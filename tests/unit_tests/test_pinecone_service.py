import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.pinecone_service import PineconeManager


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pinecone_manager_singleton():
    """Test that PineconeManager maintains a singleton pattern."""
    # Create two instances
    manager1 = PineconeManager()
    manager2 = PineconeManager()
    
    # They should be the same object
    assert manager1 is manager2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pinecone_manager_initialize():
    """Test PineconeManager initialization."""
    with patch('app.services.pinecone_service.PineconeAsyncio', MagicMock()) as mock_pinecone:
        # Set up mock instance
        mock_client = mock_pinecone.return_value
        mock_client.__aenter__ = AsyncMock()
        
        # Mock list_indexes
        mock_indexes = MagicMock()
        mock_indexes.names.return_value = ["existing-index"]
        mock_client.list_indexes = AsyncMock(return_value=mock_indexes)
        
        # Mock IndexAsyncio
        mock_index = MagicMock()
        mock_client.IndexAsyncio = MagicMock(return_value=mock_index)
        
        # Create manager and patch _check_index_health
        manager = PineconeManager()
        manager._check_index_health = AsyncMock(return_value=True)
        
        # Set configuration for testing
        from app.config import settings
        settings.pinecone_index_name = "existing-index"
        settings.pinecone_host = "test-host"
        
        # Initialize
        result = await manager.initialize()
        
        # Verify expected behavior
        mock_client.__aenter__.assert_called_once()
        mock_client.list_indexes.assert_called_once()
        mock_client.IndexAsyncio.assert_called_once_with(host="test-host")
        manager._check_index_health.assert_called_once()
        assert result == mock_index


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pinecone_manager_create_index():
    """Test PineconeManager creating a new index."""
    with patch('app.services.pinecone_service.PineconeAsyncio', MagicMock()) as mock_pinecone:
        # Set up mock instance
        mock_client = mock_pinecone.return_value
        mock_client.__aenter__ = AsyncMock()
        
        # Mock list_indexes for non-existing index
        mock_indexes = MagicMock()
        mock_indexes.names.return_value = ["other-index"]
        mock_client.list_indexes = AsyncMock(return_value=mock_indexes)
        
        # Mock create_index
        mock_client.create_index = AsyncMock()
        
        # Mock IndexAsyncio
        mock_index = MagicMock()
        mock_client.IndexAsyncio = MagicMock(return_value=mock_index)
        
        # Create manager and patch methods
        manager = PineconeManager()
        manager._wait_for_ready = AsyncMock()
        manager._check_index_health = AsyncMock(return_value=True)
        
        # Set configuration for testing
        from app.config import settings
        settings.pinecone_index_name = "new-index"
        settings.pinecone_dimension = 1536
        settings.pinecone_host = "test-host"
        settings.pinecone_cloud = "aws"
        settings.pinecone_region = "us-west-1"
        
        # Initialize
        result = await manager.initialize()
        
        # Verify expected behavior
        mock_client.create_index.assert_called_once()
        manager._wait_for_ready.assert_called_once()
        assert result == mock_index


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pinecone_manager_query_index_success():
    """Test PineconeManager's query_index method with successful query."""
    # Create a manager with mocked index
    manager = PineconeManager()
    manager.index = AsyncMock()
    manager._ensure_index_healthy = AsyncMock()
    
    # Mock response for query
    mock_response = MagicMock()
    mock_response.matches = [
        {"id": "doc1", "score": 0.95, "metadata": {"type": "room"}},
        {"id": "doc2", "score": 0.85, "metadata": {"type": "policy"}}
    ]
    manager.index.query = AsyncMock(return_value=mock_response)
    
    # Create test embedding
    embedding = [0.1] * 1536
    
    # Call query_index
    result = await manager.query_index(embedding, top_k=2)
    
    # Verify behavior
    manager._ensure_index_healthy.assert_called_once()
    manager.index.query.assert_called_once_with(
        vector=embedding,
        top_k=2,
        namespace="knowledge-base",
        include_metadata=True
    )
    assert result == mock_response.matches
    assert len(result) == 2
    assert result[0]["id"] == "doc1"
    assert result[1]["id"] == "doc2"
    
@pytest.mark.unit
@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures():
    """Test circuit breaker opens after consecutive failures."""
    manager = PineconeManager()
    manager.index = AsyncMock()
    
    # Mock query to fail 5 times
    manager.index.query = AsyncMock(side_effect=Exception("Simulated error"))
    manager._circuit_state = "closed"
    manager._consecutive_failures = 0
    
    # Make repeated failing calls
    for _ in range(5):
        await manager.query_index([0.1]*1536)
    
    assert manager._circuit_state == "open"
    assert time.time() - manager._last_failure < 1

@pytest.mark.unit
@pytest.mark.asyncio
async def test_index_readiness_timeout():
    """Test timeout while waiting for index readiness."""
    manager = PineconeManager()
    manager.pc = MagicMock()
    manager.pc.describe_index = AsyncMock(side_effect=Exception("Timeout"))
    
    with pytest.raises(TimeoutError):
        await manager._wait_for_ready(timeout=1)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_failure_handling():
    """Test query failure increments failure count."""
    manager = PineconeManager()
    manager.index = AsyncMock()
    manager.index.query = AsyncMock(side_effect=Exception("DB error"))
    
    original_failures = manager._consecutive_failures
    
    result = await manager.query_index([0.1]*1536)
    
    assert len(result) == 0
    assert manager._consecutive_failures == original_failures + 1

@pytest.mark.unit
@pytest.mark.asyncio
async def test_cleanup_on_failure():
    """Test resource cleanup during failure handling."""
    manager = PineconeManager()
    manager.index = MagicMock()
    manager.pc = MagicMock()
    manager.pc.__aexit__ = AsyncMock()
    
    await manager._handle_failure()
    
    assert manager.index is None
    manager.pc.__aexit__.assert_called_once()
