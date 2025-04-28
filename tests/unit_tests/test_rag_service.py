import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.rag_service import RAGOrchestrator, CircuitBreaker, async_retry


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rag_initialize(mock_openai_handler, mock_pinecone_manager, mock_mongodb_client):
    """Test RAGOrchestrator initialization."""
    with patch('app.services.rag_service.OpenAIHandler', return_value=mock_openai_handler), \
         patch('app.services.rag_service.PineconeManager', return_value=mock_pinecone_manager), \
         patch('app.services.rag_service.MongoDBClient', return_value=mock_mongodb_client):
        
        # Create orchestrator and initialize
        orchestrator = RAGOrchestrator()
        await orchestrator.initialize()
        
        # Verify initialization
        assert orchestrator._initialized == True
        mock_openai_handler.initialize.assert_called_once()
        mock_pinecone_manager.initialize.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_query_cache_hit(mock_openai_handler, mock_pinecone_manager, mock_mongodb_client):
    """Test process_query with cache hit."""
    with patch('app.services.rag_service.OpenAIHandler', return_value=mock_openai_handler), \
         patch('app.services.rag_service.PineconeManager', return_value=mock_pinecone_manager), \
         patch('app.services.rag_service.MongoDBClient', return_value=mock_mongodb_client):
        
        orchestrator = RAGOrchestrator()
        orchestrator._initialized = True
        orchestrator.openai = mock_openai_handler
        orchestrator.pinecone = mock_pinecone_manager
        orchestrator.mongo = mock_mongodb_client
        
        # Prepare cache
        query = "Quais quartos estão disponíveis?"
        history = []
        system_message = ""
        user_id = "test_user"
        
        cache_key = orchestrator._generate_cache_key(query, history, system_message, user_id)
        cached_response = {
            "response": "Temos dois quartos disponíveis.",
            "requires_action": False
        }
        orchestrator._response_cache[cache_key] = cached_response
        
        # Test the method
        result = await orchestrator.process_query(query, history, system_message, user_id)
        
        # Verify result came from cache
        assert result == cached_response
        # Ensure the inner processing wasn't called
        mock_openai_handler.generate_chat_completion.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_query_full_flow(mock_openai_handler, mock_pinecone_manager, mock_mongodb_client):
    """Test process_query with full flow."""
    with patch('app.services.rag_service.OpenAIHandler', return_value=mock_openai_handler), \
         patch('app.services.rag_service.PineconeManager', return_value=mock_pinecone_manager), \
         patch('app.services.rag_service.MongoDBClient', return_value=mock_mongodb_client):
        
        orchestrator = RAGOrchestrator()
        orchestrator._initialized = True
        orchestrator.openai = mock_openai_handler
        orchestrator.pinecone = mock_pinecone_manager
        orchestrator.mongo = mock_mongodb_client
        orchestrator._correct_typos = AsyncMock(side_effect=lambda x: x)  # Return input unchanged
        orchestrator._normalize_query = MagicMock(side_effect=lambda x: x)  # Return input unchanged
        orchestrator._decide_inclusions = AsyncMock(return_value=(True, False))  # Include room, no media
        orchestrator._parse_filters = AsyncMock(return_value={"availability": True})
        orchestrator._generate_response = AsyncMock(return_value={
            "response": "Temos dois quartos disponíveis: A101 e B202.",
            "requires_action": False
        })
        
        # Test the method
        query = "Quais quartos estão disponíveis?"
        history = []
        result = await orchestrator.process_query(query, history)
        
        # Verify method calls
        mock_openai_handler.check_moderation.assert_called()
        mock_openai_handler.generate_embedding.assert_called_once_with(query)
        mock_pinecone_manager.query_index.assert_called_once()
        mock_mongodb_client.get_all_rooms.assert_called_once()
        orchestrator._generate_response.assert_called_once()
        
        # Verify result
        assert result["response"] == "Temos dois quartos disponíveis: A101 e B202."
        assert result["requires_action"] == False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_query_moderation_blocked(mock_openai_handler, mock_pinecone_manager, mock_mongodb_client):
    """Test process_query when moderation blocks the query."""
    with patch('app.services.rag_service.OpenAIHandler', return_value=mock_openai_handler), \
         patch('app.services.rag_service.PineconeManager', return_value=mock_pinecone_manager), \
         patch('app.services.rag_service.MongoDBClient', return_value=mock_mongodb_client):
        
        orchestrator = RAGOrchestrator()
        orchestrator._initialized = True
        orchestrator.openai = mock_openai_handler
        orchestrator.pinecone = mock_pinecone_manager
        orchestrator.mongo = mock_mongodb_client
        
        # Mock moderation to fail
        mock_openai_handler.check_moderation.return_value = True
        
        # Test the method
        query = "Conteúdo inadequado"
        history = []
        result = await orchestrator.process_query(query, history)
        
        # Verify moderation was checked and no further processing happened
        mock_openai_handler.check_moderation.assert_called_once_with(query)
        mock_openai_handler.generate_embedding.assert_not_called()
        
        # Verify result
        assert "error" in result
        assert "content policy violation" in result["error"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_correct_typos(mock_openai_handler):
    """Test typo correction functionality."""
    with patch('app.services.rag_service.OpenAIHandler', return_value=mock_openai_handler):
        orchestrator = RAGOrchestrator()
        orchestrator._initialized = True
        orchestrator.openai = mock_openai_handler
        
        # Set up the mock to return a corrected query
        mock_openai_handler.generate_chat_completion.return_value = "Quais quartos estão disponíveis?"
        
        # Test the method
        result = await orchestrator._correct_typos("Quais quaros estao disponiveis?")
        
        # Verify the result
        assert result == "Quais quartos estão disponíveis?"
        mock_openai_handler.generate_chat_completion.assert_called_once()


@pytest.mark.unit
def test_normalize_query():
    """Test query normalization with synonym replacement."""
    orchestrator = RAGOrchestrator()
    orchestrator.synonym_map = {"valor": "preço", "lugares": "quartos"}
    
    # Test the method
    result = orchestrator._normalize_query("Qual o valor dos lugares disponíveis?")
    
    # Verify the result
    assert result == "Qual o preço dos quartos disponíveis?"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_decide_inclusions(mock_openai_handler):
    """Test decision on what data to include in response."""
    with patch('app.services.rag_service.OpenAIHandler', return_value=mock_openai_handler):
        orchestrator = RAGOrchestrator()
        orchestrator._initialized = True
        orchestrator.openai = mock_openai_handler
        
        # Set up the mock to return a decision
        mock_openai_handler.generate_chat_completion.return_value = '{"include_room_data": true, "include_media": true}'
        
        # Test the method with context that has a relevant room
        context = [
            {"score": 0.9, "metadata": {"type": "room", "room_id": "A101"}},
            {"score": 0.7, "metadata": {"type": "policy", "title": "Rules"}}
        ]
        query = "Mostre-me os quartos disponíveis com fotos."
        include_room, include_media = await orchestrator._decide_inclusions(query, context)
        
        # Verify the result - room should be included because of high score context
        assert include_room == True
        # Media should be included based on OpenAI decision
        assert include_media == True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_filters(mock_openai_handler):
    """Test parsing of query filters."""
    with patch('app.services.rag_service.OpenAIHandler', return_value=mock_openai_handler):
        orchestrator = RAGOrchestrator()
        orchestrator._initialized = True
        orchestrator.openai = mock_openai_handler
        
        # Set up the mock to return a filter JSON
        mock_filter_json = '{"price": {"$gte": 500, "$lte": 800}, "features": {"$all": ["suite"]}, "availability": true}'
        mock_openai_handler.generate_chat_completion.return_value = mock_filter_json
        
        # Test the method
        query = "Quais quartos são suítes com preço entre R$500 e R$800?"
        filters = await orchestrator._parse_filters(query)
        
        # Verify the result
        assert filters["price"]["$gte"] == 500
        assert filters["price"]["$lte"] == 800
        assert "suite" in filters["features"]["$all"]
        assert filters["availability"] == True


@pytest.mark.unit
def test_fallback_filter_parsing():
    """Test fallback filter parsing using regex."""
    orchestrator = RAGOrchestrator()
    
    # Test with price range
    query = "Quais quartos entre 500 e 800 reais?"
    filters = orchestrator._fallback_filter_parsing(query)
    assert "price" in filters
    assert filters["price"]["$gte"] == 500
    assert filters["price"]["$lte"] == 800
    
    # Test with features
    query = "Quero um quarto com suíte e varanda"
    filters = orchestrator._fallback_filter_parsing(query)
    assert "features" in filters
    assert "suíte" in filters["features"]["$all"]
    assert "varanda" in filters["features"]["$all"]
    
    # Test with availability
    query = "Quais quartos estão disponíveis?"
    filters = orchestrator._fallback_filter_parsing(query)
    assert filters["availability"] == True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_response(mock_openai_handler):
    """Test response generation with all available data."""
    with patch('app.services.rag_service.OpenAIHandler', return_value=mock_openai_handler):
        orchestrator = RAGOrchestrator()
        orchestrator._initialized = True
        orchestrator.openai = mock_openai_handler
        
        # Set up the mock to return a response
        mock_openai_handler.generate_chat_completion.return_value = "Temos dois quartos disponíveis: A101 e B202."
        
        # Test data
        query = "Quais quartos estão disponíveis?"
        context = [
            {"metadata": {"type": "room", "room_id": "A101", "description": "Suite com varanda"}},
            {"metadata": {"type": "room", "room_id": "B202", "description": "Quarto duplo com banheiro"}}
        ]
        rooms_data = [
            {"room_id": "A101", "description": "Suite com varanda", "price": 750},
            {"room_id": "B202", "description": "Quarto duplo com banheiro", "price": 650}
        ]
        history = []
        system_message = "Você é um assistente útil."
        
        # Call the method
        result = await orchestrator._generate_response(
            query, context, rooms_data, history, system_message, True
        )
        
        # Verify the result
        assert result["response"] == "Temos dois quartos disponíveis: A101 e B202."
        assert result["requires_action"] == False
        mock_openai_handler.generate_chat_completion.assert_called_once()


@pytest.mark.unit
def test_get_media_data():
    """Test extraction of media data from rooms."""
    orchestrator = RAGOrchestrator()
    
    # Test data
    rooms_data = [
        {
            "room_id": "A101",
            "description": "Suite com varanda",
            "s3_object_key": "rooms/A101/photo1.jpg"
        },
        {
            "room_id": "B202",
            "description": "Quarto duplo com banheiro",
            "s3_object_key": ["rooms/B202/photo1.jpg", "rooms/B202/photo2.jpg"]
        }
    ]
    
    # Call the method
    media_data = orchestrator._get_media_data(rooms_data)
    
    # Verify the result
    assert len(media_data) == 3
    assert "rooms/A101/photo1.jpg" in media_data
    assert "rooms/B202/photo1.jpg" in media_data
    assert "rooms/B202/photo2.jpg" in media_data


@pytest.mark.unit
def test_merge_media_data():
    """Test merging of media data into response."""
    orchestrator = RAGOrchestrator()
    
    # Test data
    response = {
        "response": "Temos dois quartos disponíveis: A101 e B202.",
        "requires_action": False
    }
    
    media_data = ["rooms/A101/photo1.jpg", "rooms/B202/photo1.jpg"]
    
    # Call the method
    result = orchestrator._merge_media_data(response, media_data)
    
    # Verify the result
    assert result["requires_action"] == True
    assert result["action_type"] == "fetch_media"
    assert result["media_list"] == media_data


@pytest.mark.unit
def test_circuit_breaker():
    """Test CircuitBreaker behavior."""
    # Create circuit breaker with small thresholds for testing
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=1)
    
    # Initial state should be closed
    assert breaker.state == "closed"
    assert breaker.should_reject() == False
    
    # Record failures up to threshold
    for _ in range(3):
        breaker.record_failure()
    
    # After threshold, circuit should be open
    assert breaker.state == "open"
    assert breaker.should_reject() == True
    
    # Wait for timeout to expire
    time.sleep(1.1)
    
    # Circuit should now allow requests again
    assert breaker.should_reject() == False
    assert breaker.state == "closed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_retry_decorator():
    """Test the async_retry decorator functionality."""
    # Create a mock function that fails twice then succeeds
    mock_func = AsyncMock()
    mock_func.side_effect = [
        Exception("First failure"),
        Exception("Second failure"),
        "Success"
    ]
    
    # Apply the decorator
    decorated_func = async_retry(max_retries=3, backoff_factor=1)(mock_func)
    
    # Call the decorated function
    result = await decorated_func()
    
    # Verify the function was called 3 times and succeeded on the third try
    assert mock_func.call_count == 3
    assert result == "Success"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_retry_decorator_failure():
    """Test the async_retry decorator when all retries fail."""
    # Create a mock function that always fails
    mock_func = AsyncMock(side_effect=Exception("Persistent failure"))
    
    # Apply the decorator
    decorated_func = async_retry(max_retries=2)(mock_func)
    
    # Call the decorated function and expect exception
    with pytest.raises(Exception) as excinfo:
        await decorated_func()
    
    # Verify the function was called the expected number of times
    assert mock_func.call_count == 2
    assert "Persistent failure" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close_method(mock_openai_handler, mock_pinecone_manager, mock_mongodb_client):
    """Test proper cleanup of resources in close method."""
    with patch('app.services.rag_service.OpenAIHandler', return_value=mock_openai_handler), \
         patch('app.services.rag_service.PineconeManager', return_value=mock_pinecone_manager), \
         patch('app.services.rag_service.MongoDBClient', return_value=mock_mongodb_client):
        
        orchestrator = RAGOrchestrator()
        orchestrator._initialized = True
        orchestrator.openai = mock_openai_handler
        orchestrator.pinecone = mock_pinecone_manager
        orchestrator.mongo = mock_mongodb_client
        
        # Call close method
        await orchestrator.close()
        
        # Verify all services were closed
        mock_openai_handler.close.assert_called_once()
        mock_pinecone_manager.close.assert_called_once()
        mock_mongodb_client.close.assert_called_once()
        
        # Verify initialized flag was reset
        assert orchestrator._initialized == False