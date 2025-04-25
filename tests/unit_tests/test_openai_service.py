import pytest
import time
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.openai_service import OpenAIHandler, OpenAIRateLimiter, UserBasedRateLimiter


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_rate_limiter_check_and_wait():
    """Test rate limiter's check_and_wait method."""
    # Create a rate limiter with small limits for testing
    rate_limiter = OpenAIRateLimiter()
    rate_limiter.rpm_limit = 5
    rate_limiter.tpm_limit = 1000
    
    # Simulate some previous requests
    rate_limiter._request_timestamps.extend([time.time() - 30] * 3)  # 3 requests in the last 30 seconds
    rate_limiter._token_usage.extend([(time.time() - 30, 200)] * 3)  # 600 tokens used in the last 30 seconds
    
    # Test regular operation - should not wait with these values
    start_time = time.time()
    await rate_limiter.check_and_wait(300)  # 300 more tokens
    elapsed = time.time() - start_time
    
    # Should return quickly as we're under the limit
    assert elapsed < 0.1, f"Rate limiter waited unnecessarily: {elapsed}s"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_rate_limiter_approaching_limit():
    """Test rate limiter's behavior when approaching limits."""
    # Create a rate limiter with small limits for testing
    rate_limiter = OpenAIRateLimiter()
    rate_limiter.rpm_limit = 5
    rate_limiter.tpm_limit = 1000
    
    # Fill up to the limit
    now = time.time()
    rate_limiter._request_timestamps.extend([now - 10] * 5)  # 5 requests (at the limit)
    
    # Mock recursive call to break the loop
    with patch.object(rate_limiter, 'check_and_wait', AsyncMock()) as mock_check:
        # Only mock the recursive call, not the first one
        mock_check.side_effect = [None]  # Return None on any subsequent call
        
        # Mock sleep to avoid actual waiting
        with patch('asyncio.sleep', AsyncMock()) as mock_sleep:
            # Calling the real method first time
            original_method = OpenAIRateLimiter.check_and_wait
            await original_method(rate_limiter, 100)
            
            # It should attempt to sleep
            mock_sleep.assert_called_once()
            # The wait time should be approximately 50 seconds (+0.1 buffer)
            assert 49.9 <= mock_sleep.call_args[0][0] <= 50.2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_rate_limiter_update_usage():
    """Test rate limiter's update_usage method."""
    rate_limiter = OpenAIRateLimiter()
    
    # Initial state
    assert len(rate_limiter._request_timestamps) == 0
    assert len(rate_limiter._token_usage) == 0
    
    # Update usage
    await rate_limiter.update_usage(150)
    
    # Should have recorded the request and token usage
    assert len(rate_limiter._request_timestamps) == 1
    assert len(rate_limiter._token_usage) == 1
    assert rate_limiter._token_usage[0][1] == 150


@pytest.mark.unit
@pytest.mark.asyncio
async def test_user_rate_limiter():
    """Test UserBasedRateLimiter functionality."""
    limiter = UserBasedRateLimiter()
    user_id = "test_user"
    
    # First request should be allowed
    assert await limiter.check_user_limit(user_id) == True
    
    # Add 9 more requests to reach the limit (default max is 10)
    limiter.user_requests[user_id] = [time.time()] * 9
    
    # 10th request should still be allowed
    assert await limiter.check_user_limit(user_id) == True
    
    # 11th request should be rejected
    assert await limiter.check_user_limit(user_id) == False
    
    # After some time, requests should expire
    # Set requests to be more than 60 seconds old
    limiter.user_requests[user_id] = [time.time() - 61] * 10
    
    # New request should be allowed again
    assert await limiter.check_user_limit(user_id) == True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_handler_initialization(mock_openai_response):
    """Test OpenAIHandler initialization."""
    # Mock the AsyncOpenAI client
    with patch('app.services.openai_service.AsyncOpenAI', MagicMock()) as mock_client:
        # Set up the mock client
        mock_instance = mock_client.return_value
        mock_instance.__aenter__ = AsyncMock()
        
        # Create handler and initialize
        handler = OpenAIHandler()
        await handler.initialize()
        
        # Verify the client was initialized correctly
        assert handler.initialized == True
        mock_instance.__aenter__.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_handler_generate_embedding(mock_embedding_response):
    """Test OpenAIHandler's generate_embedding method."""
    with patch('app.services.openai_service.AsyncOpenAI', MagicMock()) as mock_client:
        # Setup the mock client
        mock_instance = mock_client.return_value
        mock_instance.__aenter__ = AsyncMock()
        mock_instance.embeddings.create = AsyncMock(return_value=mock_embedding_response)
        
        # Create handler and mock initialization
        handler = OpenAIHandler()
        handler.initialized = True
        handler.client = mock_instance
        handler.rate_limiter = MagicMock()
        handler.rate_limiter.check_and_wait = AsyncMock()
        handler.rate_limiter.update_usage = AsyncMock()
        
        # Call the method
        result = await handler.generate_embedding("Test text")
        
        # Verify results
        assert result == mock_embedding_response.data[0].embedding
        handler.rate_limiter.check_and_wait.assert_called_once()
        handler.rate_limiter.update_usage.assert_called_once()
        mock_instance.embeddings.create.assert_called_once_with(input="Test text", model="text-embedding-ada-002")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_handler_generate_chat_completion(mock_openai_response):
    """Test OpenAIHandler's generate_chat_completion method."""
    with patch('app.services.openai_service.AsyncOpenAI', MagicMock()) as mock_client:
        # Setup the mock client
        mock_instance = mock_client.return_value
        mock_instance.__aenter__ = AsyncMock()
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        
        # Create handler and mock initialization
        handler = OpenAIHandler()
        handler.initialized = True
        handler.client = mock_instance
        handler.rate_limiter = MagicMock()
        handler.rate_limiter.check_and_wait = AsyncMock()
        handler.rate_limiter.update_usage = AsyncMock()
        
        # Mock token counting
        handler._count_message_tokens = MagicMock(return_value=50)
        
        # Call the method
        messages = [{"role": "user", "content": "Hello"}]
        result = await handler.generate_chat_completion(messages, temperature=0.7)
        
        # Verify results
        assert result == mock_openai_response.choices[0].message.content.strip()
        handler.rate_limiter.check_and_wait.assert_called_once_with(50)
        handler.rate_limiter.update_usage.assert_called_once_with(mock_openai_response.usage.total_tokens)
        mock_instance.chat.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo", 
            messages=messages,
            temperature=0.7
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_handler_check_moderation(mock_moderation_response):
    """Test OpenAIHandler's check_moderation method."""
    with patch('app.services.openai_service.AsyncOpenAI', MagicMock()) as mock_client:
        # Setup the mock client
        mock_instance = mock_client.return_value
        mock_instance.__aenter__ = AsyncMock()
        mock_instance.moderations.create = AsyncMock(return_value=mock_moderation_response)
        
        # Create handler and mock initialization
        handler = OpenAIHandler()
        handler.initialized = True
        handler.client = mock_instance
        handler.rate_limiter = MagicMock()
        handler.rate_limiter.check_and_wait = AsyncMock()
        handler.rate_limiter.update_usage = AsyncMock()
        
        # Mock token counting
        handler._count_tokens = MagicMock(return_value=20)
        
        # Call the method
        result = await handler.check_moderation("This is a test message")
        
        # Verify results
        assert result == False  # Not flagged
        handler.rate_limiter.check_and_wait.assert_called_once_with(20)
        handler.rate_limiter.update_usage.assert_called_once_with(20)
        mock_instance.moderations.create.assert_called_once_with(
            input="This is a test message",
            model="text-moderation-latest"
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_openai_handler_circuit_breaker():
    """Test OpenAIHandler's circuit breaker functionality."""
    handler = OpenAIHandler()
    handler.initialized = True
    handler._failure_threshold = 3
    handler._circuit_state = "closed"
    handler._consecutive_failures = 0
    
    # Mock the API call function
    async def failing_function():
        raise Exception("API Error")
    
    # Trigger failures to open the circuit breaker
    for _ in range(3):
        with pytest.raises(Exception):
            await handler._make_api_call(failing_function)
    
    # Circuit should now be open
    assert handler._circuit_state == "open"
    assert handler._consecutive_failures == 3
    
    # Next call should fail immediately with circuit breaker error
    with pytest.raises(Exception) as excinfo:
        await handler._make_api_call(lambda: None)
    assert "circuit breaker open" in str(excinfo.value).lower()