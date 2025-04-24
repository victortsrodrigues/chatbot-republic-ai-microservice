from openai import OpenAI
from app.config import settings
from app.utils.logger import logger
from typing import List, Dict
import time
import asyncio
from collections import deque

class OpenAIRateLimiter:
    """Global rate limiter to be shared across all client connections"""
    def __init__(self):
        # Rate limiting parameters
        self.rpm_limit = 60  # Requests per minute
        self.tpm_limit = 100000  # Tokens per minute
        self._request_timestamps = deque(maxlen=self.rpm_limit)
        self._token_usage = deque(maxlen=100)  # Track recent token usage
        self._lock = asyncio.Lock()
    
    async def check_and_wait(self, token_estimate: float):
        """Check if we're about to exceed rate limits and wait if necessary"""
        async with self._lock:
            current_time = time.time()
            
            # Clean up old timestamps (older than 1 minute)
            minute_ago = current_time - 60
            while self._request_timestamps and self._request_timestamps[0] < minute_ago:
                self._request_timestamps.popleft()
                
            # Clean up old token usage
            while self._token_usage and self._token_usage[0][0] < minute_ago:
                self._token_usage.popleft()
                
            # Calculate current usage
            request_count = len(self._request_timestamps)
            token_count = sum(tokens for _, tokens in self._token_usage)
            
            # Check if adding this request would exceed limits
            if request_count >= self.rpm_limit or token_count + token_estimate > self.tpm_limit:
                # Calculate wait time to respect rate limits
                if self._request_timestamps:
                    wait_time = 60 - (current_time - self._request_timestamps[0]) + 0.1
                    wait_time = max(0.1, min(wait_time, 60))  # Between 0.1 and 60 seconds
                    logger.warning(f"Rate limit approaching, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    # Recursive check after waiting to ensure we're under limits
                    return await self.check_and_wait(token_estimate)
    
    async def update_usage(self, tokens_used: float):
        """Update rate limit tracking after a successful request"""
        async with self._lock:
            current_time = time.time()
            self._request_timestamps.append(current_time)
            self._token_usage.append((current_time, tokens_used))

# Create a singleton instance to be shared across all client connections
global_rate_limiter = OpenAIRateLimiter()

class OpenAIHandler:
    def __init__(self):
      self.client = OpenAI(api_key=settings.openai_api_key)
      self._circuit_state = "closed"
      self._last_failure = None
      self.rate_limiter = global_rate_limiter

    async def generate_embedding(self, text: str) -> List[float]:
        token_estimate = len(text.split()) * 1.3  # Rough token estimate
        await self.rate_limiter.check_and_wait(token_estimate)
        
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=settings.embedding_model
            )
            
            # Update rate limit tracking
            await self.rate_limiter.update_usage(token_estimate)
            
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise

    async def generate_chat_completion(self, messages: List[Dict]) -> str:
        if self._circuit_state == "open":
            if time.time() - self._last_failure < 60:
                logger.warning("Circuit breaker open")
                return ""
        
        # Estimate token usage (rough estimate)
        token_estimate = sum(len(m.get("content", "").split()) * 1.3 for m in messages)
        await self.rate_limiter.check_and_wait(token_estimate)
        
        try:
            # Validate input structure
            if not isinstance(messages, list) or len(messages) == 0:
                raise ValueError("Invalid messages format")
            
            response = await self.client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,
                temperature=0
            )
            
            # Track actual usage for better future estimates
            actual_tokens = response.usage.total_tokens if hasattr(response, 'usage') else token_estimate
            await self.rate_limiter.update_usage(actual_tokens)
            
            self._circuit_state = "closed"
            return response.choices[0].message.content.strip()
        except Exception as e:
            self._handle_error(e)
            raise
        
    async def check_moderation(self, input: str) -> bool:
        token_estimate = len(input.split()) * 1.3
        await self.rate_limiter.check_and_wait(token_estimate)
        
        try:
            response = await self.client.moderations.create(
                model="omni-moderation-latest",
                input=input
            )
            await self.rate_limiter.update_usage(token_estimate)
            return response.results[0].flagged
        except Exception as e:
            logger.error(f"Moderation check failed: {str(e)}")
            return False
        
    def _handle_error(self, error):
        self._circuit_state = "open"
        self._last_failure = time.time()
        logger.error(f"OpenAI API error: {str(error)}")