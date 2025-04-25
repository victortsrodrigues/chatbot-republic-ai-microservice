import tiktoken
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from app.config import settings
from app.utils.logger import logger
from typing import List, Dict, Optional, Any
import time
import asyncio
from collections import deque


class OpenAIRateLimiter:
    """Global rate limiter with configurable limits and efficient tracking"""

    def __init__(self):
        # Set default rate limits if not specified in settings
        self.rpm_limit = getattr(settings, "openai_rpm_limit", 60)
        self.tpm_limit = getattr(settings, "openai_tpm_limit", 100000)

        # Tracking recent requests and token usage
        self._request_timestamps = deque(maxlen=self.rpm_limit)
        self._token_usage = deque(maxlen=100)
        self._lock = asyncio.Lock()

        # Track and log rate limit usage periodically
        self._last_log_time = 0
        self._log_interval = 60  # Log rate limit usage every minute

    async def check_and_wait(self, token_estimate: int) -> None:
        """Check if we're about to exceed rate limits and wait if necessary"""
        async with self._lock:
            now = time.time()

            # Purge old entries (older than 1 minute)
            minute_ago = now - 60
            while self._request_timestamps and self._request_timestamps[0] < minute_ago:
                self._request_timestamps.popleft()

            while self._token_usage and self._token_usage[0][0] < minute_ago:
                self._token_usage.popleft()

            # Calculate current usage
            request_count = len(self._request_timestamps)
            token_count = sum(tokens for _, tokens in self._token_usage)

            # Periodically log usage statistics
            if now - self._last_log_time > self._log_interval:
                usage_percent_rpm = (request_count / self.rpm_limit) * 100
                usage_percent_tpm = (token_count / self.tpm_limit) * 100
                logger.info(
                    f"OpenAI API usage: {request_count}/{self.rpm_limit} RPM ({usage_percent_rpm:.1f}%), "
                    f"{token_count}/{self.tpm_limit} TPM ({usage_percent_tpm:.1f}%)"
                )
                self._last_log_time = now

            # Wait if we're approaching limits
            if (
                (request_count >= self.rpm_limit or token_count + token_estimate > self.tpm_limit)
                and self._request_timestamps
            ):
                # Calculate wait time based on oldest timestamp
                wait_time = max(0.1, 60 - (now - self._request_timestamps[0]) + 0.1)
                logger.warning(f"Rate limit approaching, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                # Recursive check after waiting
                return await self.check_and_wait(token_estimate)

    async def update_usage(self, tokens_used: int) -> None:
        """Update rate limit tracking after a successful request"""
        async with self._lock:
            self._request_timestamps.append(time.time())
            self._token_usage.append((time.time(), tokens_used))


# Create a singleton instance to be shared across all client connections
global_rate_limiter = OpenAIRateLimiter()


class OpenAIHandler:
    """Handles async OpenAI API interactions with rate limiting and resilience features"""

    _instance = None
    _init_lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(self):
        async with self._init_lock:
            if getattr(self, "initialized", False):
                return
            logger.info("Initializing OpenAI async client...")
            await self.client.__aenter__()
            self.initialized = True

    def __init__(self):
        if hasattr(self, "client"):
            return

        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            # Max concurrent connections - adjust based on expected load
            max_retries=3,  # Let the client handle some retries internally
            timeout=30.0,  # Sensible timeout to prevent hung requests
        )

        # Circuit breaker pattern state
        self._circuit_state = "closed"
        self._last_failure = 0.0
        self._consecutive_failures = 0
        self._failure_threshold = 5

        # Initialize tokenizers for different models
        self._encoders: Dict[str, Any] = {}
        self._init_tokenizers()

        # Share the global rate limiter
        self.rate_limiter = global_rate_limiter

        # Request semaphore to limit concurrent requests from this handler
        # This prevents one client from using all connections
        self._request_semaphore = asyncio.Semaphore(
            getattr(settings, "openai_max_concurrent_requests", 10)
        )  # Adjust based on expected load

    def _init_tokenizers(self) -> None:
        """Initialize tokenizers for the models we'll use"""
        # Only load encoders for models we'll actually use
        models = [settings.chat_model, settings.embedding_model]

        # Load unique models
        for model in set(models):
            try:
                if "embedding" in model:
                    self._encoders[model] = tiktoken.get_encoding("cl100k_base")
                else:
                    self._encoders[model] = tiktoken.encoding_for_model(model)
            except Exception as e:
                logger.warning(f"Couldn't load tokenizer for {model}: {e}")
                # Fall back to cl100k_base which works with newer models
                self._encoders[model] = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str, model: str) -> int:
        """Precise token counting using tiktoken for the specific model"""
        encoder = self._encoders.get(model, next(iter(self._encoders.values())))
        return len(encoder.encode(text))

    def _count_message_tokens(self, messages: List[Dict], model: str) -> int:
        """Count tokens in a list of chat messages"""
        tokens = sum(
            len(
                self._encoders.get(model, next(iter(self._encoders.values()))).encode(
                    msg.get("content", "")
                )
            )
            + 3
            for msg in messages
        )
        return tokens + 3

    async def _make_api_call(self, func, *args, **kwargs):
        """Generic method to make API calls with rate limiting and circuit breaker"""
        # Check circuit breaker
        if self._circuit_state == "open":
            # Auto-reset circuit breaker after 2 minutes of cooldown
            if time.time() - self._last_failure > getattr(
                settings, "openai_circuit_reset", 120
            ):
                logger.info("Circuit breaker reset")
                self._circuit_state = "closed"
                self._consecutive_failures = 0
            else:
                logger.warning("Circuit breaker open - request rejected")
                raise Exception(
                    "Service temporarily unavailable (circuit breaker open)"
                )

        # Limit concurrent requests from this handler instance
        async with self._request_semaphore:
            try:
                # Call the API function with retries
                result = await func(*args, **kwargs)
                # Reset failure counter on success
                self._consecutive_failures = 0
                return result
            except Exception as e:
                # Track failures for circuit breaker
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._failure_threshold:
                    self._circuit_state = "open"
                    self._last_failure = time.time()
                    logger.error(
                        f"Circuit breaker opened after {self._consecutive_failures} failures"
                    )
                raise e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings with rate limiting and retries"""
        await self.initialize()
        model = settings.embedding_model
        token_count = self._count_tokens(text, model)

        # Check rate limits before proceeding
        await self.rate_limiter.check_and_wait(token_count)

        try:
            # Make the API call
            async def _embed():
                response = await self.client.embeddings.create(input=text, model=model)
                return response.data[0].embedding

            embedding = await self._make_api_call(_embed)

            # Update rate limit tracking
            await self.rate_limiter.update_usage(token_count)
            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def generate_chat_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate chat completion with rate limiting and retries"""
        await self.initialize()
        model = settings.chat_model

        # Count tokens accurately
        token_count = self._count_message_tokens(messages, model)

        # Check rate limits before proceeding
        await self.rate_limiter.check_and_wait(token_count)

        try:
            # Make the API call
            async def _complete():
                completion_args = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }

                # Add max_tokens if specified
                if max_tokens is not None:
                    completion_args["max_tokens"] = max_tokens

                return await self.client.chat.completions.create(**completion_args)

            response = await self._make_api_call(_complete)

            # Track actual usage (more accurate for future estimates)
            actual_tokens = response.usage.total_tokens
            await self.rate_limiter.update_usage(actual_tokens)

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((Exception,)),
    )
    async def check_moderation(self, input_text: str) -> bool:
        """Check content moderation with rate limiting"""
        await self.initialize()
        # Count tokens for rate limiting
        token_count = self._count_tokens(input_text, "text-moderation-latest")

        # Check rate limits before proceeding
        await self.rate_limiter.check_and_wait(token_count)

        try:
            # Make the API call
            async def _moderate():
                return await self.client.moderations.create(
                    input=input_text, model="text-moderation-latest"
                )

            response = await self._make_api_call(_moderate)

            # Update rate limit tracking
            await self.rate_limiter.update_usage(token_count)

            return response.results[0].flagged

        except Exception as e:
            logger.error(f"Moderation check failed: {str(e)} - failing closed")
            return True  # Safer default: assume content is flagged if check fails

    async def close(self):
        """Gracefully close the client connection"""
        if getattr(self, "initialized", False):
            await self.client.__aexit__(None, None, None)
            self.initialized = False