import time
import asyncio
from pinecone import PineconeAsyncio, ServerlessSpec
from app.config import settings
from app.utils.logger import logger
from typing import List, Dict, Any

class PineconeManager:
    _instance = None
    _circuit_state = "closed"
    _last_failure = 0.0

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.pc = PineconeAsyncio(api_key=settings.pinecone_api_key)
            cls._instance.index = None
            cls._instance._last_healthy = 0.0
            cls._instance._health_lock = asyncio.Lock()
            cls._instance._init_lock = asyncio.Lock()
        return cls._instance

    async def initialize(self) -> Any:
        """Initialize client and index safely across coroutines"""
        async with self._init_lock:
            if self.index is not None:
                return self.index
            try:
                logger.info("Initializing Pinecone async client...")
                await self.pc.__aenter__()  # enter async context

                # list or create index
                existing = await self.pc.list_indexes()
                if settings.pinecone_index_name not in existing.names():
                    await self._create_index_with_retry()
                else:
                    logger.info(f"Index exists: {settings.pinecone_index_name}")

                # prepare data-plane client
                host = settings.pinecone_host  # ensure host in settings
                self.index = self.pc.IndexAsyncio(host=host)

                # initial health check
                if not await self._check_index_health():
                    raise ConnectionError("Index health check failed on init")

                return self.index
            except Exception as e:
                logger.critical(f"Initialization failed: {e}")
                await self._handle_failure()
                raise

    async def _create_index_with_retry(self):
        """Create index with exponential backoff"""
        retries = 0
        max_retries = getattr(settings, 'pinecone_max_retries', 5)
        backoff = 1
        while retries < max_retries:
            try:
                await self.pc.create_index(
                    name=settings.pinecone_index_name,
                    dimension=settings.pinecone_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=settings.pinecone_cloud,
                                         region=settings.pinecone_region)
                )
                # wait readiness
                await self._wait_for_ready()
                return
            except Exception as e:
                retries += 1
                logger.warning(f"Index creation attempt {retries} failed: {e}")
                await asyncio.sleep(backoff)
                backoff *= 2
        raise RuntimeError(f"Failed to create index after {max_retries} attempts")

    async def _wait_for_ready(self, timeout: int = 300):
        start = time.time()
        while time.time() - start < timeout:
            try:
                desc = await self.pc.describe_index(name=settings.pinecone_index_name)
                if getattr(desc.status, 'ready', False):
                    logger.info("Index ready")
                    return
            except Exception as e:
                logger.warning(f"Readiness check error: {e}")
            await asyncio.sleep(2)
        raise TimeoutError("Index not ready within timeout")

    async def query_index(self, embedding: List[float], top_k: int = 3,
                          namespace: str = "knowledge-base", include_metadata: bool = True) -> List[Dict[str, Any]]:
        """Query with circuit breaker and health checks"""
        # circuit breaker
        if self._circuit_state == "open" and time.time() - self._last_failure < settings.pinecone_circuit_timeout:
            logger.warning("Circuit open - skipping query")
            return []

        if self.index is None:
            await self.initialize()

        try:
            # periodic health check
            await self._ensure_index_healthy()

            response = await self.index.query(
                vector=embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=include_metadata
            )
            self._circuit_state = "closed"
            return response.matches
        except Exception as e:
            logger.error(f"Query error: {e}")
            await self._handle_failure()
            return []

    async def _ensure_index_healthy(self):
        now = time.time()
        if now - self._last_healthy > settings.pinecone_health_interval:
            async with self._health_lock:
                if now - self._last_healthy > settings.pinecone_health_interval:
                    if not await self._check_index_health():
                        raise ConnectionError("Unhealthy index")

    async def _check_index_health(self) -> bool:
        try:
            desc = await self.pc.describe_index(name=settings.pinecone_index_name)
            ready = getattr(desc.status, 'ready', False)
            self._last_healthy = time.time()
            return ready
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def _handle_failure(self):
        self._circuit_state = "open"
        self._last_failure = time.time()
        # cleanup
        try:
            await self.pc.__aexit__(None, None, None)
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
        self.index = None

    async def close(self):
        """Shutdown client"""
        await self._handle_failure()  # CHANGED: reuse failure cleanup for exit