import time
import asyncio
from pinecone import PineconeAsyncio, ServerlessSpec
from app.config import settings
from app.utils.logger import logger
from typing import List, Optional, Dict, Any, Union


class PineconeManager:
    def __init__(self):
        self.pc = PineconeAsyncio(api_key=settings.pinecone_api_key)
        self.index = None
        self._last_healthy = 0
        self._health_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()

    async def initialize(self):
        """Asynchronously initialize the Pinecone client and index"""
        async with self._init_lock:  # Prevent multiple concurrent initializations
            if self.index is not None:
                return self.index

            try:
                # Initialize Pinecone client
                logger.info("Initializing Pinecone async client...")
                await self.pc.__aenter__()

                existing_indexes = await self.pc.list_indexes()
                logger.info(f"Existing Pinecone indexes: {existing_indexes.names()}")

                # Check if index exists, create if it doesn't
                if settings.pinecone_index_name not in existing_indexes.names():
                    await self._create_new_index()
                else:
                    logger.info(f"Using existing index: {settings.pinecone_index_name}")

                # # Describe index to get host ##### utilizar environment variable
                # desc = await self.pc.describe_index(name=settings.pinecone_index_name)
                # host = desc.host

                # Create IndexAsyncio for data operations
                self.index = self.pc.IndexAsyncio(host=settings.pinecone_host)

                # Validate index is working with a health check
                if not await self._check_index_health():
                    raise ConnectionError("Could not validate Pinecone index health")

                return self.index

            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {e}")
                raise

    async def _create_new_index(self):
        """Create a new Pinecone index asynchronously"""
        logger.info(f"Creating new Pinecone index: {settings.pinecone_index_name}")
        try:
            # Create index (blocking operation, use run_in_executor)
            await self.pc.create_index(
                settings.pinecone_index_name,
                dimension=settings.pinecone_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud, region=settings.pinecone_region
                ),
            )

            # Wait for index to be ready
            logger.info(
                f"Waiting for index {settings.pinecone_index_name} to be ready..."
            )
            ready = False
            retries = 0
            max_retries = (
                settings.pinecone_max_retries
                if hasattr(settings, "pinecone_max_retries")
                else 10
            )

            while not ready and retries < max_retries:
                try:
                    desc = await self.pc.describe_index(name=settings.pinecone_index_name)  # CHANGED
                    ready = getattr(desc.status, 'ready', False)
                    if not ready:
                        retries += 1
                        wait_time = min(2**retries, 60)  # Exponential backoff with 60s cap
                        logger.info(
                            f"Index not ready, waiting {wait_time}s... (attempt {retries}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.info("Index is ready")
                except Exception as e:
                    logger.error(f"Error checking index status: {e}")
                    retries += 1
                    await asyncio.sleep(2)

            if not ready:
                # Index creation started but didn't become ready - consider cleanup
                logger.error(f"Index not ready after {max_retries} retries")
                raise RuntimeError(f"Index not ready after {max_retries} retries")

        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            # Try to clean up the failed index
            try:
                await self.pc.delete_index(name=settings.pinecone_index_name)
                logger.info(f"Cleaned up failed index: {settings.pinecone_index_name}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to clean up index: {cleanup_err}")
            raise e

    async def query_index(
        self, embedding: List[float], top_k: int = 3, namespace: str = "knowledge-base", include_metadata: bool = True
    ) -> List[Dict]:
        """Query the Pinecone index for similar vectors."""
        # Ensure index is initialized
        if self.index is None:
            await self.initialize()

        try:
            # Validate index health if needed (with debouncing)
            await self._ensure_index_healthy()

            # Query the index
            response = await self.index.query(
                    vector=embedding,
                    top_k=top_k,
                    namespace=namespace,
                    include_metadata=include_metadata,
                ),
            return response.get("matches", [])
        except Exception as e:
            logger.error(f"Pinecone query failed: {str(e)}")
            return []

    async def _ensure_index_healthy(self):
        """Check index health with debouncing and concurrency control"""
        now = time.time()

        # Only check health if it's been more than 5 minutes since last check
        if now - self._last_healthy > 300:
            async with self._health_lock:  # Prevent concurrent health checks
                # Double-check after acquiring lock (another thread might have updated)
                if now - self._last_healthy > 300:
                    healthy = await self._check_index_health()
                    if not healthy:
                        raise ConnectionError("Unhealthy Pinecone index")

    async def _check_index_health(self) -> bool:
        """Check if the Pinecone index is healthy"""
        try:
            desc = await self.pc.describe_index(name=settings.pinecone_index_name)
            ready = getattr(desc.status, 'ready', False)
            self._last_healthy = time.time()
            logger.debug(f"Index health: ready={ready}")
            return ready
        except Exception as e:
            logger.critical(f"Health check failed: {str(e)}")
            return False

    async def close(self) -> None:
        """Cleanup Pinecone client session"""
        # CHANGED: close async client
        await self.pc.__aexit__(None, None, None)