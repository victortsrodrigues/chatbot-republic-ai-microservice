import time
from pinecone import Pinecone, ServerlessSpec
from app.config import settings
from app.utils.logger import logger
from typing import List, Optional, Dict

class PineconeManager:
    def __init__(self):
        self.index = self._initialize_index()

    def _initialize_index(self):
        """Initialize Pinecone index, creating it if it doesn't exist."""
        try:
            # Initialize Pinecone client
            pc = Pinecone(api_key=settings.pinecone_api_key)
            
            # Check if index exists, create if it doesn't
            if settings.pinecone_index_name not in pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {settings.pinecone_index_name}")
                pc.create_index(
                    settings.pinecone_index_name,
                    dimension=1536,  # Default for OpenAI embeddings
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=settings.pinecone_cloud,
                        region=settings.pinecone_region
                    )
                )
                
                # Wait for index to be ready
                while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
                    logger.debug("Waiting for Pinecone index to be ready...")
                    time.sleep(1)
            
            logger.info(f"Pinecone index {settings.pinecone_index_name} is ready")
            return pc.Index(settings.pinecone_index_name)
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {str(e)}")
            raise

    async def query_index(
        self,
        embedding: List[float],
        filter: Optional[Dict] = None,
        top_k: int = 3
    ) -> List[Dict]:
        """Query the Pinecone index for similar vectors."""
        try:
            response = self.index.query(
                vector=embedding,
                filter=filter,
                top_k=top_k,
                include_metadata=True
            )
            return response.get("matches", [])
        except Exception as e:
            logger.error(f"Pinecone query failed: {str(e)}")
            return []

    async def upsert_embeddings(self, vectors: List[Dict]):
        try:
            self.index.upsert(vectors=vectors)
            logger.info(f"Successfully upserted {len(vectors)} embeddings")
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {str(e)}")
            raise