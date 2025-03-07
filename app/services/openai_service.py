from openai import OpenAI
from app.config import settings
from app.utils.logger import logger
from typing import List, Dict
import time

class OpenAIHandler:
    def __init__(self):
      self.client = OpenAI(api_key=settings.openai_api_key)
      self._circuit_state = "closed"
      self._last_failure = None

    async def generate_embedding(self, text: str) -> List[float]:
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=settings.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise

    async def generate_chat_completion(self, messages: List[Dict]) -> str:
        if self._circuit_state == "open":
            if time.time() - self._last_failure < 60:
                logger.warning("Circuit breaker open")
                return ""
        
        try:
            # Validate input structure
            if not isinstance(messages, list) or len(messages) == 0:
                raise ValueError("Invalid messages format")
            
            response = await self.client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,
                temperature=0
            )
            self._circuit_state = "closed"
            return response.choices[0].message.content.strip()
        except Exception as e:
            self._handle_error(e)
            raise
        
    async def check_moderation(self, input: str) -> bool:
        try:
            response = await self.client.moderations.create(
                model="omni-moderation-latest",
                input=input
            )
            return response.results[0].flagged
        except Exception as e:
            logger.error(f"Moderation check failed: {str(e)}")
            return False
        
    def _handle_error(self, error):
        self._circuit_state = "open"
        self._last_failure = time.time()
        logger.error(f"OpenAI API error: {str(error)}")