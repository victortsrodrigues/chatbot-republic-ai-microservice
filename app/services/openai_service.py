from openai import OpenAI
from app.config import settings
from app.utils.logger import logger
from typing import List, Dict

class OpenAIHandler:
    def __init__(self):
      self.client = OpenAI(api_key=settings.openai_api_key)

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
        try:
            response = await self.client.chat.completions.create(
                model=settings.chat_model,
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Chat completion failed: {str(e)}")
            return "I'm having trouble answering that. Please try again later."
        
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