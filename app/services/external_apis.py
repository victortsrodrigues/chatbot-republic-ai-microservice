import httpx
from app.config import settings
from app.utils.logger import logger
from typing import Optional

class ExternalAPIClient:
    def __init__(self):
        self.timeout = 10
        self.retries = 2

    async def get_room_data(self, room_id: str, query_type: str) -> Optional[dict]:
        """Query room availability or pricing API"""
        endpoint = f"{settings.room_api_url}/rooms/{room_id}"
        params = {"type": query_type}
        
        async with httpx.AsyncClient() as client:
            for _ in range(self.retries):
                try:
                    response = await client.get(
                        endpoint,
                        params=params,
                        headers={"Authorization": f"Bearer {settings.room_api_key}"},
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    return response.json()
                except Exception as e:
                    logger.error(f"Room API error: {str(e)}")
                    continue
        return None