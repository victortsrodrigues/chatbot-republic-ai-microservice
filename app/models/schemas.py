from pydantic import BaseModel
from typing import List, Optional

class RAGQuery(BaseModel):
    query: str
    user_id: str
    history: list[dict] = []  # New: Conversation history
    system_message: Optional[str] = None  # New: Custom system message
    filter: Optional[dict] = None

class RAGResponse(BaseModel):
    response: str
    sources: List[dict]  # Metadata from Pinecone
    requires_action: bool
    action_type: Optional[str] = None  # e.g., "fetch_media", "live_query"
    action_parameters: Optional[dict] = None
    media_pointer: Optional[str] = None