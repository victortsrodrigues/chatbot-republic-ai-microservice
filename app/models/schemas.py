from pydantic import BaseModel
from typing import List, Optional, Dict

class RAGQuery(BaseModel):
    query: str
    user_id: str
    history: list[Dict] = []  # Conversation history
    system_message: Optional[str] = None  # Custom system message
    filter: Optional[Dict] = None  # Pinecone metadata filter

class RAGResponse(BaseModel):
    response: str
    sources: List[Dict]  # Metadata from Pinecone
    requires_action: bool
    action_type: Optional[str] = None  # 'fetch_media' or 'room_query'
    media_list: Optional[List[Dict]] = None  # Replace single media fields
    # action_parameters: Optional[Dict] = None  # {type: 'availability'/'pricing', room_id: '123'}
    # s3_object_key: Optional[str] = None
    # media_type: Optional[str] = None
    # caption: Optional[str] = None