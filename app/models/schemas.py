from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Union

class RAGQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    user_id: str = Field(..., min_length=1)
    history: list[Dict] = []  # Conversation history
    system_message: Optional[str] = None  # Custom system message
    
    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Query cannot be empty')
        return v
    
    @field_validator('user_id')
    @classmethod
    def user_id_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('User ID cannot be empty')
        return v
    
    @field_validator('history')
    @classmethod
    def validate_history(cls, v):
        # Limit history length to prevent extremely large payloads
        if len(v) > 50:  # Arbitrary limit, adjust as needed
            raise ValueError('Conversation history too long')
        
        # Validate each history item has required fields
        for item in v:
            if 'role' not in item or 'content' not in item:
                raise ValueError('History items must contain role and content fields')
            
            # Validate roles
            if item['role'] not in ['user', 'assistant', 'system']:
                raise ValueError(f"Invalid role: {item['role']}")
        
        return v

class RAGResponse(BaseModel):
    response: str
    requires_action: bool
    action_type: Optional[str] = None  # 'fetch_media'
    media_list: Optional[List[Union[str, Dict]]] = None  # Replace single media fields