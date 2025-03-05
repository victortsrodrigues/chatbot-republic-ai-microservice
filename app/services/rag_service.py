from app.services.openai_service import OpenAIHandler
from app.services.pinecone_service import PineconeManager
from app.services.external_apis import ExternalAPIClient
from app.config import settings
from typing import Optional


class RAGOrchestrator:
    def __init__(self):
        self.openai = OpenAIHandler()
        self.pinecone = PineconeManager()
        self.system_message = settings.default_system_message  # From config
        self.api_client = ExternalAPIClient()

    async def process_query(
        self,
        query: str,
        history: list,
        system_message: str = None,
        filter: Optional[dict] = None,
    ) -> dict:
        # Generate embedding
        embedding = await self.openai.generate_embedding(query)

        # Query Pinecone
        context_results = await self.pinecone.query_index(
            embedding=embedding, filter=filter, top_k=3
        )

        # Check for required API actions
        action_data = self._detect_actions(context_results)
        if action_data:
            api_result = await self.api_client.get_room_data(
                action_data['room_id'], 
                action_data['type']
            )
            return self._format_api_response(api_result, action_data, context_results)
        
        # Build messages for chat completion
        messages = self._build_messages(
            query=query,
            context=context_results,
            history=history,
            system_message=system_message,
        )

        chat_response = await self.openai.generate_chat_completion(messages)

        return self._format_response(chat_response, context_results)

    def _build_messages(self, query: str, context: list, history: list, system_message: str):
        # 1. System message (custom or default)
        messages = [
            {"role": "system", "content": system_message or self.system_message}
        ]

        # 2. Add conversation history
        messages.extend(history)  # Expects format [{role: "user|assistant", content: "..."}]

        # 3. Add context and current query
        context_str = "\n".join([c["metadata"]["text"] for c in context])
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nQuestion: {query}",
                }
            ]
        )

        return messages

    def _detect_actions(self, context_results: list) -> Optional[dict]:
        """Detect if any context requires live API call"""
        for c in context_results:
            metadata = c['metadata']
            if metadata.get('requires_live_query'):
                return {
                    "type": metadata.get('query_type', 'availability'),
                    "room_id": metadata['room_id']
                }
        return None

    def _format_api_response(self, api_data: dict, action_data: dict, context: list) -> dict:
        """Format response with live API data"""
        if not api_data:
            return {
                "response": "Unable to retrieve current information. Please try again later.",
                "sources": [c['metadata'] for c in context],
                "requires_action": False,
                "action_type": None,
                "action_parameters": None,
                "s3_object_key": None,
                "media_type": None,
                "caption": None
            }

        if action_data['type'] == 'availability':
            availability_status = 'available' if api_data['available'] else 'unavailable'
            response_text = f"Room {action_data['room_id']} is {availability_status}"
        else:
            response_text = f"Current price for room {action_data['room_id']}: ${api_data['price']}"
        
        return {
            "response": response_text,
            "sources": [c['metadata'] for c in context],
            "requires_action": False,
            "action_type": None,
            "action_parameters": None,
            "s3_object_key": None,
            "media_type": None,
            "caption": None
        }
    
    def _format_response(self, response: str, context: list) -> dict:
        # Add logic to detect required actions from metadata
        media_data = next((
            {
                "s3_object_key": c['metadata']['s3_object_key'],
                "media_type": c['metadata']['media_type'],
                "caption": c['metadata'].get('caption', '')
            } 
            for c in context if c['metadata'].get('type') == 'media'
        ), None)

        # Add action detection logic here
        
        return {
            "response": response,
            "sources": [c['metadata'] for c in context],
            "requires_action": bool(media_data),
            "action_type": "fetch_media" if media_data else None,
            "s3_object_key": media_data["s3_object_key"] if media_data else None,
            "media_type": media_data["media_type"] if media_data else None,
            "caption": media_data["caption"] if media_data else None
        }
        