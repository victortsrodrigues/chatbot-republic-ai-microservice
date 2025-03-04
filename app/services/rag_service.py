from app.services.openai_service import OpenAIHandler
from app.services.pinecone_service import PineconeManager
from app.config import settings

class RAGOrchestrator:
    def __init__(self):
        self.openai = OpenAIHandler()
        self.pinecone = PineconeManager()
        self.system_message = settings.default_system_message  # From config
    
    async def process_query(self, query: str, history: list, system_message: str = None) -> dict:
        # Generate embedding
        embedding = await self.openai.generate_embedding(query)
        
        # Query Pinecone
        context_results = await self.pinecone.query_index(
            embedding=embedding,
            filter=filter,
            top_k=3
        )
        
        # Build messages for chat completion
        messages = self._build_messages(
            query=query,
            context=context_results,
            history=history,
            system_message=system_message
        )
        
        chat_response = await self.openai.generate_chat_completion(messages)
        
        return self._format_response(chat_response, context_results)
    
    def _build_prompt(self, query: str, context: list, history: list, system_message: str):
        # 1. System message (custom or default)
        messages = [{
            "role": "system",
            "content": system_message or self.system_message
        }]

        # 2. Add conversation history
        messages.extend(history)  # Expects format [{role: "user|assistant", content: "..."}]

        # 3. Add context and current query
        context_str = "\n".join([c['metadata']['text'] for c in context])
        messages.extend([
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\nQuestion: {query}"
            }
        ])
        
        return messages
    
    def _format_response(self, response: str, context: list) -> dict:
        # Add logic to detect required actions from metadata
        return {
            "response": response,
            "sources": [c['metadata'] for c in context],
            # Add action detection logic here
        }