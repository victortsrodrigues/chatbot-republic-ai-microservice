from app.services.openai_service import OpenAIHandler
from app.services.pinecone_service import PineconeManager
from app.services.mongo_service import MongoDBClient
from app.config import settings
from typing import Optional, List, Dict
import re
import json


class RAGOrchestrator:
    def __init__(self):
        self.openai = OpenAIHandler()
        self.pinecone = PineconeManager()
        self.mongo = MongoDBClient()
        self.system_message = settings.default_system_message
        self.filter_template = """Extract filters as JSON from this query:
        {query}
        Use this schema:
        {{
            "price": {{"min": number, "max": number}},
            "features": list[str],
            "room_type": str,
            "size": {{"min": number, "max": number}}
        }}"""

    async def process_query(
        self,
        query: str,
        history: list,
        system_message: str = None,
        filter: Optional[dict] = None,
    ) -> dict:
        # Generate embedding and get context
        embedding = await self.openai.generate_embedding(query)
        context_results = await self.pinecone.query_index(
            embedding=embedding, filter=filter, top_k=5
        )

        # Extract all relevant information
        parsed_filters = await self._parse_filters(query)
        media_data = self._get_media_data(context_results)
        rooms_data = await self._fetch_rooms_data(context_results, parsed_filters)

        # Generate response
        response = await self._generate_response(
            query, context_results, rooms_data, history, 
            system_message or self.system_message
        )

        return self._merge_media_data(response, media_data)

    async def _parse_filters(self, query: str) -> dict:
        """Use OpenAI to extract structured filters from natural language"""
        prompt = self.filter_template.format(query=query)
        response = await self.openai.generate_chat_completion([{
            "role": "system",
            "content": "You are a skilled query parser. Return only valid JSON."
        }, {
            "role": "user",
            "content": prompt
        }])
        
        try:
            return json.loads(response.strip("` \n"))
        except json.JSONDecodeError:
            return self._fallback_filter_parsing(query)

    def _fallback_filter_parsing(self, query: str) -> dict:
        """Regex-based fallback for filter parsing"""
        filters = {}
        
        # Price parsing
        price_matches = re.findall(r'\$?(\d+)', query)
        if len(price_matches) >= 2:
            filters["price"] = {"min": int(price_matches[0]), "max": int(price_matches[1])}
        elif "cheaper than" in query:
            max_price = re.search(r'cheaper than \$?(\d+)', query)
            if max_price:
                filters["price"] = {"max": int(max_price.group(1))}
        
        # Feature extraction
        features = []
        for term in ["suite", "balcony", "ocean view"]:  # Extend this list
            if term in query.lower():
                features.append(term)
        if features:
            filters["features"] = features
            
        return filters

    async def _fetch_rooms_data(self, context: list, filters: dict) -> list:
        """Combine Pinecone context with MongoDB filters"""
        room_ids = list({
            c['metadata']['room_id'] 
            for c in context 
            if c['metadata'].get('type') == 'room'
        })
        
        if room_ids:
            return self.mongo.get_rooms_by_ids(room_ids, filters)
        return self.mongo.get_all_rooms(filters)
    
    def _get_media_data(self, context: list) -> Optional[dict]:
        return next(
            (
                {
                    "s3_object_key": c["metadata"]["s3_object_key"],
                    "media_type": c["metadata"]["media_type"],
                    "caption": c["metadata"].get("caption", ""),
                }
                for c in context
                if c["metadata"].get("type") == "media"
            ),
            None,
        )

    async def _generate_response(
        self,
        query: str,
        context: list,
        rooms_data: list,
        history: list,
        system_message: str,
    ) -> dict:
        context_str = "\n".join([c["metadata"]["text"] for c in context])
        rooms_str = "\n".join(
            [
                f"{r['room_id']}: {r['description']} (${r['price']}/night)"
                for r in rooms_data
            ]
        )

        messages = [
            {"role": "system", "content": system_message or self.system_message},
            *history,
            {
                "role": "user",
                "content": f"""Context:
                            {context_str}

                            Available Rooms:
                            {rooms_str}

                            Question: {query}""",
            },
        ]

        chat_response = await self.openai.generate_chat_completion(messages)
        return {
            "response": chat_response,
            "sources": [c["metadata"] for c in context],
            "requires_action": False,
        }

    def _merge_media_data(self, response: dict, media_data: Optional[dict]) -> dict:
        if media_data:
            response.update(
                {
                    "requires_action": True,
                    "action_type": "fetch_media",
                    "s3_object_key": media_data["s3_object_key"],
                    "media_type": media_data["media_type"],
                    "caption": media_data["caption"],
                }
            )
        return response
