from app.services.openai_service import OpenAIHandler
from app.services.pinecone_service import PineconeManager
from app.services.mongo_service import MongoDBClient
from app.config import settings
from app.utils.logger import logger
from typing import Optional, List, Dict
import re
import json
import asyncio


class RAGOrchestrator:
    def __init__(self):
        self.openai = OpenAIHandler()
        self.pinecone = PineconeManager()
        self.mongo = MongoDBClient()
        self.system_message = settings.default_system_message
        self.filter_template = """Convert this query to MongoDB JSON syntax:
                                {query}
                                Use this schema (all fields optional):
                                {{
                                    "price"?: {{
                                        "$lt"?: number,
                                        "$gt"?: number,
                                        "$lte"?: number,
                                        "$gte"?: number,
                                        "$eq"?: number
                                    }},
                                    "features"?: {{
                                        "$all"?: string[],
                                        "$in"?: string[],
                                        "$nin"?: string[]
                                    }},
                                    "room_type"?: string,
                                    "availability"?: boolean,
                                    "$or"?: [{{...}}],
                                    "$and"?: [{{...}}],
                                    "$nor"?: [{{...}}],
                                    "$text"?: {{
                                        "$search": string,
                                        "$language"?: string,
                                        "$caseSensitive"?: boolean
                                    }},
                                    "$geoWithin"?: {{
                                        "$geometry": {{
                                            "type": "Polygon",
                                            "coordinates": number[][][]
                                        }}
                                    }}
                                }}"""

    async def process_query(
        self,
        query: str,
        history: list,
        system_message: str = None,
        filter: Optional[dict] = None,
    ) -> dict:
        try:
            # Check query's moderation
            if await self.openai.check_moderation(query):
                return {"error": "Query not allowed"}

            # Correct typos using OpenAI
            # Add retry decorator to critical components
            query = await self._correct_typos_with_retry(query)

            # Normalize query
            query = self._normalize_query(query)

            # Generate embedding and get context
            embedding = await self.openai.generate_embedding(query)
            # Validate embedding result
            if not isinstance(embedding, list) or len(embedding) != 1536:
                raise ValueError("Invalid embedding format")

            context_results = await asyncio.wait_for(
                self.pinecone.query_index(embedding=embedding, filter=filter, top_k=3),
                timeout=5.0,
            )

            # NEW: Determine if room data is needed
            requires_room_data = await self._needs_room_data(query, context_results)

            # Extract all relevant information
            # Rooms
            parsed_filters = (
                await self._parse_filters(query) if requires_room_data else {}
            )
            rooms_data = (
                await self._fetch_rooms_data(context_results, parsed_filters)
                if requires_room_data
                else []
            )
            # Media
            media_data = (
                self._get_media_data(context_results, query)
                if await self._needs_media(query)
                else []
            )

            # Generate response
            response = await self._generate_response(
                query,
                context_results,
                rooms_data,
                history,
                system_message or self.system_message,
                needs_room_info=requires_room_data,
            )

            # Check response's moderation
            if await self.openai.check_moderation(response):
                return {"error": "Response not allowed"}

            return self._merge_media_data(response, media_data)

        except (asyncio.TimeoutError, ValueError) as e:
            logger.error(f"Pipeline failed at stage: {str(e)}")
            return {"error": "Pipeline failed"}

    async def _correct_typos_with_retry(self, query: str, retries=2):
        for attempt in range(retries):
            try:
                return await self._correct_typos(query)
            except Exception:
                if attempt == retries - 1:
                    raise

    async def _correct_typos(self, query: str) -> str:
        """Correct typos using OpenAI"""
        prompt = f"""Correct any typos in this query. Return only the corrected text.
        Original query: '{query}'
        Corrected query:"""

        try:
            response = await self.openai.generate_chat_completion(
                [
                    {
                        "role": "system",
                        "content": "You are a skilled proofreader. Only return the corrected text.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            return response.strip().strip('"')
        except Exception as e:
            logger.error(f"Typo correction failed: {str(e)}")
            return query  # Fallback to original

    # Add synonym mapping
    SYNONYM_MAP = {
        "bucks": "dollars",
        "accommodations": "rooms",
        "cottages": "suites",
        "photos": "pictures",
    }

    def _normalize_query(self, query: str) -> str:
        """Replace synonyms with canonical terms"""
        for synonym, canonical in self.SYNONYM_MAP.items():
            query = re.sub(rf"\b{synonym}\b", canonical, query, flags=re.IGNORECASE)
        return query

    # Intent detection methods
    async def _needs_room_data(self, query: str, context: list) -> bool:
        """Check if query requires room information"""
        # First check explicit context markers
        if any(c["metadata"].get("type") == "room" for c in context):
            return True

        # Then check query content
        prompt = f"""Should the response include room listings for this query?
        Query: {query}
        Answer ONLY 'YES' or 'NO'"""

        response = await self.openai.generate_chat_completion(
            [{"role": "user", "content": prompt}]
        )
        return "YES" in response.strip().upper()

    async def _parse_filters(self, query: str) -> dict:
        """Use OpenAI to extract structured filters from natural language"""
        prompt = f"""Extract EXACT filters from this query. Return empty JSON if none.
                Query: {query}
                {self.filter_template}"""
        response = await self.openai.generate_chat_completion(
            [
                {
                    "role": "system",
                    "content": "You are a skilled query parser. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        try:
            return json.loads(response.strip("` \n"))
        except json.JSONDecodeError:
            return self._fallback_filter_parsing(query)

    def _fallback_filter_parsing(self, query: str) -> dict:
        """Regex-based fallback for filter parsing"""
        filters = {}

        # Price parsing
        price_matches = re.findall(r"\$?(\d+)", query)
        if len(price_matches) >= 2:
            filters["price"] = {
                "min": int(price_matches[0]),
                "max": int(price_matches[1]),
            }
        elif "cheaper than" in query:
            max_price = re.search(r"cheaper than \$?(\d+)", query)
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

    async def _fetch_rooms_data(self, context: list, parsed_filters: dict) -> list:
        """Combine Pinecone context with MongoDB filters"""
        room_ids = list(
            {
                c["metadata"]["room_id"]
                for c in context
                if c["metadata"].get("type") == "room"
            }
        )

        if room_ids:
            return self.mongo.get_rooms_by_ids(room_ids, parsed_filters)
        return self.mongo.get_all_rooms(parsed_filters)

    async def _needs_media(self, query: str) -> bool:
        """Check if media should be attached"""
        prompt = f"""Does this query require visual media?
        Query: {query}
        Answer ONLY 'YES' or 'NO'"""

        response = await self.openai.generate_chat_completion(
            [{"role": "user", "content": prompt}]
        )
        return "YES" in response.strip().upper()

    async def _get_media_data(self, context: list, query: str) -> List[dict]:
        """Retrieve all relevant media files based on explicit or proactive triggers"""
        media_list = []

        # 1. Check for explicit media requests
        explicit_media = [
            {
                "s3_object_key": c["metadata"]["s3_object_key"],
                "media_type": c["metadata"]["media_type"],
                "caption": c["metadata"].get("caption", ""),
            }
            for c in context
            if c["metadata"].get("type") == "media"
        ]
        media_list.extend(explicit_media)

        # 2. Check for proactive media suggestions
        for c in context:
            if c["metadata"].get("suggest_media", False):
                if await self._should_attach_media(query, c["metadata"]):
                    media_list.append(
                        {
                            "s3_object_key": c["metadata"]["s3_object_key"],
                            "media_type": c["metadata"]["media_type"],
                            "caption": c["metadata"].get("caption", ""),
                        }
                    )

        # 3. Deduplicate media entries
        unique_media = []
        seen_keys = set()
        for media in media_list:
            if media["s3_object_key"] not in seen_keys:
                unique_media.append(media)
                seen_keys.add(media["s3_object_key"])

        return unique_media

    async def _should_attach_media(self, query: str, metadata: dict) -> bool:
        """Determine if media should be attached proactively"""
        # Check metadata triggers
        if any(
            trigger in query.lower() for trigger in metadata.get("media_triggers", [])
        ):
            return True

        # Use OpenAI for intent analysis
        prompt = f"""Should the response include media for this query?
        Query: {query}
        Context: {metadata.get('text', '')}
        Answer ONLY 'YES' or 'NO'"""

        response = await self.openai.generate_chat_completion(
            [{"role": "user", "content": prompt}]
        )

        return "YES" in response.strip().upper()

    async def _generate_response(
        self,
        query: str,
        context: list,
        rooms_data: list,
        history: list,
        system_message: str,
        needs_room_info: bool,
    ) -> dict:

        context_str = "\n".join([c["metadata"]["text"] for c in context])

        # Conditionally include room data
        rooms_str = ""
        if needs_room_info and rooms_data:
            rooms_str = "\nAvailable Rooms:\n" + "\n".join(
                f"{r['room_id']}: {r['description']} (${r['price']}/night)"
                for r in rooms_data
            )

        messages = [
            {"role": "system", "content": system_message},
            *history,
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\n{rooms_str}\n\nQuestion: {query}",
            },
        ]

        chat_response = await self.openai.generate_chat_completion(messages)
        return {
            "response": chat_response,
            "sources": [c["metadata"] for c in context],
            "requires_action": False,
        }

    def _merge_media_data(self, response: dict, media_data: List[dict]) -> dict:
        if media_data:
            response.update(
                {
                    "requires_action": True,
                    "action_type": "fetch_media",
                    "media_list": media_data,  # Replace single media fields with a list
                }
            )
        return response
