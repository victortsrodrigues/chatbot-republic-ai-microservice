from app.services.openai_service import OpenAIHandler
from app.services.pinecone_service import PineconeManager
from app.services.mongo_service import MongoDBClient
from app.config import settings
from app.utils.logger import logger
from typing import Optional, List, Dict, Any
import re
import json
import asyncio
import time
from functools import wraps


# Decorator for retrying critical operations
def async_retry(max_retries=3, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            backoff = 1
            
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    last_exception = e
                    logger.warning(f"Retry {retries}/{max_retries} for {func.__name__}: {str(e)}")
                    if retries < max_retries:
                        await asyncio.sleep(backoff)
                        backoff *= backoff_factor
            
            logger.error(f"All retries failed for {func.__name__}: {last_exception}")
            raise last_exception
        
        return wrapper
    return decorator


class RAGOrchestrator:
    _instance = None
    _init_lock = asyncio.Lock()
    
    # Implement singleton pattern similar to other services
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize these attributes but actual clients will be set up in initialize()
            cls._instance.openai = None
            cls._instance.pinecone = None
            cls._instance.mongo = None
            cls._instance.system_message = settings.default_system_message
            cls._instance.filter_template = """Converta esta query para a sintaxe JSON do MongoDB:
                                    {query}
                                    Use este esquema (todos os campos são opcionais):
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
            
            # Circuit breaker pattern for RAG service
            cls._instance._circuit_state = "closed"
            cls._instance._consecutive_failures = 0
            cls._instance._failure_threshold = 5
            cls._instance._last_failure = 0
            
            # Request semaphore to manage concurrent operations
            cls._instance._request_semaphore = asyncio.Semaphore(
                getattr(settings, 'rag_max_concurrent_requests', 20)
            )
            
            # Add cache for common queries
            cls._instance._response_cache = {}
            cls._instance._cache_ttl = getattr(settings, 'rag_cache_ttl', 300)  # 5 minutes
            
            # Add initialization flag
            cls._instance._initialized = False
        
        return cls._instance
    
    # Add proper async initialization
    async def initialize(self):
        """Initialize all service connections safely"""
        if self._initialized:
            return
            
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                # Initialize services
                self.openai = OpenAIHandler()
                await self.openai.initialize()
                
                self.pinecone = PineconeManager()
                await self.pinecone.initialize()
                
                self.mongo = MongoDBClient()
                
                self._initialized = True
                logger.info("RAG Orchestrator successfully initialized")
            except Exception as e:
                logger.critical(f"RAG Orchestrator initialization failed: {str(e)}")
                self._handle_failure()
                raise

    async def process_query(
        self,
        query: str,
        history: list,
        system_message: str = None,
    ) -> dict:
        """Process user query with concurrency control and circuit breaker"""
        # Check circuit breaker state
        if self._circuit_state == "open":
            if time.time() - self._last_failure < getattr(settings, 'rag_circuit_timeout', 120):
                logger.warning("Circuit breaker open - rejecting request")
                return {"error": "Service temporarily unavailable", "circuit_open": True}
            else:
                # Reset circuit breaker after timeout
                self._circuit_state = "closed"
                self._consecutive_failures = 0
        
        # Initialize services if needed
        if not self._initialized:
            await self.initialize()
        
        # Use request semaphore to limit concurrent requests
        async with self._request_semaphore:
            # Check cache for identical queries
            cache_key = f"{query}:{str(history[-5:] if history else [])}:{system_message or self.system_message}"
            cached_response = self._check_cache(cache_key)
            if cached_response:
                logger.info("Cache hit for query")
                return cached_response
            
            try:
                # Check query's moderation
                if await self.openai.check_moderation(query):
                    return {"error": "Query not allowed"}

                # Correct typos using OpenAI with retries
                query = await self._correct_typos_with_retry(query)

                # Normalize query
                query = self._normalize_query(query)

                # Generate embedding and get context
                embedding = await self.openai.generate_embedding(query)
                # Validate embedding result
                if not isinstance(embedding, list) or len(embedding) != 1536:
                    raise ValueError("Invalid embedding format")

                # Use timeouts consistently
                context_results = await asyncio.wait_for(
                    self.pinecone.query_index(embedding=embedding, top_k=3),
                    timeout=5.0,
                )

                # Determine if room data is needed
                requires_room_data, requires_media = (
                    await self._decide_inclusions_room_data_and_media(
                        query, context_results
                    )
                )

                # Extract all relevant information
                # Rooms
                parsed_filters = (
                    await self._parse_filters(query) if requires_room_data else {}
                )
                
                # Use asyncio.gather for concurrent fetching
                if requires_room_data:
                    rooms_data = await asyncio.wait_for(
                        self.mongo.get_all_rooms(parsed_filters),
                        timeout=3.0
                    )
                else:
                    rooms_data = []
                    
                # Media
                media_data = self._get_media_data(rooms_data) if requires_media else []

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
                if await self.openai.check_moderation(response.get("response", "")):
                    return {"error": "Response not allowed"}

                final_response = self._merge_media_data(response, media_data)
                
                # Store in cache
                self._update_cache(cache_key, final_response)
                
                # Reset failure counter on success
                self._consecutive_failures = 0
                
                return final_response

            except (asyncio.TimeoutError, ValueError) as e:
                logger.error(f"Pipeline failed at stage: {str(e)}")
                await self._handle_failure()
                return {"error": "Pipeline failed", "detail": str(e)}
            except Exception as e:
                logger.error(f"Unexpected error in RAG pipeline: {str(e)}")
                await self._handle_failure()
                return {"error": "Internal service error", "detail": str(e)}
    
    # Add method to handle failures with circuit breaker
    async def _handle_failure(self):
        """Update failure tracking and possibly open circuit breaker"""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._circuit_state = "open"
            self._last_failure = time.time()
            logger.error(f"Circuit breaker opened after {self._consecutive_failures} failures")
    
    # Add caching methods
    def _check_cache(self, key: str) -> Dict[str, Any]:
        """Check if response is in cache and not expired"""
        if key in self._response_cache:
            timestamp, response = self._response_cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return response
            else:
                # Clean expired cache entry
                del self._response_cache[key]
        return None
    
    def _update_cache(self, key: str, response: Dict[str, Any]):
        """Store response in cache with timestamp"""
        # Only cache successful responses
        if "error" not in response:
            self._response_cache[key] = (time.time(), response)
            
            # Clean old entries if cache is too large
            if len(self._response_cache) > 1000:  # Arbitrary limit to prevent memory issues
                oldest_key = min(self._response_cache.keys(), 
                                 key=lambda k: self._response_cache[k][0])
                del self._response_cache[oldest_key]

    @async_retry(max_retries=2)
    async def _correct_typos_with_retry(self, query: str):
        """Apply retry decorator to typo correction"""
        return await self._correct_typos(query)

    async def _correct_typos(self, query: str) -> str:
        """Correct typos using OpenAI"""
        prompt = f"""Corrija quaisquer erros de digitação nesta query. Retorne apenas o texto corrigido.
        Query original: '{query}'
        Query corrigida:"""

        try:
            response = await self.openai.generate_chat_completion(
                [
                    {
                        "role": "system",
                        "content": "Você é um revisor habilidoso. Devolva apenas o texto corrigido.",
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
        "valor": "preço",
        "acomodações": "quartos",
        "lugares": "quartos",
    }

    def _normalize_query(self, query: str) -> str:
        """Replace synonyms with canonical terms"""
        for synonym, canonical in self.SYNONYM_MAP.items():
            query = re.sub(rf"\b{synonym}\b", canonical, query, flags=re.IGNORECASE)
        return query

    # Intent detection methods
    async def _decide_inclusions_room_data_and_media(
        self, query: str, context: list
    ) -> tuple[bool, bool]:
        """
        Ask GPT whether to include room metadata and/or media links.
        Returns (include_room_data, include_media_links).
        """
        # First check explicit context markers
        relevant_rooms = [
            match
            for match in context
            if match["score"] >= 0.8 and match["metadata"].get("type") == "room"
        ]
        if relevant_rooms:
            # If we already know rooms are relevant, still let GPT decide media
            include_room = True
        else:
            include_room = None  # defer to model

        # If no explicit markers, use OpenAI to analyze intent
        prompt = [
            {
                "role": "system",
                "content": """
Você é um mecanismo de decisão. Dada uma pergunta de um usuário e um contexto recuperado,
você deve decidir duas coisas para a resposta final do chatbot:
1) include_room_data: se deve incluir informações do quarto (tamanho, preço, recursos etc.).
2) include_media: se deve incluir mídia (imagens, vídeos).

Responda com um objeto JSON exatamente assim:
{
  "include_room_data": true|false,
  "include_media": true|false
}
""",
            },
            {
                "role": "user",
                "content": f"""
Query do usuário:
"{query}"

Contexto recuperado:
{context}
""",
            },
        ]

        response = await self.openai.generate_chat_completion(prompt)
        # Extract and parse the JSON decision
        content = response.strip()
        try:
            decision = json.loads(content)
        except json.JSONDecodeError:
            # fallback defaults
            return bool(include_room), False

        # Merge explicit-room check if we short-circuited above
        include_room_final = (
            include_room
            if include_room is not None
            else bool(decision.get("include_room_data", False))
        )
        include_media = bool(decision.get("include_media", False))

        return include_room_final, include_media

    @async_retry(max_retries=2)
    async def _parse_filters(self, query: str) -> dict:
        """Use OpenAI to extract structured filters from natural language with retry logic"""
        prompt = f"""Extract EXACT filters from this query. Return empty JSON if none.
                Query: {query}
                {self.filter_template}"""
        response = await self.openai.generate_chat_completion(
            [
                {
                    "role": "system",
                    "content": "Você é um analisador de query habilidoso. Retorne apenas JSON válido.",
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
                "$gte": int(price_matches[0]),
                "$lte": int(price_matches[1]),
            }
        elif "mais barato" in query:
            max_price = re.search(r"mais barato \$?(\d+)", query)
            if max_price:
                filters["price"] = {"$lte": int(max_price.group(1))}

        # Feature extraction
        features = []
        for term in ["suíte", "varanda", "vista"]:  # Extend this list
            if term in query.lower():
                features.append(term)
        if features:
            filters["features"] = {"$all": features}

        # Availability
        if "disponível" in query.lower():
            filters["availability"] = True

        return filters

    @async_retry(max_retries=2)
    async def _generate_response(
        self,
        query: str,
        context: list,
        rooms_data: list,
        history: list,
        system_message: str,
        needs_room_info: bool,
    ) -> dict:
        """Generate the final response with OpenAI with retry logic"""
        context_str = "\n".join([str(c.get("metadata", "")) for c in context])

        # Conditionally include room data
        rooms_str = ""
        if needs_room_info and rooms_data:
            rooms_str = "\nAvailable Rooms:\n" + "\n".join(
                f"{r.get('room_id', 'unknown')}: {r.get('description', 'No description')} (${r.get('price', 'N/A')}/month)"
                for r in rooms_data
            )

        # Limit history to prevent token overflow
        limited_history = history[-settings.max_history_length:]
        
        messages = [
            {"role": "system", "content": system_message},
            *limited_history,
            {
                "role": "user",
                "content": f"Context:\n{context_str}\n\n{rooms_str}\n\nQuestion: {query}",
            },
        ]

        chat_response = await self.openai.generate_chat_completion(messages)
        return {
            "response": chat_response,
            "sources": [c.get("metadata", {}) for c in context],
            "requires_action": False,
        }

    def _get_media_data(self, rooms_data: list = None) -> List[dict]:
        """Extract S3 object keys from rooms data"""
        # If no rooms data, return empty list
        if not rooms_data:
            return []

        media_keys = []

        # Extract s3_object_keys from each room
        for room in rooms_data:
            if "s3_object_key" in room:
                # If it's a single string
                if isinstance(room["s3_object_key"], str):
                    media_keys.append(room["s3_object_key"])
                # If it's a list
                elif isinstance(room["s3_object_key"], list):
                    media_keys.extend(room["s3_object_key"])

        # Remove duplicates while preserving order
        return list(dict.fromkeys(media_keys))

    def _merge_media_data(self, response: dict, media_data: List[dict]) -> dict:
        """Merge media data into the response"""
        if media_data:
            response.update(
                {
                    "requires_action": True,
                    "action_type": "fetch_media",
                    "media_list": media_data,  # Replace single media fields with a list
                }
            )
        return response
        
    async def close(self):
        """Gracefully close all connections"""
        try:
            if self.openai:
                await self.openai.close()
            if self.pinecone:
                await self.pinecone.close()
            # MongoDB client will be closed by its own shutdown handler
            
            self._initialized = False
            logger.info("RAG Orchestrator connections closed")
        except Exception as e:
            logger.error(f"Error during RAG service shutdown: {str(e)}")