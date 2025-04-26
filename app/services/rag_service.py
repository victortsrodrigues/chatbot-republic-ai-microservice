from app.services.openai_service import OpenAIHandler
from app.services.pinecone_service import PineconeManager
from app.services.mongo_service import MongoDBClient
from app.config import settings
from app.utils.logger import logger
from typing import Optional, List, Dict, Tuple
import re
import json
import asyncio
import time
from functools import wraps
from cachetools import TTLCache


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


class CircuitBreaker:
    """Encapsulated circuit breaker logic with state management"""
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 120):
        self.state = "closed"
        self.failure_count = 0
        self.last_failure = 0.0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout

    def should_reject(self) -> bool:
        if self.state == "open":
            if time.time() - self.last_failure < self.reset_timeout:
                return True
            # Auto-reset after timeout
            self.state = "closed"
            self.failure_count = 0
        return False

    def record_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.last_failure = time.time()
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")


class RAGOrchestrator:
    _instance = None
    _init_lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_state()
        return cls._instance

    def _initialize_state(self):
        """Initialize all instance variables in one place"""
        self.openai: Optional[OpenAIHandler] = None
        self.pinecone: Optional[PineconeManager] = None
        self.mongo: Optional[MongoDBClient] = None
        
        # Circuit breaker with configurable settings
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=settings.rag_circuit_failure_threshold,
            reset_timeout=settings.rag_circuit_reset_timeout
        )
        
        # Configured concurrency limits
        self._request_semaphore = asyncio.Semaphore(
            getattr(settings, 'rag_max_concurrent_requests', 20)
        )
        
        # TTL-based caching with size limits from settings
        cache_maxsize = getattr(settings, 'rag_cache_maxsize', 1000)
        cache_ttl = getattr(settings, 'rag_cache_ttl', 300)
        self._response_cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        
        # System configuration
        self.system_message = settings.default_system_message
        self.filter_template = """Converta esta query para a sintaxe JSON do MongoDB:
                                    {query}
                                    Sua resposta deve ser um json válido seguindo a estrutura abaixo (todos os campos são opcionais):
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
        self._initialized = False
        self._embedding_dim = getattr(settings, 'embedding_dimension', 1536)

        self.synonym_map = getattr(
            settings,
            'query_synonyms',
            {"valor": "preço", "lugares": "quartos"}
        )
        
    # Add proper async initialization
    async def initialize(self):
        """Async-safe service initialization"""
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
                logger.info("RAG Orchestrator initialized successfully")
            except Exception as e:
                logger.critical(f"Initialization failed: {str(e)}")
                self.circuit_breaker.record_failure()
                raise

    @async_retry(max_retries=3)
    async def process_query(self, query: str, history: list, system_message: str = None, user_id: str = None) -> dict:
        """Process user query with enhanced concurrency and error handling"""
        # Circuit breaker check
        if self.circuit_breaker.should_reject():
            logger.warning("Circuit breaker open - rejecting request")
            return {"error": "Service temporarily unavailable", "circuit_open": True}
        
        if not self._initialized:
            await self.initialize()
        
        async with self._request_semaphore:
            cache_key = self._generate_cache_key(query, history, system_message, user_id)
            if cached := self._response_cache.get(cache_key):
                logger.info("Cache hit for query from user {user_id}")
                return cached

            try:
                return await self._process_query_unsafe(query, history, system_message, cache_key)
            except (asyncio.TimeoutError, ValueError) as e:
                logger.error(f"Pipeline failure for user {user_id}: {str(e)}")
                self.circuit_breaker.record_failure()
                return {"error": "Processing timeout", "detail": str(e)}
            except Exception as e:
                logger.error(f"Unexpected error for user {user_id}: {str(e)}")
                self.circuit_breaker.record_failure()
                return {"error": "Internal error", "detail": str(e)}

    async def _process_query_unsafe(self, query: str, history: list, system_message: str, cache_key: str) -> dict:
        """Core processing logic without error handling"""
        if await self.openai.check_moderation(query):
            logger.warning(f"Moderation failed for query: {query[:50]}...")
            return {"error": "Query content policy violation"}

        query = await self._correct_typos(query)
        query = self._normalize_query(query)

        # Parallel execution of embedding and initial processing
        embedding_task = asyncio.create_task(self.openai.generate_embedding(query))
        
        # Wait for embedding and then get context
        embedding = await embedding_task
        if not self._validate_embedding(embedding):
            raise ValueError("Invalid embedding dimensions")
            
        context_results = await asyncio.wait_for(
            self.pinecone.query_index(embedding=embedding, top_k=3),
            timeout=getattr(settings, 'pinecone_query_timeout', 5.0)
        )

        include_room, include_media = await self._decide_inclusions(query, context_results)
        
        # Parallel data fetching based on inclusion decisions
        rooms_data = []
        if include_room:
            parsed_filters = await self._parse_filters(query)
            rooms_data = await asyncio.wait_for(
                self.mongo.get_all_rooms(parsed_filters),
                timeout=getattr(settings, 'mongo_query_timeout', 3.0)
            )

        response = await self._generate_response(
            query, context_results, rooms_data, 
            history, system_message or self.system_message, include_room
        )

        if await self.openai.check_moderation(response.get("response", "")):
            logger.warning(f"Moderation failed for response: {response['response'][:50]}...")
            return {"error": "Response content policy violation"}

        # Add media data if needed
        media_data = self._get_media_data(rooms_data) if include_media else []
        
        final_response = self._merge_media_data(response, media_data)
        
        # Store in cache for future use
        self._response_cache[cache_key] = final_response
        return final_response

    def _generate_cache_key(self, query: str, history: list, system_message: str, user_id: str) -> str:
        """Generate consistent cache key with hash for efficiency"""
        history_str = json.dumps(history[-5:] if history else [], sort_keys=True)
        return f"{user_id}:{hash(query)}:{hash(history_str)}:{hash(system_message or self.system_message)}"

    def _validate_embedding(self, embedding: List[float]) -> bool:
        """Validate embedding dimensions using configurable setting"""
        return isinstance(embedding, list) and len(embedding) == self._embedding_dim
 

    @async_retry(max_retries=getattr(settings, 'typo_correction_retries', 2))
    async def _correct_typos(self, query: str) -> str:
        """Correct query typos using OpenAI with configurable retries"""
        try:
            system_msg = getattr(
                settings, 
                'typo_correction_system_message',
                "Você é um revisor habilidoso. Devolva apenas o texto corrigido."
            )
            response = await self.openai.generate_chat_completion(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Corrija erros de digitação: '{query}'"}
                ],
                temperature=getattr(settings, 'typo_correction_temperature', 0.1)
            )
            return response.strip().strip('"')
        except Exception as e:
            logger.error(f"Typo correction failed: {str(e)} - Using original query")
            return query  # Graceful fallback

    def _normalize_query(self, query: str) -> str:
        """Normalize query terms using configurable synonym map"""
        for synonym, canonical in self.synonym_map.items():
            query = re.sub(
                rf'\b{re.escape(synonym)}\b', 
                canonical, 
                query, 
                flags=re.IGNORECASE
            )
        return query

    # Intent detection methods
    async def _decide_inclusions(
        self, 
        query: str, 
        context: List[Dict]
    ) -> Tuple[bool, bool]:
        """Determine data inclusions with configurable relevance threshold"""
        relevance_threshold = getattr(settings, 'context_relevance_threshold', 0.8)
        relevant_rooms = [
            match for match in context
            if match["score"] >= relevance_threshold
            and match["metadata"].get("type") == "room"
        ]

        # If relevant rooms exist, automatically include room data
        include_room_explicit = bool(relevant_rooms)

        # Use OpenAI to analyze intent for more subtle cases
        decision_prompt = [
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

        response = await self.openai.generate_chat_completion(decision_prompt)
        
        try:
            decision = json.loads(response.strip())
            return (
                include_room_explicit or bool(decision.get("include_room_data", False)),
                bool(decision.get("include_media", False))
            )
        except json.JSONDecodeError:
            logger.warning("Failed to parse inclusion decision - Using defaults")
            return include_room_explicit, False

    @async_retry(max_retries=getattr(settings, 'filter_parse_retries', 2))
    async def _parse_filters(self, query: str) -> Dict:
        """Parse filters with configurable template and validation"""
        try:
            response = await self.openai.generate_chat_completion(
                [
                    {
                        "role": "system",
                        "content": getattr(
                            settings,
                            'filter_parser_system_message',
                            "Você é um analisador de query habilidoso. Retorne apenas JSON válido. "
                            "Certifique-se de que o JSON esteja bem formatado e siga o esquema fornecido."
                            "Caso não seja especificada consulta a quartos ocupados, considere 'availability': true"
                        )
                    },
                    {"role": "user", "content": self.filter_template.format(query=query)}
                ],
                temperature=getattr(settings, 'filter_parse_temperature', 0.0)
            )
            logger.debug(f"Filter parsing response: {response}")
            parsed = json.loads(response.strip("` \n"))
            logger.debug(f"Parsed filters: {parsed}")
            return parsed
        except (json.JSONDecodeError):
            logger.warning("Primary filter parsing failed - Using fallback")
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
        if "disponível" in query.lower() or "disponíveis" in query.lower() or "vagos" in query.lower():
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
        max_history = getattr(settings, 'max_history_length', 5)
        # In case want to limit token, use: token_limit = getattr(settings, 'response_token_limit', 5000)
        
        context_str = "\n".join([str(c.get("metadata", {})) for c in context])

        # Conditionally include room data
        rooms_str = ""
        if needs_room_info and rooms_data:
            rooms_str = "\nQuartos disponíveis:\n" + "\n".join(
                f"{r.get('room_id', '')}: {r.get('description', '')} (R${r.get('price', '')}/mês)"
                for r in rooms_data
            )

        # Limit history to prevent token overflow
        limited_history = history[-max_history:]
        
        messages = [
            {"role": "system", "content": system_message},
            *limited_history,
            {
                "role": "user",
                "content": f"Contexto:\n{context_str}\n\n{rooms_str}\n\Pergunta: {query}",
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
        """Graceful shutdown with individual service cleanup"""
        close_ops = []
        if self.openai:
            close_ops.append(self.openai.close())
        if self.pinecone:
            close_ops.append(self.pinecone.close())
        if self.mongo:
            close_ops.append(self.mongo.close())
        
        try:
            await asyncio.gather(*close_ops)
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
        finally:
            self._initialized = False
            logger.info("RAG services closed")