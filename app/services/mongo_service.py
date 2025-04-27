from pymongo.errors import PyMongoError
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from typing import List, Dict
from app.utils.logger import logger

class MongoDBClient:
    _instance = None
    _client = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            try:
                # Async connection pool setup
                cls._client = AsyncIOMotorClient(
                    settings.mongo_uri,
                    maxPoolSize=settings.mongo_max_pool_size if hasattr(settings, 'mongo_max_pool_size') else 100,
                    minPoolSize=settings.mongo_min_pool_size if hasattr(settings, 'mongo_min_pool_size') else 10,
                    connectTimeoutMS=settings.mongo_connect_timeout_ms,
                    socketTimeoutMS=settings.mongo_socket_timeout_ms,
                    serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
                    tls=settings.mongo_tls_enabled
                )
                # Initialize database reference
                cls._instance.db = cls._client.get_database(settings.mongo_db)
                logger.info("MongoDB connection pool initialized")
            except PyMongoError as e:
                logger.critical(f"MongoDB connection failed: {str(e)}")
                raise
        return cls._instance
    
    async def get_all_rooms(self, filters: Dict) -> List[Dict]:
        """Async room query using Motor client"""
        try:
            query = self._build_mongo_query(filters)
            cursor = self.db.rooms.find(query, {'_id': 0})
            return await cursor.to_list(length=1000)
        except Exception as e:
            logger.error(f"Mongo query failed: {str(e)}")
            return []

    def _build_mongo_query(self, filters: Dict) -> Dict:
        """Convert structured filters to MongoDB query with recursive processing"""
        if not isinstance(filters, dict):
            logger.warning(f"Invalid filters type: {type(filters)}")
            return {}

        mongo_query = {}
        mongo_query.update(self._process_price(filters))
        mongo_query.update(self._process_features(filters))
        mongo_query.update(self._process_logical_operators(filters))
        mongo_query.update(self._process_direct_fields(filters))
        mongo_query.update(self._process_special_operators(filters))

        return mongo_query

    def _process_price(self, filters: Dict) -> Dict:
        """Process price-related filters."""
        price_query = {}
        if 'price' in filters:
            if isinstance(filters['price'], dict):
                price_query['price'] = {
                    op: self._process_value(filters['price'][op])
                    for op in ['$lt', '$lte', '$gt', '$gte', '$eq']
                    if op in filters['price'] and isinstance(self._process_value(filters['price'][op]), (int, float))
                }
            elif isinstance(filters['price'], (int, float)):
                price_query['price'] = filters['price']
        return price_query

    def _process_features(self, filters: Dict) -> Dict:
        """Process feature-related filters."""
        features_query = {}
        if 'features' in filters:
            features_query['features'] = {
                op: [str(item) for item in filters['features'][op]]
                for op in ['$all', '$in', '$nin']
                if op in filters['features'] and isinstance(filters['features'][op], list)
            }
        return features_query

    def _process_logical_operators(self, filters: Dict) -> Dict:
        """Process logical operators like $and, $or, $nor."""
        logical_query = {}
        for logical_op in ['$and', '$or', '$nor']:
            if logical_op in filters:
                clauses = [
                    self._build_mongo_query(clause)
                    for clause in filters[logical_op]
                    if self._build_mongo_query(clause)
                ]
                if clauses:
                    logical_query[logical_op] = clauses
        return logical_query

    def _process_direct_fields(self, filters: Dict) -> Dict:
        """Process direct field mappings."""
        direct_query = {}
        for field in ['availability']:
            if field in filters:
                direct_query[field] = self._process_value(filters[field])
        return direct_query

    def _process_special_operators(self, filters: Dict) -> Dict:
        """Process special operators like $text and $geoWithin."""
        special_query = {}
        if '$text' in filters:
            text_search = {
                opt: self._process_value(filters['$text'][opt])
                for opt in ['$search', '$language', '$caseSensitive']
                if opt in filters['$text']
            }
            if text_search:
                special_query['$text'] = text_search

        if '$geoWithin' in filters and '$geometry' in filters['$geoWithin']:
            special_query['location'] = {
                '$geoWithin': self._process_value(filters['$geoWithin'])
            }
        return special_query

    def _process_value(self, value):
        """Recursive type conversion and validation."""
        if isinstance(value, dict):
            return {k: self._process_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._process_value(item) for item in value]
        try:
            return float(value) if isinstance(value, (int, str)) and str(value).isdigit() else value
        except:
            return value
    
    async def close(self) -> None:
        """Close the MongoDB client and free resources"""
        # CHANGED: gracefully close motor client
        self._client.close()