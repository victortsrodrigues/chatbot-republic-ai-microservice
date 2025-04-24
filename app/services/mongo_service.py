from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
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
                    serverSelectionTimeoutMS=5000
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

        def process_value(value):
            """Recursive type conversion and validation"""
            if isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            if isinstance(value, list):
                return [process_value(item) for item in value]
            try:
                return float(value) if isinstance(value, (int, str)) and str(value).isdigit() else value
            except:
                return value

        # Process price operators
        if 'price' in filters:
            if isinstance(filters['price'], dict):  # Operators like $lt, $gt
                price_ops = {}
                for op in ['$lt', '$lte', '$gt', '$gte', '$eq']:
                    if op in filters['price']:
                        converted_val = process_value(filters['price'][op])
                        if isinstance(converted_val, (int, float)):
                            price_ops[op] = converted_val
                if price_ops:
                    mongo_query['price'] = price_ops
            elif isinstance(filters['price'], (int, float)):  # Direct price match
                mongo_query['price'] = filters['price']

        # Process features with array operators
        if 'features' in filters:
            features_ops = {}
            for op in ['$all', '$in', '$nin']:
                if op in filters['features'] and isinstance(filters['features'][op], list):
                    features_ops[op] = [
                        str(item) for item in filters['features'][op]
                    ]
            if features_ops:
                mongo_query['features'] = features_ops

        # Handle logical operators recursively
        for logical_op in ['$and', '$or', '$nor']:
            if logical_op in filters:
                processed_clauses = []
                for clause in filters[logical_op]:
                    processed = self._build_mongo_query(clause)
                    if processed:
                        processed_clauses.append(processed)
                if processed_clauses:
                    mongo_query[logical_op] = processed_clauses

        # Direct field mappings
        for field in ['room_type', 'availability']:
            if field in filters:
                mongo_query[field] = process_value(filters[field])

        # Special operators
        if '$text' in filters:
            text_search = {}
            if '$search' in filters['$text']:
                text_search['$search'] = str(filters['$text']['$search'])
                for opt in ['$language', '$caseSensitive']:
                    if opt in filters['$text']:
                        text_search[opt] = process_value(filters['$text'][opt])
            if text_search:
                mongo_query['$text'] = text_search

        if '$geoWithin' in filters and '$geometry' in filters['$geoWithin']:
            mongo_query['location'] = {
                '$geoWithin': process_value(filters['$geoWithin'])
            }

        return mongo_query
    
    async def close(self) -> None:
        """Close the MongoDB client and free resources"""
        # CHANGED: gracefully close motor client
        self._client.close()