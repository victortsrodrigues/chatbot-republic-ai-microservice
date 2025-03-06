from pymongo import MongoClient
from app.config import settings
from typing import List, Dict
from app.utils.logger import logger

class MongoDBClient:
    def __init__(self):
        self.client = MongoClient(settings.mongo_uri)
        self.db = self.client[settings.mongo_db]
        self.rooms = self.db.rooms

    def get_rooms_by_ids(self, room_ids: List[str], filters: Dict) -> List[Dict]:
        query = {"room_id": {"$in": room_ids}}
        query.update(self._build_mongo_query(filters))
        return list(self.rooms.find(query, {'_id': 0}))

    def get_all_rooms(self, filters: Dict) -> List[Dict]:
        return list(self.rooms.find(self._build_mongo_query(filters), {'_id': 0}))

    def _build_mongo_query(self, filters: Dict) -> Dict:
        """Convert structured filters to MongoDB query with recursive processing"""       
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
            price_ops = {}
            for op in ['$lt', '$lte', '$gt', '$gte', '$eq']:
                if op in filters['price']:
                    converted_val = process_value(filters['price'][op])
                    if isinstance(converted_val, (int, float)):
                        price_ops[op] = converted_val
            if price_ops:
                mongo_query['price'] = price_ops

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