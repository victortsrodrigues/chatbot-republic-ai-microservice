from pymongo import MongoClient
from app.config import settings
from typing import List, Dict

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
        """Convert structured filters to MongoDB query"""
        mongo_query = {}
        
        if 'price' in filters:
            price = {}
            if 'min' in filters['price']:
                price['$gte'] = filters['price']['min']
            if 'max' in filters['price']:
                price['$lte'] = filters['price']['max']
            if price:
                mongo_query['price'] = price
                
        if 'features' in filters:
            mongo_query['features'] = {'$all': filters['features']}
            
        if 'room_type' in filters:
            mongo_query['type'] = filters['room_type']
            
        if 'size' in filters:
            size = {}
            if 'min' in filters['size']:
                size['$gte'] = filters['size']['min']
            if 'max' in filters['size']:
                size['$lte'] = filters['size']['max']
            if size:
                mongo_query['size'] = size
                
        return mongo_query