# from app.services.mongo_service import MongoDBClient

# client = MongoDBClient()

# # Test 1: Get all available rooms
# filters = {"availability": True}
# print(client.get_all_rooms(filters))

# # Test 2: Get rooms with price < 100
# filters = {"price": {"$lt": 100}}
# print(client.get_all_rooms(filters))

# # Test 3: Use logical operators
# filters = {
#     "$and": [
#         {"room_type": "presidencial"},
#         {"price": {"$lt": 200}}
#     ]
# }
# print(client.get_all_rooms(filters))

import json
response = '{\n    "price": {\n        "$lt": number,\n        "$gt": number,\n        "$lte": number,\n        "$gte": number,\n        "$eq": number\n    },\n    "features": {\n        "$all": ["string"],\n        "$in": ["string"],\n        "$nin": ["string"]\n    },\n    "availability": boolean,\n    "$or": [{}],\n    "$and": [{}],\n    "$nor": [{}],\n    "$text": {\n        "$search": "string",\n        "$language": "string",\n        "$caseSensitive": boolean\n    },\n    "$geoWithin": {\n        "$geometry": {\n            "type": "Polygon",\n            "coordinates": [[[]]]\n        }\n    }\n}'
try:
    decision = json.loads(response.strip())
except json.JSONDecodeError:
    print("Failed to parse JSON")
