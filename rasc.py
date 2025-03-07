from app.services.mongo_service import MongoDBClient

client = MongoDBClient()

# Test 1: Get all available rooms
filters = {"availability": True}
print(client.get_all_rooms(filters))

# Test 2: Get rooms with price < 100
filters = {"price": {"$lt": 100}}
print(client.get_all_rooms(filters))

# Test 3: Use logical operators
filters = {
    "$and": [
        {"room_type": "presidencial"},
        {"price": {"$lt": 200}}
    ]
}
print(client.get_all_rooms(filters))