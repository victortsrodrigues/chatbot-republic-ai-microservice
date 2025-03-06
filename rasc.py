context = [
  {
    "id": "room-123",
    "values": [...],
    "metadata": {
      "type": "room",
      "room_id": "123",
      "text": "Luxury suite with ocean view, 80m², king bed",
      "features": ["suite", "ocean-view", "king-bed"],
      "room_type": "suite",
      "size": 80,
      "price": 450
    }
  },
  {
    "id": "room-456",
    "values": [...],
    "metadata": {
      "type": "room",
      "room_id": "456",
      "text": "Standard suite with city view, 50m², queen bed",
      "features": ["suite", "city-view"],
      "room_type": "suite",
      "size": 50,
      "price": 300
    }
  }
]

print(context[0]["metadata"].get("type"))
print(context[0]["metadata"]["type"])
print(context[0]["metadata"]["room_id"])

room_ids = list(
            {
                c["metadata"]["room_id"]
                for c in context
                if c["metadata"].get("type") == "room"
            }
        )

print(room_ids)