import pytest
from unittest.mock import AsyncMock, patch, MagicMock, call
from pymongo.errors import PyMongoError
from motor.motor_asyncio import AsyncIOMotorClient

from app.services.mongo_service import MongoDBClient


@pytest.mark.unit
def test_mongo_singleton_pattern():
    """Test that MongoDBClient follows singleton pattern."""
    # Create two instances
    instance1 = MongoDBClient()
    instance2 = MongoDBClient()
    
    # Verify they are the same object
    assert instance1 is instance2


@pytest.mark.unit
def test_mongo_initialization():
    """Test MongoDB client initialization."""
    with patch('app.services.mongo_service.AsyncIOMotorClient') as mock_client:
        # Set up mock
        mock_instance = mock_client.return_value
        mock_instance.get_database.return_value = "mock_db"
        
        # Force new instance by clearing class variable
        MongoDBClient._instance = None
        
        # Create new instance
        client = MongoDBClient()
        
        # Verify client was initialized correctly
        mock_client.assert_called_once()
        assert client.db == "mock_db"


@pytest.mark.unit
def test_mongo_initialization_error():
    """Test MongoDB client initialization error handling."""
    with patch('app.services.mongo_service.AsyncIOMotorClient', side_effect=PyMongoError("Connection failed")), \
         patch('app.services.mongo_service.logger') as mock_logger:
        
        # Force new instance by clearing class variable
        MongoDBClient._instance = None
        
        # Attempt to create instance and expect exception
        with pytest.raises(PyMongoError) as excinfo:
            MongoDBClient()
        
        # Verify error was logged
        assert "Connection failed" in str(excinfo.value)
        mock_logger.critical.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_all_rooms_success():
    """Test successful retrieval of all rooms."""
    # Sample data
    sample_rooms = [
        {"room_id": "A101", "description": "Suite com varanda", "price": 750},
        {"room_id": "B202", "description": "Quarto duplo com banheiro", "price": 650}
    ]
    
    # Create a client with mocked query
    client = MongoDBClient()
    
    # Mock the database and cursor
    mock_cursor = AsyncMock()
    mock_cursor.to_list.return_value = sample_rooms
    
    client.db = MagicMock()
    client.db.rooms.find.return_value = mock_cursor
    
    # Mock the query builder
    client._build_mongo_query = MagicMock(return_value={"availability": True})
    
    # Call the method
    result = await client.get_all_rooms({"availability": True})
    
    # Verify the results
    assert result == sample_rooms
    client.db.rooms.find.assert_called_once_with({"availability": True}, {'_id': 0})
    mock_cursor.to_list.assert_called_once_with(length=1000)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_all_rooms_exception():
    """Test error handling during room retrieval."""
    # Create a client with mocked query that raises an exception
    client = MongoDBClient()
    
    # Mock the database to raise an exception
    client.db = MagicMock()
    client.db.rooms.find.side_effect = Exception("Database error")
    
    # Mock the query builder and logger
    client._build_mongo_query = MagicMock(return_value={"availability": True})
    
    with patch('app.services.mongo_service.logger') as mock_logger:
        # Call the method
        result = await client.get_all_rooms({"availability": True})
        
        # Verify error was logged and empty result returned
        assert result == []
        mock_logger.error.assert_called_once()


@pytest.mark.unit
def test_build_mongo_query_invalid_filters():
    """Test _build_mongo_query with invalid filter types."""
    client = MongoDBClient()
    
    with patch('app.services.mongo_service.logger') as mock_logger:
        # Test with None
        result = client._build_mongo_query(None)
        assert result == {}
        
        # Test with string
        result = client._build_mongo_query("not a dict")
        assert result == {}
        
        # Verify warnings were logged
        assert mock_logger.warning.call_count == 2


@pytest.mark.unit
def test_build_mongo_query_price_filters():
    """Test _build_mongo_query with price filters."""
    client = MongoDBClient()
    
    # Test with price comparison operators
    filters = {
        "price": {
            "$lt": 800,
            "$gt": 500,
            "$eq": 750
        }
    }
    
    result = client._build_mongo_query(filters)
    
    # Verify query contains price operators
    assert result["price"]["$lt"] == 800
    assert result["price"]["$gt"] == 500
    assert result["price"]["$eq"] == 750
    
    # Test with direct price value
    filters = {"price": 750}
    result = client._build_mongo_query(filters)
    assert result["price"] == 750


@pytest.mark.unit
def test_build_mongo_query_features_filters():
    """Test _build_mongo_query with features filters."""
    client = MongoDBClient()
    
    # Test with features operators
    filters = {
        "features": {
            "$all": ["suite", "varanda"],
            "$in": ["ar-condicionado", "frigobar"],
            "$nin": ["barulhento"]
        }
    }
    
    result = client._build_mongo_query(filters)

    # Verify query contains features operators
    assert "suite" in result["features"]["$all"]
    assert "varanda" in result["features"]["$all"]
    assert "ar-condicionado" in result["features"]["$in"]
    assert "frigobar" in result["features"]["$in"]
    assert "barulhento" in result["features"]["$nin"]


@pytest.mark.unit
def test_build_mongo_query_logical_operators():
    """Test _build_mongo_query with logical operators."""
    client = MongoDBClient()
        
    # Original method for final call
    original_method = client._build_mongo_query
    
    # Test with logical operators
    filters = {
        "$or": [
            {"price": {"$lt": 800}},
            {"features": {"$all": ["varanda"]}}
        ],
        "$and": [
            {"availability": True}
        ]
    }
    
    # Call the real method with our filters
    result = original_method(filters)
    
    # Verify logical operators structure
    assert "$or" in result
    assert "$and" in result
    assert len(result["$or"]) == 2
    assert len(result["$and"]) == 1


@pytest.mark.unit
def test_build_mongo_query_direct_fields():
    """Test _build_mongo_query with direct field mappings."""
    client = MongoDBClient()
    
    # Test with availability field
    filters = {"availability": True}
    
    result = client._build_mongo_query(filters)
    
    # Verify direct field mapping
    assert result["availability"] == True


@pytest.mark.unit
def test_build_mongo_query_special_operators():
    """Test _build_mongo_query with special operators."""
    client = MongoDBClient()
    
    # Test with text search
    filters = {
        "$text": {
            "$search": "varanda",
            "$language": "portuguese",
            "$caseSensitive": False
        }
    }
    
    result = client._build_mongo_query(filters)
    
    # Verify text search operator
    assert result["$text"]["$search"] == "varanda"
    assert result["$text"]["$language"] == "portuguese"
    assert result["$text"]["$caseSensitive"] == False
    
    # Test with geo query
    filters = {
        "$geoWithin": {
            "$geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]
            }
        }
    }
    
    result = client._build_mongo_query(filters)
    
    # Verify geo query
    assert "location" in result
    assert "$geoWithin" in result["location"]
    assert "$geometry" in result["location"]["$geoWithin"]


@pytest.mark.unit
def test_process_value():
    """Test _process_value with various input types."""
    client = MongoDBClient()
    
    # Test with different types
    from math import isclose
    assert isclose(client._process_value(100), 100.0, rel_tol=1e-9)
    assert isclose(client._process_value("100"), 100.0, rel_tol=1e-9)
    assert client._process_value("string") == "string"
    assert client._process_value(True) == True
    
    # Test with dictionary
    result = client._process_value({"key1": "100", "key2": "string"})
    assert isclose(result["key1"], 100.0, rel_tol=1e-9)
    assert result["key2"] == "string"
    
    # Test with list
    result = client._process_value(["100", "string", True])
    assert isclose(result[0], 100.0, rel_tol=1e-9)
    assert result[1] == "string"
    assert result[2] == True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close():
    """Test close method properly closes the MongoDB client."""
    # Mock the client's close method
    mock_client = MagicMock()
    
    # Create client and set mocked _client
    client = MongoDBClient()
    client._client = mock_client
    
    # Call close method
    await client.close()
    
    # Verify close was called
    mock_client.close.assert_called_once()