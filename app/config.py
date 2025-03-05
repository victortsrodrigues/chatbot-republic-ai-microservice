from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str
    embedding_model: str = "text-embedding-ada-002"
    chat_model: str = "gpt-3.5-turbo"
    
    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_env: str
    pinecone_index_name: str
    pinecone_cloud: str  # e.g., "aws" or "gcp"
    pinecone_region: str  # e.g., "us-west1"
    
    # AWS Configuration
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "us-east-1"
    
    # S3 Configuration
    s3_bucket_name: str
    
    # Application Settings
    max_history_length: int = 5
    default_system_message: str = "You are a helpful assistant for Student Republic. Be polite and professional."
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()