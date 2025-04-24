from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str
    embedding_model: str = "text-embedding-ada-002"
    chat_model: str = "gpt-3.5-turbo"

    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_host: str
    pinecone_index_name: str
    pinecone_cloud: str  # e.g., "aws" or "gcp"
    pinecone_region: str  # e.g., "us-west1"
    pinecone_max_retries: int = 10
    pinecone_dimension: int = 1536  # Default for text-embedding-ada-002
    pinecone_circuit_timeout: int = 60
    pinecone_health_interval: int = 300

    # AWS Configuration
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str

    # S3 Configuration
    s3_bucket_name: str

    # MongoDB Configuration
    mongo_uri: str
    mongo_db: str
    mongo_max_pool_size: int = 100
    mongo_min_pool_size: int = 10

    # Application Settings
    max_history_length: int = 5
    default_system_message: str = (
        """Você é um atendente prestativo da República dos Estudantes.
        Seu objetivo é construir uma conversa agradável com o cliente e enviar todas as informações necessárias.
        Seja simpático, educado e profissional."""
    )
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
