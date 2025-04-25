from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OpenAI Configuration
    openai_api_key: str
    embedding_model: str = "text-embedding-ada-002"
    chat_model: str = "gpt-3.5-turbo"
    openai_rpm_limit: int = 60  
    openai_tpm_limit: int = 90_000
    openai_max_concurrent_requests: int = 10
    openai_circuit_reset: int = 120

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

    # RAG Configuration
    rag_max_concurrent_requests: int = 50  # Simultaneous requests
    rag_circuit_timeout: int = 60  # Seconds
    rag_cache_maxsize: int = 1000
    rag_cache_ttl: int = 300
    rag_retries: int = 3
    rag_retry_backoff: int = 2
    rag_circuit_failure_threshold: int = 5   # max failures before open
    rag_circuit_reset_timeout: int = 120     # seconds before reset
    
    # Application Settings
    max_history_length: int = 5
    default_system_message: str = (
        """Você é um atendente prestativo da República dos Estudantes.
        Seu objetivo é construir uma conversa agradável com o cliente e enviar todas as informações necessárias.
        Seja simpático, educado e profissional."""
    )
    log_level: str = "DEBUG"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
