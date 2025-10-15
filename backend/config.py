"""
Configuration management for the PyTorch RAG Assistant.
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Local LLM Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    huggingface_model_name: str = Field(default="microsoft/DialoGPT-medium", env="HUGGINGFACE_MODEL_NAME")
    use_local_model: bool = Field(default=True, env="USE_LOCAL_MODEL")
    
    # Neo4j Database
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")
    
    # ChromaDB Configuration
    chroma_persist_directory: str = Field(default="./data/chroma_db", env="CHROMA_PERSIST_DIRECTORY")
    
    # Redis Configuration (Optional)
    redis_url: Optional[str] = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_debug: bool = Field(default=False, env="API_DEBUG")
    
    # CORS Configuration
    allowed_origins: List[str] = Field(default=["chrome-extension://*"], env="ALLOWED_ORIGINS")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Data Directories
    pytorch_docs_dir: str = Field(default="./data/pytorch_docs", env="PYTORCH_DOCS_DIR")
    cache_dir: str = Field(default="./data/cache", env="CACHE_DIR")
    logs_dir: str = Field(default="./logs", env="LOGS_DIR")
    
    # Scraping Configuration
    scraping_delay: float = Field(default=1.0, env="SCRAPING_DELAY")
    max_concurrent_requests: int = Field(default=5, env="MAX_CONCURRENT_REQUESTS")
    user_agent: str = Field(default="PyTorch-RAG-Assistant/1.0", env="USER_AGENT")
    
    # RAG Configuration
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    max_retrieval_docs: int = Field(default=10, env="MAX_RETRIEVAL_DOCS")
    relevance_threshold: float = Field(default=0.5, env="RELEVANCE_THRESHOLD")
    
    # Evaluation
    ragas_dataset_path: str = Field(default="./data/evaluation/test_dataset.json", env="RAGAS_DATASET_PATH")
    evaluation_cache_dir: str = Field(default="./data/evaluation_cache", env="EVALUATION_CACHE_DIR")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.pytorch_docs_dir,
            self.cache_dir,
            self.logs_dir,
            self.chroma_persist_directory,
            os.path.dirname(self.ragas_dataset_path),
            self.evaluation_cache_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
