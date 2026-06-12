"""
Application Configuration Management
"""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    # =========================================================
    # Application Settings
    # =========================================================

    APP_NAME: str = "INDIA RUNS Candidate Engine"
    APP_VERSION: str = "1.0.0"

    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    LOG_LEVEL: str = "INFO"

    API_PREFIX: str = "/api/v1"

    # =========================================================
    # Security
    # =========================================================

    SECRET_KEY: str = "change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # =========================================================
    # CORS (FIXED: supports env string parsing)
    # =========================================================

    ALLOWED_ORIGINS: List[str] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # fallback if env provides string instead of list
        if isinstance(self.ALLOWED_ORIGINS, str):
            self.ALLOWED_ORIGINS = [
                origin.strip()
                for origin in self.ALLOWED_ORIGINS.split(",")
                if origin.strip()
            ]

        if not self.ALLOWED_ORIGINS:
            self.ALLOWED_ORIGINS = [
                "http://localhost:8501",
                "http://localhost:3000",
                "http://127.0.0.1:8501",
                "http://127.0.0.1:3000",
            ]

    # =========================================================
    # Data Paths
    # =========================================================

    DATA_RAW_PATH: str = "data/raw"
    DATA_PROCESSED_PATH: str = "data/processed"

    VECTOR_DB_PATH: str = "vector_db"
    FAISS_INDEX_PATH: str = "vector_db/faiss.index"

    # =========================================================
    # Embedding / ML Models
    # =========================================================

    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION: int = 1024

    # backward compatibility (FIX for your crash)
    EMBEDDING_MODEL_NAME: Optional[str] = None

    BATCH_SIZE: int = 32
    DEVICE: str = "cpu"

    # =========================================================
    # Retrieval Settings
    # =========================================================

    HYBRID_SEARCH_WEIGHT: float = 0.6
    TOP_K_RETRIEVAL: int = 100
    TOP_K_RERANK: int = 50
    SIMILARITY_THRESHOLD: float = 0.65

    # =========================================================
    # LLM Settings
    # =========================================================

    DEFAULT_LLM_PROVIDER: str = "openai"

    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-latest"

    USE_LLM_RERANKING: bool = False
    ENABLE_EXPLANATIONS: bool = True

    # =========================================================
    # Cache Settings
    # =========================================================

    USE_REDIS_CACHE: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600

    # =========================================================
    # Monitoring
    # =========================================================

    ENABLE_METRICS: bool = True
    ENABLE_HEALTHCHECKS: bool = True

    # =========================================================
    # Performance
    # =========================================================

    NUM_WORKERS: int = 4
    MAX_CONCURRENT_REQUESTS: int = 100
    REQUEST_TIMEOUT_SECONDS: int = 120

    # =========================================================
    # Ranking Defaults
    # =========================================================

    DEFAULT_SKILL_WEIGHT: float = 0.40
    DEFAULT_EXPERIENCE_WEIGHT: float = 0.25
    DEFAULT_RECRUITER_SIGNAL_WEIGHT: float = 0.15
    DEFAULT_AVAILABILITY_WEIGHT: float = 0.10
    DEFAULT_SALARY_WEIGHT: float = 0.10

    # =========================================================
    # Explainability
    # =========================================================

    ENABLE_BIAS_DETECTION: bool = True
    ENABLE_SKILL_GAP_ANALYSIS: bool = True
    ENABLE_MARKET_INSIGHTS: bool = True


# Singleton instance
settings = Settings()