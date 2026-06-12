"""
FastAPI dependencies for dependency injection (SAFE VERSION)
"""

from typing import Optional
from pathlib import Path

import pandas as pd

from backend.app.config import settings
from backend.app.constants import SCORING_WEIGHTS
from backend.embeddings.embedding_model import EmbeddingModel
from backend.embeddings.vector_store import VectorStore
from backend.ranking.final_ranker import FinalRanker
from backend.retrieval.hybrid_retrieval import HybridRetriever
from backend.utils.cache import CacheManager
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


# =========================================================
# Global singletons
# =========================================================

_embedding_model: Optional[EmbeddingModel] = None
_vector_store: Optional[VectorStore] = None
_retriever: Optional[HybridRetriever] = None
_ranker: Optional[FinalRanker] = None
_cache_manager: Optional[CacheManager] = None
_candidates_df: Optional[pd.DataFrame] = None


# =========================================================
# Embedding Model
# =========================================================

async def get_embedding_model() -> EmbeddingModel:
    global _embedding_model

    if _embedding_model is None:
        logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL}")

        _embedding_model = EmbeddingModel(
            model_name=settings.EMBEDDING_MODEL,
            device=getattr(settings, "DEVICE", "cpu"),
            batch_size=settings.BATCH_SIZE,
        )

    return _embedding_model


# =========================================================
# Vector Store (SAFE LOAD)
# =========================================================

async def get_vector_store() -> Optional[VectorStore]:
    global _vector_store

    if _vector_store is not None:
        return _vector_store

    logger.info("Initializing vector store")

    try:
        _vector_store = VectorStore(
            index_path=settings.VECTOR_DB_PATH,
            dimension=settings.EMBEDDING_DIMENSION,
        )

        await _vector_store.load()
        logger.info("Vector store loaded successfully")

    except Exception as e:
        logger.warning(f"Vector store unavailable (fallback mode): {e}")
        _vector_store = None

    return _vector_store


# =========================================================
# Retriever (CRITICAL FIX - NEVER CRASH APP)
# =========================================================

async def get_retriever() -> Optional[HybridRetriever]:
    global _retriever

    if _retriever is not None:
        return _retriever

    logger.info("Initializing hybrid retriever")

    try:
        _retriever = HybridRetriever(
            vector_weight=settings.HYBRID_SEARCH_WEIGHT,
            top_k=settings.TOP_K_RETRIEVAL,
        )

        await _retriever.initialize()
        logger.info("Hybrid retriever initialized successfully")

    except Exception as e:
        logger.error(f"Retriever failed but continuing in degraded mode: {e}")
        _retriever = None

    return _retriever


# =========================================================
# Ranker
# =========================================================

async def get_ranker() -> FinalRanker:
    global _ranker

    if _ranker is None:
        logger.info("Initializing final ranker")

        _ranker = FinalRanker(
            weights=SCORING_WEIGHTS,
            use_llm_reranking=settings.USE_LLM_RERANKING,
        )

    return _ranker


# =========================================================
# Cache Manager
# =========================================================

async def get_cache_manager() -> CacheManager:
    global _cache_manager

    if _cache_manager is None:
        logger.info("Initializing cache manager")

        _cache_manager = CacheManager(
            cache_type="redis" if settings.USE_REDIS_CACHE else "memory",
            ttl_seconds=settings.CACHE_TTL_SECONDS,
        )

    return _cache_manager


# =========================================================
# Candidates Dataset (SAFE + NEVER CRASH)
# =========================================================

async def get_candidates_data() -> pd.DataFrame:
    global _candidates_df

    if _candidates_df is not None:
        return _candidates_df

    logger.info("Loading candidates dataset (safe mode)")

    cache = await get_cache_manager()

    cached = cache.get("candidates_df")
    if cached is not None:
        logger.info("Loaded candidates from cache")
        _candidates_df = cached
        return _candidates_df

    processed_path = Path(settings.DATA_PROCESSED_PATH) / "cleaned_candidates.parquet"
    raw_json_path = Path(settings.DATA_RAW_PATH) / "candidates.jsonl"
    raw_csv_path = Path(settings.DATA_RAW_PATH) / "candidates.csv"

    df = pd.DataFrame()

    try:
        # parquet
        if processed_path.exists() and processed_path.stat().st_size > 0:
            logger.info(f"Loading parquet: {processed_path}")
            df = pd.read_parquet(processed_path)

        # jsonl fallback
        elif raw_json_path.exists():
            logger.info(f"Loading JSONL: {raw_json_path}")
            df = pd.read_json(raw_json_path, lines=True)

        # csv fallback
        elif raw_csv_path.exists():
            logger.info(f"Loading CSV: {raw_csv_path}")
            df = pd.read_csv(raw_csv_path)

        else:
            logger.warning("No dataset found - returning empty dataframe")
            df = pd.DataFrame()

    except Exception as e:
        logger.error(f"Dataset load failed: {e}")
        df = pd.DataFrame()

    # =====================================================
    # Normalization (SAFE)
    # =====================================================

    if not df.empty:

        if "candidate_id" not in df.columns:
            df["candidate_id"] = df.index.astype(str)

        if "skills_normalized" not in df.columns:
            df["skills_normalized"] = df.get("skills", "")

        if "years_experience" in df.columns:
            df["years_experience"] = pd.to_numeric(
                df["years_experience"],
                errors="coerce",
            ).fillna(0)

        df = df.fillna("")

    _candidates_df = df

    cache.set("candidates_df", df)

    logger.info(f"Candidates loaded: {len(df)} rows")

    return df


# =========================================================
# DB Placeholder
# =========================================================

async def get_db():
    return None