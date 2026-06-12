"""
Hybrid retrieval combining semantic and keyword search (SAFE VERSION)
"""
import os
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from backend.app.config import settings
from backend.embeddings.embedding_model import EmbeddingModel
from backend.embeddings.faiss_index import FAISSIndex
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridRetriever:
    """Hybrid search combining semantic and keyword-based retrieval"""

    def __init__(self, vector_weight: float = 0.6, top_k: int = 100):
        self.vector_weight = vector_weight
        self.top_k = top_k

        self.embedding_model: Optional[EmbeddingModel] = None
        self.vector_index: Optional[FAISSIndex] = None
        self.bm25_index: Optional[BM25Okapi] = None

        self.candidates_text: List[List[str]] = []
        self.candidate_ids: List[str] = []
        self.df: Optional[pd.DataFrame] = None

    # =========================================================
    # INIT
    # =========================================================
    async def initialize(self):
        logger.info("Initializing hybrid retriever")

        # -------------------------
        # Embedding model
        # -------------------------
        self.embedding_model = EmbeddingModel(
            model_name=settings.EMBEDDING_MODEL,
            device=getattr(settings, "DEVICE", "cpu"),
            batch_size=settings.BATCH_SIZE,
        )

        # -------------------------
        # FAISS index (SAFE LOAD)
        # -------------------------
        self.vector_index = FAISSIndex(dimension=settings.EMBEDDING_DIMENSION)

        try:
            self.vector_index.load()
            logger.info("FAISS index loaded successfully")
        except Exception as e:
            logger.warning(f"FAISS load failed → switching to BM25 only: {e}")
            self.vector_index = None

        # -------------------------
        # LOAD PARQUET (CRITICAL FIX)
        # -------------------------
        parquet_path = f"{settings.DATA_PROCESSED_PATH}/cleaned_candidates.parquet"

        try:
            if not os.path.exists(parquet_path):
                logger.error(f"Parquet file not found: {parquet_path}")
                self.df = pd.DataFrame()

            elif os.path.getsize(parquet_path) == 0:
                logger.error("Parquet file is EMPTY (0 bytes). Using empty dataset.")
                self.df = pd.DataFrame()

            else:
                self.df = pd.read_parquet(parquet_path)

        except Exception as e:
            logger.error(f"Failed to load parquet safely: {e}")
            self.df = pd.DataFrame()

        # -------------------------
        # Candidate IDs (SAFE)
        # -------------------------
        if self.df is None or self.df.empty:
            self.candidate_ids = []
            logger.warning("Empty dataset loaded - retriever will run in degraded mode")
        else:
            self.candidate_ids = self.df["candidate_id"].tolist()

        # -------------------------
        # BM25 index
        # -------------------------
        await self._build_bm25_index()

        logger.info("Hybrid retriever initialized (safe mode)")

    # =========================================================
    # BM25 INDEX (SAFE)
    # =========================================================
    async def _build_bm25_index(self):

        if self.df is None or self.df.empty:
            logger.warning("Skipping BM25 build (empty dataset)")
            self.candidates_text = []
            self.bm25_index = None
            return

        self.candidates_text = []

        for _, row in self.df.iterrows():
            text = self._candidate_to_search_text(row.to_dict())
            tokens = self._tokenize(text)
            self.candidates_text.append(tokens)

        self.bm25_index = BM25Okapi(self.candidates_text)

        logger.info(f"BM25 index built with {len(self.candidates_text)} docs")

    # =========================================================
    # SEARCH
    # =========================================================
    async def search(
        self,
        query: str,
        top_k: int = None,
        filters: Dict = None,
        use_hybrid: bool = True,
    ) -> List[Dict]:

        if self.df is None or self.df.empty:
            logger.warning("Search requested but dataset is empty")
            return []

        top_k = top_k or self.top_k

        semantic_results = await self._semantic_search(query, top_k * 2)

        if not use_hybrid or self.vector_index is None:
            return semantic_results[:top_k]

        keyword_results = await self._keyword_search(query, top_k * 2)

        combined = self._fusion_rrf(semantic_results, keyword_results)

        if filters:
            combined = await self._apply_filters(combined, filters)

        return combined[:top_k]

    # =========================================================
    # SEMANTIC SEARCH
    # =========================================================
    async def _semantic_search(self, query: str, k: int) -> List[Dict]:
        if self.embedding_model is None or self.vector_index is None:
            return []

        embedding = self.embedding_model.encode(query)
        return self.vector_index.search(embedding, k)

    # =========================================================
    # KEYWORD SEARCH
    # =========================================================
    async def _keyword_search(self, query: str, k: int) -> List[Dict]:
        if self.bm25_index is None or self.df is None or self.df.empty:
            return []

        tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(tokens)

        top_idx = np.argsort(scores)[-k:][::-1]

        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                results.append({
                    "id": self.candidate_ids[idx],
                    "score": float(scores[idx]),
                })

        return results

    # =========================================================
    # FUSION
    # =========================================================
    def _fusion_rrf(self, semantic, keyword, k: int = 60):

        scores = {}

        for rank, r in enumerate(semantic, 1):
            scores[r["id"]] = scores.get(r["id"], 0) + 1 / (k + rank)

        for rank, r in enumerate(keyword, 1):
            scores[r["id"]] = scores.get(r["id"], 0) + 1 / (k + rank)

        return sorted(
            [{"id": cid, "score": float(score)} for cid, score in scores.items()],
            key=lambda x: x["score"],
            reverse=True,
        )

    # =========================================================
    # FILTERS
    # =========================================================
    async def _apply_filters(self, results, filters):

        if self.df is None or self.df.empty:
            return results

        filtered = []

        for r in results:
            row = self.df[self.df["candidate_id"] == r["id"]]
            if row.empty:
                continue

            c = row.iloc[0]

            if "min_experience" in filters:
                if c.get("years_experience", 0) < filters["min_experience"]:
                    continue

            if "location" in filters:
                if filters["location"].lower() not in str(c.get("location", "")).lower():
                    continue

            filtered.append(r)

        return filtered

    # =========================================================
    # TEXT BUILDER
    # =========================================================
    def _candidate_to_search_text(self, c: Dict) -> str:
        parts = []

        if c.get("current_role"):
            parts.append(str(c["current_role"]))

        if isinstance(c.get("skills"), list):
            parts.extend(c["skills"])

        if c.get("education"):
            parts.append(str(c["education"]))

        if isinstance(c.get("previous_companies"), list):
            parts.extend(c["previous_companies"])

        return " ".join(parts)

    # =========================================================
    # TOKENIZER (FIXED BUG)
    # =========================================================
    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b[a-z0-9]+\b", text.lower())