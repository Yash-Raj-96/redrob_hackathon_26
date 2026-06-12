"""
Pure semantic search using embeddings
"""
from typing import List, Dict, Optional

from backend.app.config import settings
from backend.embeddings.embedding_model import EmbeddingModel
from backend.embeddings.vector_store import VectorStore
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class SemanticSearch:
    """Semantic search using vector embeddings"""

    def __init__(self):
        self.embedding_model: Optional[EmbeddingModel] = None
        self.vector_store: Optional[VectorStore] = None

    async def initialize(self):
        """Initialize components safely (NO None model issues)"""

        logger.info("Initializing SemanticSearch...")

        # ✅ FIX: always pass model config explicitly
        self.embedding_model = EmbeddingModel(
            model_name=settings.EMBEDDING_MODEL,
            device=getattr(settings, "DEVICE", "cpu"),
        )

        # ✅ FIX: ensure dimension consistency
        self.vector_store = VectorStore(
            index_path=settings.VECTOR_DB_PATH,
            dimension=settings.EMBEDDING_DIMENSION,
        )

        try:
            await self.vector_store.load()
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.warning(f"Vector store load failed (will continue empty): {e}")

        logger.info("SemanticSearch initialized")

    async def search(
        self,
        query: str,
        top_k: int = 100,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Perform semantic search"""

        if not self.embedding_model or not self.vector_store:
            raise RuntimeError("SemanticSearch not initialized")

        # 🔥 Encode query safely
        query_embedding = self.embedding_model.encode([query])[0]

        # 🔥 Vector search
        results = await self.vector_store.search(query_embedding, top_k)

        # Optional filtering
        if filters:
            results = await self._apply_filters(results, filters)

        return results

    async def batch_search(
        self,
        queries: List[str],
        top_k: int = 100,
    ) -> List[List[Dict]]:
        """Batch semantic search (optimized encoding)"""

        if not self.embedding_model or not self.vector_store:
            raise RuntimeError("SemanticSearch not initialized")

        # 🔥 Faster: batch encode instead of loop
        embeddings = self.embedding_model.encode(queries)

        all_results = []
        for emb in embeddings:
            results = await self.vector_store.search(emb, top_k)
            all_results.append(results)

        return all_results

    async def _apply_filters(
        self,
        results: List[Dict],
        filters: Dict,
    ) -> List[Dict]:
        """Apply metadata filters"""

        filtered = []

        for result in results:
            include = True
            metadata = result.get("metadata", {})

            # Experience filter
            if "min_experience" in filters:
                if metadata.get("years_experience", 0) < filters["min_experience"]:
                    include = False

            if "max_experience" in filters:
                if metadata.get("years_experience", 0) > filters["max_experience"]:
                    include = False

            # Location filter
            if "location" in filters:
                if filters["location"].lower() not in str(metadata.get("location", "")).lower():
                    include = False

            # Salary filter
            if "max_salary" in filters:
                if metadata.get("salary_expectation", float("inf")) > filters["max_salary"]:
                    include = False

            if include:
                filtered.append(result)

        return filtered