"""
Generate and manage embeddings for all candidates (OPTIMIZED CPU VERSION)
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from backend.embeddings.embedding_model import EmbeddingModel
from backend.embeddings.faiss_index import FAISSIndex
from backend.ingestion.chunk_profiles import ProfileChunker
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingGenerator:
    """High-performance embedding pipeline (CPU optimized)"""

    def __init__(self):
        self.model = EmbeddingModel()
        self.chunker = ProfileChunker()
        self.index = FAISSIndex(dimension=self.model.embedding_dim)

    # =====================================================
    # FAST BATCH ITERATOR
    # =====================================================
    def _batch_iter(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    # =====================================================
    # MAIN PIPELINE (OPTIMIZED)
    # =====================================================
    async def generate_all_embeddings(self, df, force_rebuild=False):

        logger.info(f"Generating embeddings for {len(df)} candidates...")

        chunks_df = await self.chunker.chunk_profiles(df)

        texts = []
        weights = []
        candidate_ids = []

        # =====================================================
        # FAST FLATTENING (NO PANDAS OPERATIONS INSIDE LOOP)
        # =====================================================
        for row in chunks_df.itertuples(index=False):
            cid = row.candidate_id

            for chunk in row.chunks:
                texts.append(chunk["text"])
                weights.append(chunk["weight"])
                candidate_ids.append(cid)

        logger.info(f"Total chunks: {len(texts)}")

        # =====================================================
        # BATCH EMBEDDING (FAST CPU)
        # =====================================================
        BATCH = 1024
        all_embeddings = []

        for i, batch in enumerate(self._batch_iter(texts, BATCH)):
            emb = self.model.encode(batch)
            all_embeddings.append(emb)

            if i % 10 == 0:
                logger.info(f"Processed {i * BATCH}/{len(texts)} chunks")

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        # =====================================================
        # FAST GROUPING (NO PANDAS MASKING IN LOOP)
        # =====================================================
        df_meta = pd.DataFrame({
            "cid": np.array(candidate_ids),
            "w": np.array(weights, dtype=np.float32)
        })

        unique_cids = df_meta["cid"].unique()

        final_embeddings = np.zeros((len(unique_cids), embeddings.shape[1]), dtype=np.float32)
        candidate_order = []

        start = 0
        idx = 0

        for cid in unique_cids:

            mask = (df_meta["cid"].values == cid)
            n = int(mask.sum())

            emb_slice = embeddings[start:start + n]
            w = df_meta["w"].values[mask].reshape(-1, 1)

            # weighted mean (stable)
            final_embeddings[idx] = (emb_slice * w).sum(axis=0) / (w.sum() + 1e-9)

            candidate_order.append(cid)

            start += n
            idx += 1

        # =====================================================
        # SAVE OUTPUT
        # =====================================================
        np.save("data/processed/embeddings.npy", final_embeddings)

        self.index.build_index(final_embeddings, candidate_order)

        logger.info(f"Index built successfully: {len(final_embeddings)} vectors")

        return final_embeddings


# =========================================================
# INDEX BUILDER
# =========================================================
async def build_index(data_path: str):

    from backend.ingestion.preprocess_pipeline import PreprocessingPipeline

    logger.info("Loading dataset...")

    pipeline = PreprocessingPipeline()
    df = await pipeline.run()

    generator = EmbeddingGenerator()

    embeddings = await generator.generate_all_embeddings(df, force_rebuild=True)

    logger.info(f"Index ready: {len(embeddings)} vectors")

    return generator.index


if __name__ == "__main__":
    import asyncio
    asyncio.run(build_index("data/processed/cleaned_candidates.parquet"))