"""
FAISS vector index for efficient similarity search
"""
import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
from backend.utils.logger import setup_logger
from backend.app.config import settings

logger = setup_logger(__name__)

class FAISSIndex:
    """FAISS index wrapper for candidate embeddings"""
    
    def __init__(self, dimension: int, index_path: str = None):
        self.dimension = dimension
        self.index_path = index_path or settings.VECTOR_DB_PATH
        self.index = None
        self.candidate_ids = []
        
    def build_index(self, embeddings: np.ndarray, candidate_ids: List[str]):
        """Build FAISS index from embeddings"""
        logger.info(f"Building FAISS index with {len(embeddings)} vectors")
        
        # Normalize embeddings for inner product (cosine similarity)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Create index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine
        self.index.add(embeddings)
        
        self.candidate_ids = candidate_ids
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
        
        # Save index
        self.save()
        
    def search(self, query_embedding: np.ndarray, k: int = 100) -> Tuple[List[str], List[float]]:
        """Search for similar candidates"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Get candidate IDs and scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.candidate_ids):
                results.append({
                    'id': self.candidate_ids[idx],
                    'score': float(score)
                })
        
        return results
    
    def save(self):
        """Save index and metadata to disk"""
        os.makedirs(self.index_path, exist_ok=True)
        
        index_file = os.path.join(self.index_path, "faiss.index")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        
        faiss.write_index(self.index, index_file)
        
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'candidate_ids': self.candidate_ids,
                'dimension': self.dimension
            }, f)
        
        logger.info(f"Index saved to {index_file}")
    
    def load(self):
        """Load index from disk"""
        index_file = os.path.join(self.index_path, "faiss.index")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        
        if not os.path.exists(index_file):
            logger.warning(f"Index file not found: {index_file}")
            return False
        
        self.index = faiss.read_index(index_file)
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
            self.candidate_ids = metadata['candidate_ids']
            self.dimension = metadata['dimension']
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
        return True
    
    def get_size(self) -> int:
        """Get number of vectors in index"""
        return self.index.ntotal if self.index else 0