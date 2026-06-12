"""
Vector store abstraction for managing embeddings
"""
import numpy as np
from typing import List, Dict, Optional
import pickle
import os
from backend.embeddings.faiss_index import FAISSIndex
from backend.app.config import settings
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class VectorStore:
    """High-level vector store interface"""
    
    def __init__(self, index_path: str = None, dimension: int = None):
        self.index_path = index_path or settings.VECTOR_DB_PATH
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.index = FAISSIndex(dimension=self.dimension, index_path=self.index_path)
        self.metadata = {}
        
    async def load(self):
        """Load vector store from disk"""
        success = self.index.load()
        if success:
            await self._load_metadata()
            logger.info("Vector store loaded successfully")
        else:
            logger.warning("No existing vector store found")
        
        return success
    
    async def _load_metadata(self):
        """Load metadata"""
        metadata_path = os.path.join(self.index_path, "store_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
    
    async def save_metadata(self):
        """Save metadata"""
        os.makedirs(self.index_path, exist_ok=True)
        metadata_path = os.path.join(self.index_path, "store_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    async def add_vectors(self, embeddings: np.ndarray, ids: List[str], metadata: List[Dict] = None):
        """Add vectors to store"""
        self.index.build_index(embeddings, ids)
        
        if metadata:
            for i, meta in enumerate(metadata):
                self.metadata[ids[i]] = meta
        
        await self.save_metadata()
    
    async def search(self, query_embedding: np.ndarray, k: int = 100) -> List[Dict]:
        """Search for similar vectors"""
        results = self.index.search(query_embedding, k)
        
        # Add metadata to results
        for result in results:
            candidate_id = result['id']
            if candidate_id in self.metadata:
                result['metadata'] = self.metadata[candidate_id]
        
        return results
    
    def get_size(self) -> int:
        """Get number of vectors"""
        return self.index.get_size()
    
    async def delete(self):
        """Delete vector store"""
        import shutil
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
            logger.info("Vector store deleted")