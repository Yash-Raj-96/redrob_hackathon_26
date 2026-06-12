"""
Tests for embeddings module
"""
import pytest
import numpy as np
from backend.embeddings.embedding_model import EmbeddingModel

@pytest.mark.asyncio
async def test_embedding_generation():
    """Test embedding generation"""
    model = EmbeddingModel()
    
    texts = ["Python developer", "Machine learning engineer"]
    embeddings = model.encode(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0

def test_embedding_normalization():
    """Test embedding normalization"""
    model = EmbeddingModel()
    
    embedding = model.encode("test text")
    
    # Check if normalized (L2 norm should be ~1)
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.01