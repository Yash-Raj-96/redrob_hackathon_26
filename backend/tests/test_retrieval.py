"""
Tests for retrieval module
"""
import pytest
import asyncio
from backend.retrieval.hybrid_retrieval import HybridRetriever

@pytest.mark.asyncio
async def test_hybrid_search():
    """Test hybrid search functionality"""
    retriever = HybridRetriever()
    await retriever.initialize()
    
    results = await retriever.search("Python developer with ML experience", top_k=10)
    
    assert len(results) > 0
    assert 'id' in results[0]
    assert 'score' in results[0]

@pytest.mark.asyncio
async def test_semantic_search():
    """Test semantic search"""
    from backend.retrieval.semantic_search import SemanticSearch
    
    searcher = SemanticSearch()
    await searcher.initialize()
    
    results = await searcher.search("data scientist", top_k=5)
    
    assert len(results) <= 5

@pytest.mark.asyncio
async def test_keyword_search():
    """Test keyword search"""
    from backend.retrieval.keyword_search import KeywordSearch
    
    searcher = KeywordSearch()
    await searcher.initialize()
    
    results = await searcher.search("Python", top_k=10)
    
    assert len(results) > 0