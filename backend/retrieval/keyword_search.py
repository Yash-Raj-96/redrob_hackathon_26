"""
Keyword-based search using BM25
"""
import numpy as np
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import pandas as pd
from backend.app.config import settings
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class KeywordSearch:
    """Keyword search using BM25 algorithm"""
    
    def __init__(self):
        self.bm25_index = None
        self.candidate_ids = []
        self.corpus = []
    
    async def initialize(self):
        """Initialize BM25 index"""
        logger.info("Initializing keyword search index...")
        
        # Load processed candidates
        df = pd.read_parquet(f"{settings.DATA_PROCESSED_PATH}/cleaned_candidates.parquet")
        
        self.candidate_ids = df['candidate_id'].tolist()
        
        # Create corpus
        self.corpus = []
        for _, row in df.iterrows():
            doc = self._create_document(row)
            tokenized_doc = self._tokenize(doc)
            self.corpus.append(tokenized_doc)
        
        # Build BM25 index
        self.bm25_index = BM25Okapi(self.corpus)
        logger.info(f"BM25 index initialized with {len(self.corpus)} documents")
    
    async def search(self, query: str, top_k: int = 100) -> List[Dict]:
        """Perform keyword search"""
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    'id': self.candidate_ids[idx],
                    'score': float(scores[idx]),
                    'rank': len(results) + 1
                })
        
        return results
    
    async def search_with_highlights(self, query: str, top_k: int = 100) -> List[Dict]:
        """Search with highlighted matching terms"""
        results = await self.search(query, top_k)
        
        # Add highlighting info
        query_terms = set(self._tokenize(query))
        
        for result in results:
            idx = self.candidate_ids.index(result['id'])
            doc_tokens = self.corpus[idx]
            
            # Find matching terms
            matching_terms = [term for term in query_terms if term in doc_tokens]
            result['matching_terms'] = matching_terms
            result['match_count'] = len(matching_terms)
        
        return results
    
    def _create_document(self, row: pd.Series) -> str:
        """Create searchable document from candidate"""
        parts = []
        
        # Role and experience
        if pd.notna(row.get('current_role')):
            parts.append(row['current_role'])
        
        # Skills
        skills = row.get('skills_normalized', row.get('skills', []))
        if skills and len(skills) > 0:
            parts.extend(skills)
        
        # Education
        if pd.notna(row.get('education')):
            parts.append(row['education'])
        
        # Companies
        companies = row.get('previous_companies', [])
        if companies:
            parts.extend(companies[:3])
        
        return ' '.join(parts).lower()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        import re
        # Simple tokenization
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        return tokens