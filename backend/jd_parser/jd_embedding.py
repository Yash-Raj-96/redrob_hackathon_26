"""
Generate embeddings for job descriptions
"""
import numpy as np
from backend.embeddings.embedding_model import EmbeddingModel
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class JDEmbedding:
    """Generate and manage JD embeddings"""
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
    
    async def generate_embedding(self, job_description: str) -> np.ndarray:
        """Generate embedding for job description"""
        # Enhance JD with structured representation
        enhanced_jd = await self._enhance_jd(job_description)
        embedding = self.embedding_model.encode(enhanced_jd)
        return embedding
    
    async def _enhance_jd(self, job_description: str) -> str:
        """Enhance JD with structured format for better embeddings"""
        # Add section markers
        enhanced = f"""
        JOB DESCRIPTION:
        {job_description}
        
        This role requires strong technical skills and relevant experience.
        """
        return enhanced
    
    async def match_jd_to_candidates(self, jd_embedding: np.ndarray, candidate_embeddings: np.ndarray, 
                                     top_k: int = 100) -> List[int]:
        """Match JD to candidates using embeddings"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity([jd_embedding], candidate_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return top_indices.tolist()