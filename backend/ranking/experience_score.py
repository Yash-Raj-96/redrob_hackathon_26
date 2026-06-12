"""
Experience scoring for candidate ranking
"""
from typing import Dict, Any, Tuple
import numpy as np
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class ExperienceScorer:
    """Calculate experience fit score"""
    
    def __init__(self):
        self.role_weights = {
            'entry': 0.6,
            'junior': 0.8,
            'mid': 1.0,
            'senior': 1.2,
            'lead': 1.3,
            'principal': 1.4,
            'architect': 1.5
        }
    
    async def calculate_score(self, years_experience: float, current_role: str, 
                             job_requirements: Dict[str, Any]) -> float:
        """Calculate experience fit score"""
        
        # Get target experience from job requirements
        target_years = job_requirements.get('required_experience', 3)
        target_role = job_requirements.get('target_role', '').lower()
        
        # Calculate experience score (Gaussian around target)
        exp_diff = abs(years_experience - target_years)
        exp_score = np.exp(-0.5 * (exp_diff / 2) ** 2)  # Gentle decay
        
        # Role relevance score
        role_score = await self._calculate_role_relevance(current_role, target_role)
        
        # Combine scores
        final_score = (0.7 * exp_score) + (0.3 * role_score)
        
        # Bonus for exceeding expectations (but not too much)
        if years_experience > target_years:
            excess = min(0.2, (years_experience - target_years) / target_years * 0.1)
            final_score = min(1.0, final_score + excess)
        
        return final_score
    
    async def _calculate_role_relevance(self, candidate_role: str, target_role: str) -> float:
        """Calculate role relevance score"""
        if not candidate_role or not target_role:
            return 0.5
        
        candidate_role = candidate_role.lower()
        target_role = target_role.lower()
        
        # Exact match
        if candidate_role == target_role:
            return 1.0
        
        # Partial match
        candidate_words = set(candidate_role.split())
        target_words = set(target_role.split())
        
        overlap = len(candidate_words.intersection(target_words))
        union = len(candidate_words.union(target_words))
        
        if union == 0:
            return 0.5
        
        jaccard = overlap / union
        
        # Check role levels
        candidate_level = self._extract_role_level(candidate_role)
        target_level = self._extract_role_level(target_role)
        
        if candidate_level and target_level:
            level_match = self._get_level_match_score(candidate_level, target_level)
            jaccard = (jaccard + level_match) / 2
        
        return jaccard
    
    def _extract_role_level(self, role: str) -> str:
        """Extract seniority level from role title"""
        role_lower = role.lower()
        
        for level in self.role_weights.keys():
            if level in role_lower:
                return level
        
        return 'mid'  # Default
    
    def _get_level_match_score(self, candidate_level: str, target_level: str) -> float:
        """Get match score between role levels"""
        levels = list(self.role_weights.keys())
        
        if candidate_level == target_level:
            return 1.0
        
        candidate_idx = levels.index(candidate_level)
        target_idx = levels.index(target_level)
        
        diff = abs(candidate_idx - target_idx)
        
        if diff == 1:
            return 0.7
        elif diff == 2:
            return 0.4
        else:
            return 0.2