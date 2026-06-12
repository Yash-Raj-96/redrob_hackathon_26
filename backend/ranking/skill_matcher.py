"""
Skill matching logic between job and candidates
"""
from typing import List, Tuple, Dict, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class SkillMatcher:
    """Match candidate skills with job requirements"""
    
    def __init__(self):
        self.skill_embeddings = None  # For semantic skill matching
        self.skill_synonyms = {
            'python': ['python3', 'py', 'python programming'],
            'javascript': ['js', 'javascript es6', 'node.js'],
            'java': ['java8', 'java11', 'j2ee'],
            'sql': ['mysql', 'postgresql', 'database query'],
            'aws': ['amazon web services', 'ec2', 's3'],
            'azure': ['microsoft azure', 'azure devops'],
            'docker': ['containerization', 'docker containers', 'docker compose'],
            'kubernetes': ['k8s', 'kubernetes clusters', 'kubectl']
        }
    
    async def calculate_score(self, candidate_skills: List[str], 
                             required_skills: List[str], 
                             preferred_skills: List[str] = None) -> Tuple[float, List[str], List[str]]:
        """Calculate skill match score"""
        if not candidate_skills:
            return 0.0, [], required_skills or []
        
        candidate_set = set(self._normalize_skill_list(candidate_skills))
        required_set = set(self._normalize_skill_list(required_skills)) if required_skills else set()
        preferred_set = set(self._normalize_skill_list(preferred_skills)) if preferred_skills else set()
        
        # Find matched skills
        matched_required = candidate_set.intersection(required_set)
        matched_preferred = candidate_set.intersection(preferred_set)
        matched_skills = list(matched_required.union(matched_preferred))
        
        # Find missing skills
        missing_required = required_set - candidate_set
        missing_skills = list(missing_required)
        
        # Calculate score
        required_score = len(matched_required) / len(required_set) if required_set else 1.0
        preferred_score = len(matched_preferred) / len(preferred_set) if preferred_set else 1.0
        
        # Weighted score (required skills matter more)
        final_score = (0.7 * required_score) + (0.3 * preferred_score)
        
        # Bonus for semantic skill matches
        semantic_bonus = await self._check_semantic_matches(candidate_set, required_set)
        final_score = min(1.0, final_score + semantic_bonus)
        
        return final_score, matched_skills, missing_skills
    
    async def _check_semantic_matches(self, candidate_skills: Set[str], required_skills: Set[str]) -> float:
        """Check for semantic matches using synonyms"""
        bonus = 0.0
        
        for required in required_skills:
            if required in self.skill_synonyms:
                synonyms = set(self.skill_synonyms[required])
                if candidate_skills.intersection(synonyms):
                    bonus += 0.05  # Small bonus for synonym match
        
        return min(0.2, bonus)  # Max 20% bonus
    
    def _normalize_skill_list(self, skills: List[str]) -> List[str]:
        """Normalize skills to standard form"""
        normalized = []
        
        for skill in skills:
            if not skill:
                continue
            
            skill_lower = skill.lower().strip()
            
            # Check for synonyms
            found = False
            for standard, variants in self.skill_synonyms.items():
                if skill_lower == standard or skill_lower in variants:
                    normalized.append(standard)
                    found = True
                    break
            
            if not found:
                normalized.append(skill_lower)
        
        return normalized
    
    async def get_skill_gaps(self, candidate_skills: List[str], job_skills: List[str]) -> Dict:
        """Get detailed skill gap analysis"""
        candidate_set = set(self._normalize_skill_list(candidate_skills))
        job_set = set(self._normalize_skill_list(job_skills))
        
        return {
            'present_skills': list(candidate_set.intersection(job_set)),
            'missing_skills': list(job_set - candidate_set),
            'extra_skills': list(candidate_set - job_set),
            'match_percentage': len(candidate_set.intersection(job_set)) / len(job_set) if job_set else 100
        }