"""
Extract experience requirements from JD
"""
import re
from typing import Dict, Any
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class ExperienceExtractor:
    """Extract experience requirements"""
    
    async def extract(self, job_description: str) -> Dict[str, Any]:
        """Extract experience requirements"""
        jd_lower = job_description.lower()
        
        # Look for experience patterns
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*experience',
            r'experience\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'minimum\s*(\d+)\s*(?:years?|yrs?)',
            r'at least\s*(\d+)\s*(?:years?|yrs?)'
        ]
        
        min_years = 3  # default
        
        for pattern in patterns:
            match = re.search(pattern, jd_lower)
            if match:
                min_years = int(match.group(1))
                break
        
        # Extract target role level
        target_role = self._extract_role_level(jd_lower)
        
        return {
            'min_years': min_years,
            'max_years': min_years + 3,  # typical range
            'target_role': target_role
        }
    
    def _extract_role_level(self, text: str) -> str:
        """Extract target role level"""
        levels = ['entry', 'junior', 'mid', 'senior', 'lead', 'principal', 'architect']
        
        for level in levels:
            if level in text:
                return level.title() + " Level"
        
        return "Mid Level"