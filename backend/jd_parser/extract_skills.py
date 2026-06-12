"""
Extract skills from job description
"""
import re
from typing import Dict, List, Set
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class SkillExtractor:
    """Extract required and preferred skills from JD"""
    
    def __init__(self):
        self.skill_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'go', 'rust', 'ruby', 'php', 'swift', 'kotlin', 'typescript', 'scala', 'r', 'matlab'],
            'ml_ai': ['machine learning', 'deep learning', 'artificial intelligence', 'nlp', 'computer vision', 'reinforcement learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'transformers', 'langchain'],
            'data': ['sql', 'spark', 'hadoop', 'kafka', 'airflow', 'dbt', 'databricks', 'snowflake', 'bigquery', 'redshift'],
            'cloud': ['aws', 'azure', 'gcp', 'kubernetes', 'docker', 'terraform', 'jenkins', 'gitlab', 'github actions'],
            'web': ['react', 'vue', 'angular', 'node.js', 'express', 'django', 'flask', 'fastapi', 'spring boot']
        }
    
    async def extract(self, job_description: str) -> Dict[str, List[str]]:
        """Extract skills from JD"""
        jd_lower = job_description.lower()
        
        required_skills = set()
        preferred_skills = set()
        
        # Check for required vs preferred sections
        required_section = self._extract_section(jd_lower, 'required', 'preferred')
        preferred_section = self._extract_section(jd_lower, 'preferred', 'nice to have')
        
        # Extract from required section
        for category, skills in self.skill_keywords.items():
            for skill in skills:
                if skill in required_section:
                    required_skills.add(skill.title())
                elif skill in preferred_section:
                    preferred_skills.add(skill.title())
                elif skill in jd_lower:
                    # Default to required if in main JD
                    required_skills.add(skill.title())
        
        return {
            'required': list(required_skills),
            'preferred': list(preferred_skills)
        }
    
    def _extract_section(self, text: str, section_name: str, next_section: str) -> str:
        """Extract a specific section from JD"""
        patterns = [
            f'{section_name}.*?(?:{next_section}|$)',
            f'{section_name} skills.*?(?:{next_section}|$)',
            f'{section_name} qualifications.*?(?:{next_section}|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(0)
        
        return text