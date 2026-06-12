"""
Main job description parser orchestrator
"""
from typing import Dict, Any
from backend.jd_parser.extract_skills import SkillExtractor
from backend.jd_parser.extract_experience import ExperienceExtractor
from backend.jd_parser.extract_requirements import RequirementExtractor
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class JDParser:
    """Parse job description and extract structured requirements"""
    
    def __init__(self):
        self.skill_extractor = SkillExtractor()
        self.exp_extractor = ExperienceExtractor()
        self.requirement_extractor = RequirementExtractor()
    
    async def parse(self, job_description: str) -> Dict[str, Any]:
        """Parse JD and extract all requirements"""
        logger.info("Parsing job description")
        
        requirements = {
            'required_skills': [],
            'preferred_skills': [],
            'required_experience': 3,  # default
            'target_role': '',
            'salary_range': [0, float('inf')],
            'education_requirements': [],
            'responsibilities': []
        }
        
        # Extract skills
        skills = await self.skill_extractor.extract(job_description)
        requirements['required_skills'] = skills.get('required', [])
        requirements['preferred_skills'] = skills.get('preferred', [])
        
        # Extract experience
        exp = await self.exp_extractor.extract(job_description)
        requirements['required_experience'] = exp.get('min_years', 3)
        requirements['target_role'] = exp.get('target_role', '')
        
        # Extract other requirements
        other_reqs = await self.requirement_extractor.extract(job_description)
        requirements.update(other_reqs)
        
        logger.info(f"Extracted {len(requirements['required_skills'])} required skills")
        return requirements