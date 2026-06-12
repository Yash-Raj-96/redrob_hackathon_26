"""
Extract other requirements from JD
"""
import re
from typing import Dict, List
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class RequirementExtractor:
    """Extract education, salary, and other requirements"""
    
    async def extract(self, job_description: str) -> Dict:
        """Extract various requirements"""
        jd_lower = job_description.lower()
        
        requirements = {
            'education_requirements': [],
            'salary_range': [0, float('inf')],
            'responsibilities': []
        }
        
        # Extract education requirements
        edu_patterns = [
            r"bachelor'?s\s+degree",
            r"master'?s\s+degree",
            r"ph\.?d",
            r"b\.?tech",
            r"m\.?tech",
            r"b\.?e",
            r"m\.?e",
            r"bca",
            r"mca"
        ]
        
        for pattern in edu_patterns:
            if re.search(pattern, jd_lower):
                requirements['education_requirements'].append(pattern)
        
        # Extract salary range (if present)
        salary_pattern = r'(\d+(?:\.\d+)?)\s*(?:lakh|L|LPA)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(?:lakh|L|LPA)'
        match = re.search(salary_pattern, jd_lower)
        if match:
            min_salary = float(match.group(1)) * 100000
            max_salary = float(match.group(2)) * 100000
            requirements['salary_range'] = [min_salary, max_salary]
        
        # Extract responsibilities (first few bullet points)
        resp_section = self._extract_responsibilities(jd_lower)
        requirements['responsibilities'] = resp_section[:5]
        
        return requirements
    
    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract responsibilities section"""
        # Look for responsibilities/role section
        patterns = [
            r'responsibilities?:?(.*?)(?:\n\n|\Z)',
            r'what you\'?ll do:?(.*?)(?:\n\n|\Z)',
            r'role:?(.*?)(?:\n\n|\Z)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                resp_text = match.group(1)
                # Extract bullet points
                bullets = re.findall(r'[•\-*]\s*(.+?)(?=\n[•\-*]|\n\n|\Z)', resp_text, re.DOTALL)
                if bullets:
                    return [b.strip() for b in bullets]
        
        return []