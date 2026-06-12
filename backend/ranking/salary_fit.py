"""
Salary expectation scoring
"""
from typing import Dict, Any, List
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class SalaryFitScorer:
    """Calculate salary fit score"""
    
    async def calculate_score(self, expected_salary: float, salary_range: List[float]) -> float:
        """Calculate how well candidate's salary fits within budget"""
        if not salary_range or len(salary_range) < 2:
            return 0.5  # Neutral if no range provided
        
        min_salary, max_salary = salary_range[0], salary_range[1]
        
        if expected_salary <= 0:
            return 0.3
        
        # Perfect fit if within range
        if min_salary <= expected_salary <= max_salary:
            # Higher score for middle of range
            mid_point = (min_salary + max_salary) / 2
            proximity = 1 - abs(expected_salary - mid_point) / (max_salary - min_salary)
            return 0.8 + (0.2 * proximity)
        
        # Below minimum (great for company)
        elif expected_salary < min_salary:
            # Score decreases as we go lower (candidate might be underqualified)
            diff_ratio = (min_salary - expected_salary) / min_salary
            return max(0.5, 0.9 - diff_ratio)
        
        # Above maximum
        else:
            # Penalty increases with how much above max
            excess_ratio = (expected_salary - max_salary) / max_salary
            
            if excess_ratio <= 0.2:
                return 0.4
            elif excess_ratio <= 0.5:
                return 0.2
            else:
                return 0.0  # Too expensive