"""
Availability and notice period scoring
"""
from typing import Dict, Any
from backend.utils.logger import setup_logger
from backend.app.constants import NOTICE_PERIOD_SCORES

logger = setup_logger(__name__)

class AvailabilityScorer:
    """Calculate availability score"""
    
    def calculate_score(self, notice_period_days: int) -> float:
        """Calculate score based on notice period"""
        if notice_period_days <= 0:
            return 1.0  # Immediate joiner
        elif notice_period_days <= 15:
            return 0.9
        elif notice_period_days <= 30:
            return 0.7
        elif notice_period_days <= 45:
            return 0.5
        elif notice_period_days <= 60:
            return 0.3
        else:
            return 0.1
        
    def get_availability_text(self, notice_period_days: int) -> str:
        """Get human-readable availability description"""
        if notice_period_days <= 0:
            return "Immediate joining"
        elif notice_period_days <= 15:
            return f"Available in {notice_period_days} days"
        elif notice_period_days <= 30:
            return "Available in 1 month"
        elif notice_period_days <= 60:
            return "Available in 2 months"
        else:
            return "Long notice period (>2 months)"