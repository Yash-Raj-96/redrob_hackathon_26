"""
Score candidates based on recruiter signals and activity
"""
from typing import Dict, Any
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class RecruiterSignalScorer:
    """Calculate score based on recruiter signals"""
    
    def __init__(self):
        self.signal_weights = {
            'github_activity': 0.25,
            'linkedin_endorsements': 0.20,
            'platform_activity': 0.20,
            'profile_completeness': 0.20,
            'certifications': 0.15
        }
    
    async def calculate_score(self, candidate: Dict[str, Any]) -> float:
        """Calculate recruiter signal score"""
        total_score = 0.0
        
        # GitHub activity
        github_score = await self._score_github_activity(candidate)
        total_score += github_score * self.signal_weights['github_activity']
        
        # LinkedIn endorsements
        linkedin_score = await self._score_linkedin_presence(candidate)
        total_score += linkedin_score * self.signal_weights['linkedin_endorsements']
        
        # Platform activity
        platform_score = await self._score_platform_activity(candidate)
        total_score += platform_score * self.signal_weights['platform_activity']
        
        # Profile completeness
        completeness_score = candidate.get('profile_completeness', 0.5)
        total_score += completeness_score * self.signal_weights['profile_completeness']
        
        # Certifications
        cert_score = await self._score_certifications(candidate)
        total_score += cert_score * self.signal_weights['certifications']
        
        return total_score
    
    async def _score_github_activity(self, candidate: Dict) -> float:
        """Score GitHub activity"""
        github_url = candidate.get('github_url', '')
        
        if not github_url:
            return 0.3  # No GitHub presence
        
        # This would ideally fetch actual GitHub data
        # For now, use heuristics
        score = 0.5
        
        # Check if profile has activity (based on metadata)
        if candidate.get('github_commits', 0) > 100:
            score = 0.9
        elif candidate.get('github_repos', 0) > 5:
            score = 0.7
        
        return score
    
    async def _score_linkedin_presence(self, candidate: Dict) -> float:
        """Score LinkedIn presence"""
        linkedin_url = candidate.get('linkedin_url', '')
        
        if not linkedin_url:
            return 0.2
        
        # Use profile completeness as proxy
        return candidate.get('profile_score', 0.5) / 100
    
    async def _score_platform_activity(self, candidate: Dict) -> float:
        """Score platform activity"""
        last_active = candidate.get('last_active', None)
        
        if not last_active:
            return 0.4
        
        # Simple recency scoring
        from datetime import datetime, timedelta
        
        if isinstance(last_active, str):
            try:
                last_active = datetime.fromisoformat(last_active)
            except:
                return 0.5
        
        days_since_active = (datetime.now() - last_active).days
        
        if days_since_active < 7:
            return 1.0
        elif days_since_active < 30:
            return 0.8
        elif days_since_active < 90:
            return 0.5
        else:
            return 0.3
    
    async def _score_certifications(self, candidate: Dict) -> float:
        """Score certifications"""
        certifications = candidate.get('certifications', [])
        
        if not certifications:
            return 0.2
        
        # Weight different certification types
        high_value_certs = ['AWS', 'Azure', 'Google Cloud', 'TensorFlow', 'PyTorch', 'Kubernetes']
        
        score = 0.5  # Base score
        high_value_count = sum(1 for cert in certifications if any(hv in cert for hv in high_value_certs))
        
        if high_value_count > 0:
            score = min(1.0, 0.5 + (high_value_count * 0.1))
        
        return score