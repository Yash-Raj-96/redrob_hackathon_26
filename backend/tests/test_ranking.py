"""
Tests for ranking module
"""
import pytest
import pandas as pd
from backend.ranking.skill_matcher import SkillMatcher
from backend.ranking.experience_score import ExperienceScorer

@pytest.mark.asyncio
async def test_skill_matching():
    """Test skill matching logic"""
    matcher = SkillMatcher()
    
    candidate_skills = ['Python', 'SQL', 'Machine Learning']
    required_skills = ['Python', 'Java', 'Machine Learning']
    preferred_skills = ['SQL', 'AWS']
    
    score, matched, missing = await matcher.calculate_score(
        candidate_skills, required_skills, preferred_skills
    )
    
    assert 0 <= score <= 1
    assert 'Python' in matched
    assert 'Java' in missing

@pytest.mark.asyncio
async def test_experience_scoring():
    """Test experience scoring"""
    scorer = ExperienceScorer()
    
    score = await scorer.calculate_score(
        years_experience=5,
        current_role="Senior Developer",
        job_requirements={'required_experience': 4, 'target_role': 'Senior'}
    )
    
    assert 0 <= score <= 1
    assert score > 0.7  # Should be high for close match

def test_availability_scoring():
    """Test availability scoring"""
    from backend.ranking.availability_score import AvailabilityScorer
    
    scorer = AvailabilityScorer()
    
    assert scorer.calculate_score(0) == 1.0  # Immediate
    assert scorer.calculate_score(15) == 0.9
    assert scorer.calculate_score(90) == 0.1