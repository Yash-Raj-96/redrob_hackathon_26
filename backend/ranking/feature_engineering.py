"""
Feature engineering for candidate ranking
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    """Engineer features for candidate ranking"""
    
    def __init__(self):
        self.feature_names = []
    
    async def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for ranking"""
        logger.info("Engineering features...")
        
        df = df.copy()
        
        # Experience features
        if 'years_experience' in df.columns:
            df['exp_seniority'] = pd.cut(
                df['years_experience'],
                bins=[0, 2, 5, 8, 12, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
        
        # Skill diversity
        if 'skills_normalized' in df.columns:
            df['skill_count'] = df['skills_normalized'].apply(len)
            df['skill_diversity'] = df['skills_categorized'].apply(lambda x: len(x))
        
        # Profile completeness score
        completeness_features = [
            'name', 'email', 'phone', 'current_role', 
            'years_experience', 'skills', 'education'
        ]
        
        existing_features = [f for f in completeness_features if f in df.columns]
        df['profile_completeness'] = df[existing_features].notna().sum(axis=1) / len(existing_features)
        
        # Recruiter interaction features
        if 'profile_score' in df.columns:
            df['recruiter_interest'] = df['profile_score'] / 100
        
        # Location tier (based on city)
        if 'location' in df.columns:
            tier1_cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata']
            df['is_tier1_city'] = df['location'].apply(lambda x: 1 if x in tier1_cities else 0)
        
        # Salary features
        if 'salary_expectation' in df.columns:
            df['salary_log'] = np.log1p(df['salary_expectation'])
        
        # Interaction features
        if 'skill_count' in df.columns and 'years_experience' in df.columns:
            df['skill_per_year'] = df['skill_count'] / (df['years_experience'] + 1)
        
        # Company prestige score (if company names available)
        if 'previous_companies' in df.columns:
            prestige_companies = ['Google', 'Microsoft', 'Amazon', 'Facebook', 'Apple', 'Netflix']
            
            def get_prestige_score(companies):
                if not companies:
                    return 0
                return sum(1 for company in companies if company in prestige_companies) / len(companies)
            
            df['company_prestige'] = df['previous_companies'].apply(get_prestige_score)
        
        self.feature_names = [col for col in df.columns if col not in self.feature_names]
        
        logger.info(f"Created {len(self.feature_names)} features")
        return df
    
    async def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to 0-1 scale"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['candidate_id', 'rank']:
                min_val = df[col].min()
                max_val = df[col].max()
                
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0
        
        return df
    
    async def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (for explainability)"""
        # This would typically come from a trained model
        importance = {
            'skill_match': 0.35,
            'years_experience': 0.25,
            'profile_completeness': 0.10,
            'skill_count': 0.10,
            'company_prestige': 0.08,
            'is_tier1_city': 0.07,
            'recruiter_interest': 0.05
        }
        
        return importance