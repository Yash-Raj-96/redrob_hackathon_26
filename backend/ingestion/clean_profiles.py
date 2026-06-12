"""
Clean and standardize candidate profiles (SAFE + PRODUCTION READY)
"""
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from backend.utils.logger import setup_logger
from backend.utils.text_cleaning import clean_text, normalize_text

logger = setup_logger(__name__)


class ProfileCleaner:
    """Clean candidate profile data (robust version)"""

    def __init__(self):
        self.skill_synonyms = {
            "python": ["python3", "py", "python programming"],
            "javascript": ["js", "javascript es6", "node.js"],
            "machine learning": ["ml", "deep learning", "ai"],
            "data science": ["data analytics", "data mining"]
        }

    # =========================================================
    # SAFE TEXT HANDLER (🔥 FIXED CORE)
    # =========================================================
    def _safe_text(self, x) -> str:
        """Convert ANY input into safe clean string"""

        if x is None:
            return ""

        # handle NaN safely
        if isinstance(x, float) and pd.isna(x):
            return ""

        # handle numpy arrays / lists / tuples
        if isinstance(x, (list, tuple, np.ndarray)):
            return " ".join([str(i) for i in x if i is not None])

        # handle dict / weird objects
        if isinstance(x, dict):
            return " ".join([str(v) for v in x.values()])

        # fallback
        return str(x)

    # =========================================================
    # MAIN PIPELINE
    # =========================================================
    async def clean_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting profile cleaning for {len(df)} candidates")

        # Clean text fields safely
        text_fields = ['name', 'current_role', 'location', 'education', 'summary']

        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].apply(
                    lambda x: clean_text(self._safe_text(x))
                )

        # Standardize skills
        if 'skills' in df.columns:
            df['skills'] = df['skills'].apply(self._standardize_skills)

        # Standardize experience
        if 'years_experience' in df.columns:
            df['years_experience'] = df['years_experience'].apply(
                self._parse_experience
            )

        # Standardize notice period
        if 'notice_period' in df.columns:
            df['notice_period_days'] = df['notice_period'].apply(
                self._parse_notice_period
            )

        # Handle missing values
        df = self._handle_missing_values(df)

        logger.info("Profile cleaning completed successfully")

        return df

    # =========================================================
    # SKILL NORMALIZATION
    # =========================================================
    def _standardize_skills(self, skills) -> List[str]:

        if skills is None:
            return []

        if isinstance(skills, float) and pd.isna(skills):
            return []

        if isinstance(skills, str):
            skills = [s.strip().lower() for s in skills.split(',')]

        if not isinstance(skills, list):
            return []

        standardized = []

        for skill in skills:
            skill_lower = str(skill).lower().strip()

            found = False
            for standard, variants in self.skill_synonyms.items():
                if skill_lower in variants or skill_lower == standard:
                    standardized.append(standard)
                    found = True
                    break

            if not found:
                standardized.append(skill_lower)

        return list(set(standardized))

    # =========================================================
    # EXPERIENCE PARSING
    # =========================================================
    def _parse_experience(self, exp) -> float:

        if exp is None:
            return 0.0

        if isinstance(exp, float) and pd.isna(exp):
            return 0.0

        if isinstance(exp, (int, float)):
            return float(exp)

        numbers = re.findall(r'\d+(?:\.\d+)?', str(exp))
        return float(numbers[0]) if numbers else 0.0

    # =========================================================
    # NOTICE PERIOD PARSING
    # =========================================================
    def _parse_notice_period(self, notice) -> int:

        if notice is None:
            return 30

        if isinstance(notice, float) and pd.isna(notice):
            return 30

        notice_str = str(notice).lower()

        if 'immediate' in notice_str:
            return 0
        if '15' in notice_str:
            return 15
        if '30' in notice_str or 'month' in notice_str:
            return 30
        if '45' in notice_str:
            return 45
        if '60' in notice_str or '2 month' in notice_str:
            return 60
        if '90' in notice_str or '3 month' in notice_str:
            return 90

        return 30

    # =========================================================
    # MISSING VALUE HANDLING
    # =========================================================
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if col in ['candidate_id', 'name']:
                continue
            df[col] = df[col].fillna('Unknown')

        return df