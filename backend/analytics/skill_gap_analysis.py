
"""
Advanced Skill Gap Analyzer
"""

from collections import Counter
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class SkillGapAnalyzer:
    """
    Analyze organizational skill gaps and talent supply.
    """

    HIGH_GAP_THRESHOLD = 15
    MEDIUM_GAP_THRESHOLD = 30

    async def find_gaps(
        self,
        df: pd.DataFrame,
        target_skills: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify missing or underrepresented skills.
        """

        if df.empty:
            return []

        all_skills = self._extract_skills(df)
        skill_counts = Counter(all_skills)

        total_candidates = max(len(df), 1)

        gaps = []

        if target_skills:

            normalized_targets = [s.lower().strip() for s in target_skills]

            for skill in normalized_targets:

                count = skill_counts.get(skill, 0)
                percentage = round((count / total_candidates) * 100, 2)

                if percentage < self.MEDIUM_GAP_THRESHOLD:

                    severity = (
                        "High"
                        if percentage < self.HIGH_GAP_THRESHOLD
                        else "Medium"
                    )

                    gaps.append({
                        "skill": skill,
                        "candidates_with_skill": count,
                        "percentage": percentage,
                        "severity": severity,
                        "recommendation": self._recommendation(skill, severity)
                    })

        gaps.sort(key=lambda x: x["percentage"])

        return gaps

    async def get_skill_forecast(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect emerging skills.
        """

        if "last_active" not in df.columns:
            return {"trending_skills": []}

        try:
            recent_df = df[
                pd.to_datetime(df["last_active"], errors="coerce")
                > pd.Timestamp.now() - pd.Timedelta(days=90)
            ]

            recent_skills = self._extract_skills(recent_df)
            all_skills = self._extract_skills(df)

            recent_counts = Counter(recent_skills)
            overall_counts = Counter(all_skills)

            trending = []

            for skill, recent_count in recent_counts.items():

                total_count = overall_counts.get(skill, 0)

                if total_count == 0:
                    continue

                momentum = round((recent_count / total_count) * 100, 2)

                if momentum >= 30:
                    trending.append({
                        "skill": skill,
                        "momentum_score": momentum,
                        "trend": "Emerging"
                    })

            trending.sort(
                key=lambda x: x["momentum_score"],
                reverse=True
            )

            return {
                "trending_skills": trending[:10]
            }

        except Exception as e:
            logger.exception("Skill forecast failed")
            return {"trending_skills": []}

    async def generate_skill_report(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate skill intelligence report.
        """

        if df.empty:
            return {}

        all_skills = self._extract_skills(df)

        skill_counts = Counter(all_skills)

        total_candidates = len(df)

        avg_skills = round(
            len(all_skills) / total_candidates,
            2
        ) if total_candidates else 0

        diversity_score = round(
            len(skill_counts) / total_candidates,
            4
        ) if total_candidates else 0

        return {
            "total_unique_skills": len(skill_counts),
            "average_skills_per_candidate": avg_skills,
            "skill_diversity_score": diversity_score,
            "most_common_skills": [
                {
                    "skill": skill,
                    "count": count
                }
                for skill, count in skill_counts.most_common(20)
            ],
            "rare_skills": [
                {
                    "skill": skill,
                    "count": count
                }
                for skill, count in skill_counts.items()
                if count < 5
            ][:20]
        }

    def _extract_skills(
        self,
        df: pd.DataFrame
    ) -> List[str]:
        """
        Normalize and extract skills.
        """

        all_skills = []

        if "skills_normalized" not in df.columns:
            return all_skills

        for skills in df["skills_normalized"].dropna():

            if isinstance(skills, list):
                normalized = [
                    str(skill).lower().strip()
                    for skill in skills
                    if skill
                ]

                all_skills.extend(normalized)

        return all_skills

    def _recommendation(
        self,
        skill: str,
        severity: str
    ) -> str:

        if severity == "High":
            return (
                f"Urgent hiring/training initiative recommended for '{skill}'"
            )

        return (
            f"Consider improving sourcing and training for '{skill}'"
        )
