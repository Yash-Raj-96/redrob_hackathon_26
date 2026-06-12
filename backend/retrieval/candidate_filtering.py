"""
Advanced candidate filtering capabilities
"""
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class CandidateFilter:
    """Filter candidates based on various criteria"""

    def __init__(self, candidates_df: pd.DataFrame):
        self.df = candidates_df
        self.filters_applied: List[str] = []

    async def apply_filters(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply multiple filters to candidate dataframe"""

        filtered_df = self.df.copy()
        self.filters_applied = []  # ✅ reset per run

        # =========================================================
        # Experience
        # =========================================================
        if "min_experience" in filters:
            filtered_df = filtered_df[
                filtered_df["years_experience"] >= filters["min_experience"]
            ]
            self.filters_applied.append(f"min_experience >= {filters['min_experience']}")

        if "max_experience" in filters:
            filtered_df = filtered_df[
                filtered_df["years_experience"] <= filters["max_experience"]
            ]
            self.filters_applied.append(f"max_experience <= {filters['max_experience']}")

        # =========================================================
        # Location
        # =========================================================
        if "location" in filters and "location" in filtered_df.columns:
            loc = str(filters["location"]).lower()
            filtered_df = filtered_df[
                filtered_df["location"].fillna("").str.lower().str.contains(loc)
            ]
            self.filters_applied.append(f"location contains '{loc}'")

        if "locations" in filters and "location" in filtered_df.columns:
            locations = [l.lower() for l in filters["locations"]]
            filtered_df = filtered_df[
                filtered_df["location"].fillna("").str.lower().isin(locations)
            ]
            self.filters_applied.append(f"location in {locations}")

        # =========================================================
        # Skills (safe handling)
        # =========================================================
        def safe_skills(x):
            if isinstance(x, list):
                return x
            return []

        filtered_df["skills_normalized"] = filtered_df["skills_normalized"].apply(safe_skills)

        if "required_skills" in filters:
            required = set(filters["required_skills"])

            def match_required(skills):
                if not skills:
                    return False
                return len(required.intersection(set(skills))) >= max(1, int(len(required) * 0.7))

            filtered_df = filtered_df[filtered_df["skills_normalized"].apply(match_required)]
            self.filters_applied.append(f"required_skills match >=70%")

        if "any_skills" in filters:
            any_skills = set(filters["any_skills"])

            def match_any(skills):
                return bool(set(skills).intersection(any_skills))

            filtered_df = filtered_df[filtered_df["skills_normalized"].apply(match_any)]
            self.filters_applied.append(f"any_skills match")

        # =========================================================
        # Salary
        # =========================================================
        if "max_salary" in filters:
            filtered_df = filtered_df[
                filtered_df["salary_expectation"] <= filters["max_salary"]
            ]
            self.filters_applied.append(f"salary <= {filters['max_salary']}")

        if "min_salary" in filters:
            filtered_df = filtered_df[
                filtered_df["salary_expectation"] >= filters["min_salary"]
            ]
            self.filters_applied.append(f"salary >= {filters['min_salary']}")

        # =========================================================
        # Notice period
        # =========================================================
        if "max_notice_days" in filters and "notice_period_days" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["notice_period_days"] <= filters["max_notice_days"]
            ]
            self.filters_applied.append(
                f"notice <= {filters['max_notice_days']} days"
            )

        # =========================================================
        # Education
        # =========================================================
        if "education_level" in filters and "education" in filtered_df.columns:
            levels = filters["education_level"]
            if isinstance(levels, str):
                levels = [levels]

            pattern = "|".join(levels)
            filtered_df = filtered_df[
                filtered_df["education"].fillna("").str.contains(pattern, case=False)
            ]
            self.filters_applied.append(f"education in {levels}")

        # =========================================================
        # Profile score
        # =========================================================
        if "min_profile_score" in filters and "profile_completeness" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["profile_completeness"] >= filters["min_profile_score"]
            ]
            self.filters_applied.append(
                f"profile_score >= {filters['min_profile_score']}"
            )

        # =========================================================
        # Last active
        # =========================================================
        if "last_active_days" in filters and "last_active" in filtered_df.columns:
            cutoff = datetime.now() - pd.Timedelta(days=filters["last_active_days"])

            filtered_df = filtered_df[
                pd.to_datetime(filtered_df["last_active"], errors="coerce") >= cutoff
            ]
            self.filters_applied.append(
                f"active within {filters['last_active_days']} days"
            )

        logger.info(
            f"Filters applied={len(self.filters_applied)} | "
            f"remaining={len(filtered_df)} / {len(self.df)}"
        )

        return filtered_df

    async def get_filter_summary(self, filtered_df: pd.DataFrame) -> Dict[str, Any]:
        """Return correct filter summary"""

        return {
            "filters_applied": self.filters_applied,
            "original_count": len(self.df),
            "filtered_count": len(filtered_df),
            "reduction_pct": round(
                (1 - len(filtered_df) / max(1, len(self.df))) * 100, 2
            ),
        }

    async def suggest_filters(self, job_description: str) -> List[Dict]:
        """Suggest filters based on job description"""

        return [
            {
                "filter_type": "min_experience",
                "suggested_value": 3,
                "reason": "Most roles require baseline experience",
            },
            {
                "filter_type": "max_notice_days",
                "suggested_value": 30,
                "reason": "Faster joiners preferred in hiring cycles",
            },
        ]