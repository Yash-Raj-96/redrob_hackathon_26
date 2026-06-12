"""
Generate hiring insights and recommendations
"""

from typing import List, Dict, Any
from collections import Counter

import numpy as np
import pandas as pd

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class HiringInsights:
    """
    Generate actionable hiring insights from candidate pool data.
    """

    # ==========================================
    # Public APIs
    # ==========================================

    async def generate_recommendations(
        self,
        df: pd.DataFrame,
        bias_metrics: Dict[str, Any],
        skill_gaps: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate intelligent hiring recommendations.
        """

        recommendations: List[str] = []

        if df.empty:
            return ["No candidate data available for recommendations"]

        try:
            recommendations.extend(
                self._generate_skill_recommendations(df, skill_gaps)
            )

            recommendations.extend(
                self._generate_location_recommendations(df)
            )

            recommendations.extend(
                self._generate_experience_recommendations(df)
            )

            recommendations.extend(
                self._generate_diversity_recommendations(bias_metrics)
            )

            recommendations.extend(
                self._generate_market_recommendations(df)
            )

            # Deduplicate
            recommendations = list(dict.fromkeys(recommendations))

            return recommendations[:10]

        except Exception as e:
            logger.exception("Failed generating hiring recommendations")
            return [f"Failed generating insights: {str(e)}"]

    async def get_market_insights(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate market-level hiring insights.
        """

        if df.empty:
            return self._empty_market_response()

        years_exp = pd.to_numeric(
            df.get("years_experience"),
            errors="coerce"
        )

        salary_data = pd.to_numeric(
            df.get("salary_expectation"),
            errors="coerce"
        )

        insights = {
            "candidate_pool_size": int(len(df)),

            "average_experience": round(
                float(years_exp.mean(skipna=True)),
                2
            ),

            "median_experience": round(
                float(years_exp.median(skipna=True)),
                2
            ),

            "median_salary": round(
                float(salary_data.median(skipna=True)),
                2
            ) if salary_data.notna().any() else None,

            "top_skills": await self._get_top_skills(df, 10),

            "top_locations": self._get_top_locations(df),

            "experience_distribution": self._get_experience_distribution(df),

            "supply_demand_score": self._calculate_supply_demand(df),

            "hiring_difficulty": self._assess_hiring_difficulty(df),

            "candidate_quality_index": self._calculate_quality_index(df),

            "remote_readiness_score": self._calculate_remote_readiness(df),

            "market_competitiveness": self._assess_market_competitiveness(df)
        }

        return insights

    # ==========================================
    # Recommendation Generators
    # ==========================================

    def _generate_skill_recommendations(
        self,
        df: pd.DataFrame,
        skill_gaps: List[Dict[str, Any]]
    ) -> List[str]:

        recommendations = []

        if not skill_gaps:
            return recommendations

        top_gap = skill_gaps[0]

        skill_name = top_gap.get("skill", "Unknown")
        gap_count = top_gap.get("count", len(df))

        recommendations.append(
            f"Prioritize upskilling initiatives for '{skill_name}' "
            f"({gap_count} candidates currently missing this competency)"
        )

        if len(skill_gaps) >= 3:
            critical_skills = ", ".join(
                gap.get("skill", "")
                for gap in skill_gaps[:3]
            )

            recommendations.append(
                f"High-demand missing skills detected: {critical_skills}"
            )

        return recommendations

    def _generate_location_recommendations(
        self,
        df: pd.DataFrame
    ) -> List[str]:

        recommendations = []

        if "location" not in df.columns:
            return recommendations

        location_counts = (
            df["location"]
            .dropna()
            .value_counts()
        )

        if location_counts.empty:
            return recommendations

        top_location = location_counts.index[0]
        top_count = int(location_counts.iloc[0])

        recommendations.append(
            f"Strong talent concentration in {top_location} "
            f"({top_count} candidates available)"
        )

        if len(location_counts) < 3:
            recommendations.append(
                "Candidate sourcing appears geographically concentrated — "
                "consider expanding outreach to additional regions"
            )

        return recommendations

    def _generate_experience_recommendations(
        self,
        df: pd.DataFrame
    ) -> List[str]:

        recommendations = []

        if "years_experience" not in df.columns:
            return recommendations

        experience = pd.to_numeric(
            df["years_experience"],
            errors="coerce"
        )

        avg_exp = experience.mean(skipna=True)

        if np.isnan(avg_exp):
            return recommendations

        if avg_exp < 3:
            recommendations.append(
                "Candidate pool skews junior — consider adjusting "
                "role seniority expectations or adding mentorship capacity"
            )

        elif avg_exp > 8:
            recommendations.append(
                "Candidate pool is highly experienced — "
                "ensure compensation bands remain competitive"
            )

        return recommendations

    def _generate_diversity_recommendations(
        self,
        bias_metrics: Dict[str, Any]
    ) -> List[str]:

        recommendations = []

        gender_ratio = bias_metrics.get("gender_ratio", 1.0)

        if gender_ratio > 1.5:
            recommendations.append(
                "Potential gender imbalance detected — "
                "review sourcing channels and outreach strategy"
            )

        education_bias = (
            bias_metrics
            .get("education_bias", {})
            .get("score", 0)
        )

        if education_bias > 0.5:
            recommendations.append(
                "Consider strengthening skills-based evaluations "
                "to reduce institutional pedigree bias"
            )

        return recommendations

    def _generate_market_recommendations(
        self,
        df: pd.DataFrame
    ) -> List[str]:

        recommendations = []

        candidate_count = len(df)

        if candidate_count < 500:
            recommendations.append(
                "Limited candidate supply detected — "
                "expand sourcing channels and referral programs"
            )

        elif candidate_count > 5000:
            recommendations.append(
                "Large talent pool available — "
                "automated screening and ranking is recommended"
            )

        return recommendations

    # ==========================================
    # Market Insights Helpers
    # ==========================================

    async def _get_top_skills(
        self,
        df: pd.DataFrame,
        n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top skills across candidate pool.
        """

        all_skills = []

        skill_columns = [
            "skills_normalized",
            "skills"
        ]

        for col in skill_columns:
            if col not in df.columns:
                continue

            for skills in df[col].dropna():

                if isinstance(skills, list):
                    all_skills.extend([
                        str(skill).strip().lower()
                        for skill in skills
                        if skill
                    ])

        counter = Counter(all_skills)

        return [
            {
                "skill": skill,
                "count": count
            }
            for skill, count in counter.most_common(n)
        ]

    def _get_top_locations(
        self,
        df: pd.DataFrame,
        n: int = 10
    ) -> List[Dict[str, Any]]:

        if "location" not in df.columns:
            return []

        counts = (
            df["location"]
            .dropna()
            .value_counts()
            .head(n)
        )

        return [
            {
                "location": location,
                "count": int(count)
            }
            for location, count in counts.items()
        ]

    def _get_experience_distribution(
        self,
        df: pd.DataFrame
    ) -> Dict[str, int]:

        if "years_experience" not in df.columns:
            return {}

        experience = pd.to_numeric(
            df["years_experience"],
            errors="coerce"
        )

        bins = [0, 2, 5, 8, 12, 20, 50]
        labels = [
            "0-2 years",
            "2-5 years",
            "5-8 years",
            "8-12 years",
            "12-20 years",
            "20+ years"
        ]

        distribution = (
            pd.cut(
                experience,
                bins=bins,
                labels=labels
            )
            .value_counts()
            .sort_index()
        )

        return {
            str(k): int(v)
            for k, v in distribution.items()
        }

    # ==========================================
    # Scoring / Assessment Helpers
    # ==========================================

    def _calculate_supply_demand(
        self,
        df: pd.DataFrame
    ) -> float:
        """
        Calculate supply-demand score.
        1.0 = abundant supply
        0.0 = scarce talent
        """

        candidate_count = len(df)

        score = min(1.0, candidate_count / 5000)

        return round(score, 2)

    def _assess_hiring_difficulty(
        self,
        df: pd.DataFrame
    ) -> str:

        count = len(df)

        if count > 5000:
            return "Easy"
        elif count > 2000:
            return "Moderate"
        elif count > 500:
            return "Challenging"
        else:
            return "Difficult"

    def _calculate_quality_index(
        self,
        df: pd.DataFrame
    ) -> float:
        """
        Estimate overall candidate quality.
        """

        score = 0.0

        if "years_experience" in df.columns:
            exp_score = min(
                1.0,
                pd.to_numeric(
                    df["years_experience"],
                    errors="coerce"
                ).mean(skipna=True) / 10
            )
            score += exp_score * 0.5

        if "skills_normalized" in df.columns:
            avg_skills = np.mean([
                len(skills)
                for skills in df["skills_normalized"]
                if isinstance(skills, list)
            ]) if len(df) else 0

            skill_score = min(1.0, avg_skills / 20)
            score += skill_score * 0.5

        return round(score, 2)

    def _calculate_remote_readiness(
        self,
        df: pd.DataFrame
    ) -> float:
        """
        Estimate remote-work readiness.
        """

        if "remote_experience" not in df.columns:
            return 0.5

        remote_ratio = (
            df["remote_experience"]
            .fillna(False)
            .mean()
        )

        return round(float(remote_ratio), 2)

    def _assess_market_competitiveness(
        self,
        df: pd.DataFrame
    ) -> str:
        """
        Assess how competitive the market is.
        """

        avg_exp = pd.to_numeric(
            df.get("years_experience"),
            errors="coerce"
        ).mean(skipna=True)

        if avg_exp >= 8:
            return "Highly Competitive"

        elif avg_exp >= 4:
            return "Competitive"

        return "Moderate"

    # ==========================================
    # Fallback Helpers
    # ==========================================

    def _empty_market_response(self) -> Dict[str, Any]:
        """
        Empty fallback response.
        """

        return {
            "candidate_pool_size": 0,
            "average_experience": 0,
            "median_experience": 0,
            "median_salary": None,
            "top_skills": [],
            "top_locations": [],
            "experience_distribution": {},
            "supply_demand_score": 0.0,
            "hiring_difficulty": "Unknown",
            "candidate_quality_index": 0.0,
            "remote_readiness_score": 0.0,
            "market_competitiveness": "Unknown"
        }