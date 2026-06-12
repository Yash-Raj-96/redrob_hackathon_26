"""
Analytics and insights API endpoints
"""

from datetime import datetime
from collections import Counter
from typing import Dict, Any, List, Optional

import asyncio
import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    Query
)

from pydantic import BaseModel, Field

from backend.app.dependencies import (
    get_candidates_data
)

from backend.analytics.bias_detection import (
    BiasDetector
)

from backend.analytics.skill_gap_analysis import (
    SkillGapAnalyzer
)

from backend.analytics.hiring_insights import (
    HiringInsights
)

from backend.utils.logger import (
    setup_logger
)

router = APIRouter(
    prefix="/analytics",
    tags=["Analytics"]
)

logger = setup_logger(__name__)


# =========================================================
# RESPONSE MODELS
# =========================================================

class BiasMetric(BaseModel):
    score: float
    severity: str
    message: str


class SkillGap(BaseModel):
    skill: str
    candidates_with_skill: int
    percentage: float
    severity: str
    recommendation: str


class AnalyticsHealth(BaseModel):
    data_quality_score: float
    duplicate_profiles: int
    incomplete_profiles: int
    active_candidates_percentage: float


class AnalyticsResponse(BaseModel):
    total_candidates: int

    skill_distribution: Dict[str, int]

    experience_distribution: Dict[str, int]

    location_distribution: Dict[str, int]

    top_roles: Dict[str, int]

    bias_metrics: Dict[str, Any]

    skill_gaps: List[SkillGap]

    hiring_recommendations: List[str]

    analytics_health: AnalyticsHealth

    generated_at: str


# =========================================================
# HELPERS
# =========================================================

def build_experience_distribution(
    candidates_df: pd.DataFrame
) -> Dict[str, int]:

    if "years_experience" not in candidates_df.columns:
        return {}

    exp_bins = [0, 2, 5, 8, 12, 20, 50]

    exp_labels = [
        "0-2 years",
        "2-5 years",
        "5-8 years",
        "8-12 years",
        "12-20 years",
        "20+ years"
    ]

    distribution = pd.cut(
        candidates_df["years_experience"].fillna(0),
        bins=exp_bins,
        labels=exp_labels,
        include_lowest=True
    )

    return {
        str(k): int(v)
        for k, v in distribution.value_counts().to_dict().items()
    }


def build_skill_distribution(
    candidates_df: pd.DataFrame,
    limit: int = 25
) -> Dict[str, int]:

    all_skills = []

    if "skills" not in candidates_df.columns:
        return {}

    for skills in candidates_df["skills"].dropna():

        if isinstance(skills, list):
            all_skills.extend([
                str(skill).strip()
                for skill in skills
            ])

    return dict(
        Counter(all_skills).most_common(limit)
    )


def build_location_distribution(
    candidates_df: pd.DataFrame,
    limit: int = 15
) -> Dict[str, int]:

    if "location" not in candidates_df.columns:
        return {}

    locations = (
        candidates_df["location"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(limit)
    )

    return {
        str(k): int(v)
        for k, v in locations.to_dict().items()
    }


def build_role_distribution(
    candidates_df: pd.DataFrame,
    limit: int = 15
) -> Dict[str, int]:

    if "current_role" not in candidates_df.columns:
        return {}

    roles = (
        candidates_df["current_role"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(limit)
    )

    return {
        str(k): int(v)
        for k, v in roles.to_dict().items()
    }


def calculate_data_health(
    candidates_df: pd.DataFrame
) -> AnalyticsHealth:

    total = len(candidates_df)

    if total == 0:

        return AnalyticsHealth(
            data_quality_score=0,
            duplicate_profiles=0,
            incomplete_profiles=0,
            active_candidates_percentage=0
        )

    duplicate_profiles = int(
        candidates_df.duplicated(
            subset=["candidate_id"]
        ).sum()
    )

    incomplete_profiles = int(
        candidates_df.isnull().sum(axis=1).gt(3).sum()
    )

    quality_score = max(
        0,
        1 - (
            (
                duplicate_profiles
                + incomplete_profiles
            ) / total
        )
    )

    active_candidates = 0

    if "profile_active" in candidates_df.columns:

        active_candidates = int(
            candidates_df[
                candidates_df["profile_active"] == True
            ].shape[0]
        )

    else:
        active_candidates = total

    return AnalyticsHealth(
        data_quality_score=round(quality_score, 2),
        duplicate_profiles=duplicate_profiles,
        incomplete_profiles=incomplete_profiles,
        active_candidates_percentage=round(
            active_candidates / total,
            2
        )
    )


# =========================================================
# DASHBOARD ENDPOINT
# =========================================================

@router.get(
    "/dashboard",
    response_model=AnalyticsResponse
)
async def get_analytics_dashboard(
    candidates_df=Depends(get_candidates_data)
):
    """
    Get complete analytics dashboard data.
    """

    try:

        logger.info(
            "Generating analytics dashboard"
        )

        if candidates_df is None or candidates_df.empty:

            raise HTTPException(
                status_code=404,
                detail="No candidate data available"
            )

        # Initialize analyzers
        bias_detector = BiasDetector()

        skill_analyzer = SkillGapAnalyzer()

        insights_generator = HiringInsights()

        # Parallel analytics execution
        (
            bias_metrics,
            skill_gaps,
            recommendations
        ) = await asyncio.gather(

            bias_detector.analyze(candidates_df),

            skill_analyzer.find_gaps(
                candidates_df
            ),

            insights_generator.generate_recommendations(
                candidates_df
            )
        )

        response = AnalyticsResponse(

            total_candidates=len(candidates_df),

            skill_distribution=build_skill_distribution(
                candidates_df
            ),

            experience_distribution=(
                build_experience_distribution(
                    candidates_df
                )
            ),

            location_distribution=(
                build_location_distribution(
                    candidates_df
                )
            ),

            top_roles=build_role_distribution(
                candidates_df
            ),

            bias_metrics=bias_metrics,

            skill_gaps=skill_gaps,

            hiring_recommendations=recommendations,

            analytics_health=calculate_data_health(
                candidates_df
            ),

            generated_at=datetime.utcnow().isoformat()
        )

        logger.info(
            "Analytics dashboard generated successfully"
        )

        return response

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            f"Analytics dashboard failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate analytics dashboard"
        )


# =========================================================
# SKILL GAPS FOR ROLE
# =========================================================

@router.get("/skill-gaps/{job_title}")
async def get_skill_gaps_for_role(
    job_title: str,
    candidates_df=Depends(get_candidates_data)
):
    """
    Get role-specific skill gaps.
    """

    try:

        analyzer = SkillGapAnalyzer()

        gaps = await analyzer.find_role_specific_gaps(
            job_title=job_title,
            candidates_df=candidates_df
        )

        return {
            "job_title": job_title,
            "skill_gaps": gaps,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:

        logger.exception(
            f"Skill gap analysis failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to analyze skill gaps"
        )


# =========================================================
# BIAS REPORT
# =========================================================

@router.get("/bias-report")
async def get_bias_report(
    candidates_df=Depends(get_candidates_data)
):
    """
    Generate full bias & fairness report.
    """

    try:

        detector = BiasDetector()

        report = await detector.generate_report(
            candidates_df
        )

        return {
            "report": report,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:

        logger.exception(
            f"Bias report generation failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate bias report"
        )


# =========================================================
# HIRING TRENDS
# =========================================================

@router.get("/trends")
async def get_hiring_trends(
    days: int = Query(
        default=30,
        ge=1,
        le=365
    ),
    candidates_df=Depends(get_candidates_data)
):
    """
    Get hiring trends and candidate growth metrics.
    """

    try:

        if (
            "created_at"
            not in candidates_df.columns
        ):

            return {
                "message": (
                    "Historical trend data unavailable"
                ),
                "generated_at": (
                    datetime.utcnow().isoformat()
                )
            }

        df = candidates_df.copy()

        df["created_at"] = pd.to_datetime(
            df["created_at"],
            errors="coerce"
        )

        cutoff = datetime.utcnow() - pd.Timedelta(
            days=days
        )

        recent_df = df[
            df["created_at"] >= cutoff
        ]

        trend_counts = (
            recent_df
            .groupby(
                recent_df["created_at"].dt.date
            )
            .size()
            .to_dict()
        )

        return {
            "days": days,
            "candidate_growth_trend": {
                str(k): int(v)
                for k, v in trend_counts.items()
            },
            "total_new_candidates": int(
                len(recent_df)
            ),
            "generated_at": (
                datetime.utcnow().isoformat()
            )
        }

    except Exception as e:

        logger.exception(
            f"Hiring trends failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate hiring trends"
        )


# =========================================================
# ANALYTICS HEALTH
# =========================================================

@router.get("/health")
async def analytics_health_check(
    candidates_df=Depends(get_candidates_data)
):
    """
    Analytics system health endpoint.
    """

    try:

        health = calculate_data_health(
            candidates_df
        )

        return {
            "status": "healthy",
            "analytics_health": health,
            "generated_at": (
                datetime.utcnow().isoformat()
            )
        }

    except Exception as e:

        logger.exception(
            f"Analytics health check failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Analytics health check failed"
        )
