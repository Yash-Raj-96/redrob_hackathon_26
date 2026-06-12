"""
Explainability API endpoints - Enterprise Grade
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

from backend.app.dependencies import (
    get_ranker,
    get_candidates_data
)
from backend.llm.explanation_generator import ExplanationGenerator
from backend.utils.logger import setup_logger

router = APIRouter(
    prefix="/explain",
    tags=["Explainability"]
)

logger = setup_logger(__name__)

# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================

class ExplainRequest(BaseModel):
    """Explainability request"""

    candidate_id: str = Field(..., description="Candidate identifier")
    job_description: str = Field(..., min_length=10)
    include_llm_explanation: bool = True
    include_similar_candidates: bool = True


class BatchExplainRequest(BaseModel):
    """Batch explanation request"""

    candidate_ids: List[str]
    job_description: str
    include_llm_explanation: bool = False


class FeatureContribution(BaseModel):
    """Individual feature contribution"""

    feature_name: str
    score: float
    weight: float
    weighted_score: float
    explanation: str
    improvement_suggestions: List[str] = []


class SimilarCandidate(BaseModel):
    """Similar candidate response"""

    candidate_id: str
    name: str
    similarity_score: float
    current_role: Optional[str] = None
    years_experience: Optional[float] = None


class CandidateExplanation(BaseModel):
    """Complete explanation payload"""

    candidate_id: str
    name: str
    total_score: float
    generated_at: str

    feature_contributions: List[FeatureContribution]

    strengths: List[str]
    weaknesses: List[str]

    recruiter_summary: str
    llm_explanation: Optional[str] = None

    similar_candidates: List[SimilarCandidate] = []
    improvement_tips: List[str] = []

    matched_skills: List[str] = []
    missing_skills: List[str] = []


# =========================================================
# MAIN EXPLANATION ENDPOINT
# =========================================================

@router.post(
    "/candidate",
    response_model=CandidateExplanation
)
async def explain_candidate_ranking(
    request: ExplainRequest,
    ranker=Depends(get_ranker),
    candidates_df=Depends(get_candidates_data),
    explanation_gen=Depends(lambda: ExplanationGenerator())
):
    """
    Generate detailed explanation for candidate ranking
    """

    try:

        logger.info(
            f"Generating explanation for candidate "
            f"{request.candidate_id}"
        )

        # -------------------------------------------------
        # GET CANDIDATE
        # -------------------------------------------------

        candidate_rows = candidates_df[
            candidates_df["candidate_id"] == request.candidate_id
        ]

        if candidate_rows.empty:
            raise HTTPException(
                status_code=404,
                detail="Candidate not found"
            )

        candidate_data = candidate_rows.iloc[0].to_dict()

        # -------------------------------------------------
        # GET DETAILED SCORES
        # -------------------------------------------------

        score_details = await ranker.get_detailed_scores(
            job_description=request.job_description,
            candidate=candidate_data
        )

        # -------------------------------------------------
        # FEATURE CONTRIBUTIONS
        # -------------------------------------------------

        feature_contributions = []

        for feature_name, details in score_details.get(
            "features",
            {}
        ).items():

            feature_contributions.append(
                FeatureContribution(
                    feature_name=feature_name,
                    score=round(details.get("score", 0), 4),
                    weight=round(details.get("weight", 0), 4),
                    weighted_score=round(
                        details.get("weighted_score", 0),
                        4
                    ),
                    explanation=details.get(
                        "explanation",
                        "No explanation available"
                    ),
                    improvement_suggestions=details.get(
                        "suggestions",
                        []
                    )
                )
            )

        # -------------------------------------------------
        # LLM EXPLANATION
        # -------------------------------------------------

        llm_explanation = None

        if request.include_llm_explanation:

            try:

                llm_explanation = (
                    await explanation_gen.generate_explanation(
                        candidate=candidate_data,
                        job_description=request.job_description,
                        score_breakdown=score_details
                    )
                )

            except Exception as e:

                logger.warning(
                    f"LLM explanation failed: {str(e)}"
                )

                llm_explanation = (
                    "AI explanation currently unavailable."
                )

        # -------------------------------------------------
        # SIMILAR CANDIDATES
        # -------------------------------------------------

        similar_candidates = []

        if request.include_similar_candidates:

            similar_candidates = await find_similar_candidates(
                candidate_data,
                candidates_df
            )

        # -------------------------------------------------
        # RECRUITER SUMMARY
        # -------------------------------------------------

        recruiter_summary = build_recruiter_summary(
            candidate_data,
            score_details
        )

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return CandidateExplanation(
            candidate_id=request.candidate_id,
            name=candidate_data.get("name", "Unknown"),
            total_score=round(
                score_details.get("total_score", 0),
                4
            ),
            generated_at=datetime.utcnow().isoformat(),

            feature_contributions=feature_contributions,

            strengths=score_details.get(
                "strengths",
                []
            ),

            weaknesses=score_details.get(
                "weaknesses",
                []
            ),

            recruiter_summary=recruiter_summary,

            llm_explanation=llm_explanation,

            similar_candidates=similar_candidates,

            improvement_tips=generate_improvement_tips(
                score_details
            ),

            matched_skills=score_details.get(
                "matched_skills",
                []
            ),

            missing_skills=score_details.get(
                "missing_skills",
                []
            )
        )

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            f"Explanation generation failed: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate explanation"
        )


# =========================================================
# BATCH EXPLANATIONS
# =========================================================

@router.post("/batch")
async def explain_multiple_candidates(
    request: BatchExplainRequest,
    ranker=Depends(get_ranker),
    candidates_df=Depends(get_candidates_data)
):
    """
    Generate explanations for multiple candidates
    """

    explanations = []

    for candidate_id in request.candidate_ids[:25]:

        try:

            candidate_rows = candidates_df[
                candidates_df["candidate_id"] == candidate_id
            ]

            if candidate_rows.empty:
                continue

            candidate_data = candidate_rows.iloc[0].to_dict()

            score_details = await ranker.get_detailed_scores(
                request.job_description,
                candidate_data
            )

            explanations.append({
                "candidate_id": candidate_id,
                "name": candidate_data.get("name"),
                "total_score": score_details.get(
                    "total_score",
                    0
                ),
                "strengths": score_details.get(
                    "strengths",
                    []
                ),
                "weaknesses": score_details.get(
                    "weaknesses",
                    []
                ),
                "matched_skills": score_details.get(
                    "matched_skills",
                    []
                ),
                "missing_skills": score_details.get(
                    "missing_skills",
                    []
                )
            })

        except Exception as e:

            logger.warning(
                f"Failed for candidate "
                f"{candidate_id}: {str(e)}"
            )

    return {
        "count": len(explanations),
        "results": explanations,
        "generated_at": datetime.utcnow().isoformat()
    }


# =========================================================
# FEATURE IMPORTANCE ENDPOINT
# =========================================================

@router.get("/feature-importance")
async def get_feature_importance(
    ranker=Depends(get_ranker)
):
    """
    Return current scoring weights
    """

    weights = ranker.scorer.weights

    return {
        "weights": weights,
        "normalized": abs(sum(weights.values()) - 1.0) < 0.01,
        "timestamp": datetime.utcnow().isoformat()
    }


# =========================================================
# SIMILARITY SEARCH
# =========================================================

async def find_similar_candidates(
    candidate: Dict,
    candidates_df: pd.DataFrame,
    limit: int = 5
) -> List[SimilarCandidate]:
    """
    Find similar candidates using skills + experience
    """

    try:

        candidate_skills = set(
            candidate.get("skills", [])
        )

        candidate_exp = candidate.get(
            "years_experience",
            0
        )

        similarities = []

        for _, row in candidates_df.iterrows():

            if (
                row["candidate_id"]
                == candidate["candidate_id"]
            ):
                continue

            row_skills = set(row.get("skills", []))

            # Jaccard skill similarity
            union = candidate_skills | row_skills

            if not union:
                skill_similarity = 0
            else:
                skill_similarity = (
                    len(candidate_skills & row_skills)
                    / len(union)
                )

            # Experience similarity
            exp_diff = abs(
                candidate_exp -
                row.get("years_experience", 0)
            )

            exp_similarity = max(
                0,
                1 - (exp_diff / 15)
            )

            final_similarity = (
                0.7 * skill_similarity +
                0.3 * exp_similarity
            )

            similarities.append({
                "candidate_id": row["candidate_id"],
                "name": row.get("name"),
                "similarity_score": round(
                    final_similarity,
                    4
                ),
                "current_role": row.get(
                    "current_role"
                ),
                "years_experience": row.get(
                    "years_experience"
                )
            })

        similarities = sorted(
            similarities,
            key=lambda x: x["similarity_score"],
            reverse=True
        )

        return [
            SimilarCandidate(**item)
            for item in similarities[:limit]
        ]

    except Exception as e:

        logger.warning(
            f"Similarity search failed: {str(e)}"
        )

        return []


# =========================================================
# RECRUITER SUMMARY
# =========================================================

def build_recruiter_summary(
    candidate: Dict,
    score_details: Dict
) -> str:
    """
    Build concise recruiter-facing summary
    """

    strengths = score_details.get(
        "strengths",
        []
    )

    weaknesses = score_details.get(
        "weaknesses",
        []
    )

    matched_skills = score_details.get(
        "matched_skills",
        []
    )

    summary = (
        f"{candidate.get('name', 'Candidate')} "
        f"is a "
        f"{candidate.get('current_role', 'professional')} "
        f"with "
        f"{candidate.get('years_experience', 0)} "
        f"years of experience. "
    )

    if matched_skills:
        summary += (
            f"Strong alignment in "
            f"{', '.join(matched_skills[:5])}. "
        )

    if strengths:
        summary += (
            f"Key strengths include: "
            f"{'; '.join(strengths[:2])}. "
        )

    if weaknesses:
        summary += (
            f"Primary concerns: "
            f"{'; '.join(weaknesses[:2])}."
        )

    return summary


# =========================================================
# IMPROVEMENT TIPS
# =========================================================

def generate_improvement_tips(
    score_details: Dict
) -> List[str]:
    """
    Generate actionable candidate improvement tips
    """

    tips = []

    missing_skills = score_details.get(
        "missing_skills",
        []
    )

    if missing_skills:

        tips.append(
            "Develop expertise in: "
            + ", ".join(missing_skills[:3])
        )

    experience_score = (
        score_details.get("features", {})
        .get("experience", {})
        .get("score", 1)
    )

    if experience_score < 0.5:

        tips.append(
            "Highlight more domain-specific "
            "project experience"
        )

    recruiter_signal_score = (
        score_details.get("features", {})
        .get("recruiter_signal", {})
        .get("score", 1)
    )

    if recruiter_signal_score < 0.5:

        tips.append(
            "Improve profile completeness with "
            "certifications, GitHub, and achievements"
        )

    availability_score = (
        score_details.get("features", {})
        .get("availability", {})
        .get("score", 1)
    )

    if availability_score < 0.5:

        tips.append(
            "Consider reducing notice period "
            "to improve hiring competitiveness"
        )

    return tips[:5]