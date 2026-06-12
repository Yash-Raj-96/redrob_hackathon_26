"""
Ranking API Endpoints
Enterprise-grade candidate ranking APIs
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
    Body,
    Query,
    status
)

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator
)

from backend.app.dependencies import (
    get_ranker,
    get_candidates_data,
    get_retriever
)

from backend.utils.logger import setup_logger

router = APIRouter(
    prefix="/ranking",
    tags=["Ranking"]
)

logger = setup_logger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

DEFAULT_WEIGHTS = {
    "skill_match": 0.40,
    "experience": 0.25,
    "recruiter_signal": 0.15,
    "availability": 0.10,
    "salary_fit": 0.10
}

VALID_WEIGHT_KEYS = set(DEFAULT_WEIGHTS.keys())


# =========================================================
# REQUEST MODELS
# =========================================================

class RankingRequest(BaseModel):
    """
    Candidate ranking request
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )

    job_description: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Complete job description text"
    )

    candidate_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific candidate IDs to rank"
    )

    top_k: int = Field(
        default=25,
        ge=1,
        le=200,
        description="Number of candidates to return"
    )

    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom scoring weights"
    )

    use_llm_reranking: bool = Field(
        default=False,
        description="Enable LLM reranking"
    )

    include_candidate_data: bool = Field(
        default=True,
        description="Include raw candidate profile data"
    )

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value):
        if value is None:
            return value

        invalid_keys = set(value.keys()) - VALID_WEIGHT_KEYS

        if invalid_keys:
            raise ValueError(
                f"Invalid weight keys: {invalid_keys}"
            )

        total = sum(value.values())

        if total <= 0:
            raise ValueError("Weights total must be > 0")

        # Normalize automatically
        return {
            k: round(v / total, 4)
            for k, v in value.items()
        }


# =========================================================
# RESPONSE MODELS
# =========================================================

class RankedCandidate(BaseModel):
    """
    Ranked candidate response
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    candidate_id: str
    name: str
    rank: int

    total_score: float

    score_breakdown: Dict[str, float]

    matched_skills: List[str] = []
    missing_skills: List[str] = []

    strengths: List[str] = []
    weaknesses: List[str] = []

    candidate_data: Optional[Dict[str, Any]] = None

    llm_explanation: Optional[str] = None


class RankingMetadata(BaseModel):
    """
    Ranking metadata
    """

    ranking_time_ms: float
    retrieved_candidates: int
    reranking_enabled: bool
    weights_used: Dict[str, float]


class RankingResponse(BaseModel):
    """
    Ranking response payload
    """

    timestamp: str

    job_description_summary: str

    total_candidates_ranked: int

    rankings: List[RankedCandidate]

    metadata: RankingMetadata


# =========================================================
# UTILITIES
# =========================================================

def summarize_job_description(text: str) -> str:
    """
    Short JD summary for responses
    """

    clean = " ".join(text.split())

    if len(clean) <= 200:
        return clean

    return clean[:197] + "..."


def normalize_breakdown(
    breakdown: Any
) -> Dict[str, float]:
    """
    Convert ScoreBreakdown objects into dicts
    """

    if breakdown is None:
        return {}

    if hasattr(breakdown, "__dict__"):
        return {
            k: float(v)
            for k, v in breakdown.__dict__.items()
        }

    if isinstance(breakdown, dict):
        return {
            k: float(v)
            for k, v in breakdown.items()
        }

    return {}


# =========================================================
# RANK CANDIDATES
# =========================================================

@router.post(
    "/",
    response_model=RankingResponse,
    status_code=status.HTTP_200_OK
)
async def rank_candidates(
    request: RankingRequest,
    ranker=Depends(get_ranker),
    candidates_df=Depends(get_candidates_data),
    retriever=Depends(get_retriever)
):
    """
    Rank candidates against a job description
    """

    start_time = time.perf_counter()

    try:

        logger.info(
            "Ranking request received",
            extra={
                "top_k": request.top_k,
                "reranking": request.use_llm_reranking
            }
        )

        # =====================================================
        # RETRIEVE CANDIDATES
        # =====================================================

        if request.candidate_ids:

            candidate_ids = request.candidate_ids

        else:

            logger.info("Running semantic retrieval")

            retrieval_results = await retriever.search(
                query=request.job_description,
                top_k=request.top_k * 2
            )

            candidate_ids = [
                r.get("candidate_id") or r.get("id")
                for r in retrieval_results
            ]

            candidate_ids = [
                cid for cid in candidate_ids
                if cid
            ][: request.top_k]

        if not candidate_ids:
            raise HTTPException(
                status_code=404,
                detail="No candidates found"
            )

        logger.info(
            f"Retrieved {len(candidate_ids)} candidates"
        )

        # =====================================================
        # RUN RANKING
        # =====================================================

        ranked_results = await ranker.rank(
            job_description=request.job_description,
            candidate_ids=candidate_ids,
            candidates_df=candidates_df,
            custom_weights=request.weights,
            use_llm_reranking=request.use_llm_reranking
        )

        # =====================================================
        # FORMAT RESPONSE
        # =====================================================

        rankings: List[RankedCandidate] = []

        for idx, candidate in enumerate(
            ranked_results,
            start=1
        ):

            breakdown = normalize_breakdown(
                candidate.get("score_breakdown")
            )

            rankings.append(
                RankedCandidate(
                    candidate_id=candidate.get(
                        "candidate_id",
                        ""
                    ),
                    name=candidate.get(
                        "name",
                        "Unknown"
                    ),
                    rank=idx,
                    total_score=round(
                        float(
                            candidate.get(
                                "total_score",
                                0.0
                            )
                        ),
                        4
                    ),
                    score_breakdown=breakdown,
                    matched_skills=candidate.get(
                        "matched_skills",
                        []
                    ),
                    missing_skills=candidate.get(
                        "missing_skills",
                        []
                    ),
                    strengths=candidate.get(
                        "strengths",
                        []
                    ),
                    weaknesses=candidate.get(
                        "weaknesses",
                        []
                    ),
                    candidate_data=(
                        candidate.get("candidate_data")
                        if request.include_candidate_data
                        else None
                    ),
                    llm_explanation=candidate.get(
                        "llm_explanation"
                    )
                )
            )

        elapsed_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2
        )

        logger.info(
            f"Ranking completed in {elapsed_ms} ms"
        )

        return RankingResponse(
            timestamp=datetime.utcnow().isoformat(),
            job_description_summary=summarize_job_description(
                request.job_description
            ),
            total_candidates_ranked=len(rankings),
            rankings=rankings,
            metadata=RankingMetadata(
                ranking_time_ms=elapsed_ms,
                retrieved_candidates=len(candidate_ids),
                reranking_enabled=request.use_llm_reranking,
                weights_used=(
                    request.weights
                    or DEFAULT_WEIGHTS
                )
            )
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Candidate ranking failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Ranking failed: {str(exc)}"
        )


# =========================================================
# UPDATE WEIGHTS
# =========================================================

@router.post("/weights/update")
async def update_scoring_weights(
    weights: Dict[str, float] = Body(...)
):
    """
    Update scoring weights dynamically
    """

    try:

        if not weights:
            raise ValueError(
                "Weights payload cannot be empty"
            )

        invalid_keys = (
            set(weights.keys()) - VALID_WEIGHT_KEYS
        )

        if invalid_keys:
            raise ValueError(
                f"Invalid keys: {invalid_keys}"
            )

        total = sum(weights.values())

        if total <= 0:
            raise ValueError(
                "Total weight must be > 0"
            )

        normalized = {
            k: round(v / total, 4)
            for k, v in weights.items()
        }

        logger.info(
            f"Scoring weights updated: {normalized}"
        )

        return {
            "success": True,
            "message": "Weights updated successfully",
            "weights": normalized,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as exc:

        logger.error(str(exc))

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )


# =========================================================
# COMPARE CANDIDATES
# =========================================================

@router.get("/compare")
async def compare_candidates(
    candidate_id_1: str = Query(...),
    candidate_id_2: str = Query(...),
    job_description: str = Query(...),

    ranker=Depends(get_ranker),
    candidates_df=Depends(get_candidates_data)
):
    """
    Compare two candidates side-by-side
    """

    try:

        ranked = await ranker.rank(
            job_description=job_description,
            candidate_ids=[
                candidate_id_1,
                candidate_id_2
            ],
            candidates_df=candidates_df
        )

        if len(ranked) < 2:
            raise HTTPException(
                status_code=404,
                detail="Unable to compare candidates"
            )

        ranked = sorted(
            ranked,
            key=lambda x: x["total_score"],
            reverse=True
        )

        winner = ranked[0]
        loser = ranked[1]

        comparison = {
            "winner_candidate_id": winner["candidate_id"],
            "winner_name": winner["name"],

            "score_difference": round(
                abs(
                    winner["total_score"] -
                    loser["total_score"]
                ),
                4
            ),

            "candidate_1": ranked[0],
            "candidate_2": ranked[1],

            "comparison_summary": (
                f"{winner['name']} ranked higher "
                f"primarily due to stronger "
                f"overall alignment with the role."
            ),

            "timestamp": datetime.utcnow().isoformat()
        }

        return comparison

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Candidate comparison failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# =========================================================
# GET DEFAULT WEIGHTS
# =========================================================

@router.get("/weights/default")
async def get_default_weights():
    """
    Get default scoring weights
    """

    return {
        "weights": DEFAULT_WEIGHTS,
        "timestamp": datetime.utcnow().isoformat()
    }
