"""
Search API Endpoints
Enterprise-grade candidate search APIs
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from fastapi import (
    APIRouter,
    HTTPException,
    Depends,
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
    get_retriever,
    get_candidates_data
)

from backend.utils.logger import setup_logger

router = APIRouter(
    prefix="/search",
    tags=["Search"]
)

logger = setup_logger(__name__)


# =========================================================
# REQUEST MODELS
# =========================================================

class SearchFilters(BaseModel):
    """
    Candidate filtering options
    """

    model_config = ConfigDict(extra="forbid")

    min_experience: Optional[float] = Field(
        default=None,
        ge=0
    )

    max_experience: Optional[float] = Field(
        default=None,
        le=50
    )

    location: Optional[str] = None

    current_role: Optional[str] = None

    skills: Optional[List[str]] = None

    availability: Optional[str] = None


class SearchRequest(BaseModel):
    """
    Search request payload
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    query: str = Field(
        ...,
        min_length=2,
        max_length=5000,
        description="Search query or job description"
    )

    top_k: int = Field(
        default=25,
        ge=1,
        le=500,
        description="Maximum number of results"
    )

    page: int = Field(
        default=1,
        ge=1
    )

    page_size: int = Field(
        default=25,
        ge=1,
        le=100
    )

    filters: Optional[SearchFilters] = None

    use_hybrid: bool = Field(
        default=True,
        description="Enable hybrid semantic + keyword retrieval"
    )

    include_resume_text: bool = Field(
        default=False,
        description="Include resume text in response"
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        return value.strip()


# =========================================================
# RESPONSE MODELS
# =========================================================

class SearchResult(BaseModel):
    """
    Individual search result
    """

    candidate_id: str

    name: str

    current_role: Optional[str] = None

    years_experience: float = 0

    location: Optional[str] = None

    relevance_score: float

    matched_skills: List[str] = []

    highlights: List[str] = []

    resume_summary: Optional[str] = None


class SearchMetadata(BaseModel):
    """
    Search metadata
    """

    search_time_ms: float

    hybrid_search: bool

    filters_applied: Optional[Dict[str, Any]]

    page: int

    page_size: int


class SearchResponse(BaseModel):
    """
    Search response payload
    """

    timestamp: str

    query: str

    total_results: int

    results: List[SearchResult]

    metadata: SearchMetadata


# =========================================================
# UTILITIES
# =========================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert values to float
    """

    try:
        return float(value)

    except Exception:
        return default


def extract_highlights(
    query: str,
    skills: List[str]
) -> List[str]:
    """
    Generate search highlights
    """

    query_terms = set(
        query.lower().split()
    )

    highlights = []

    for skill in skills:

        if skill.lower() in query_terms:
            highlights.append(
                f"Matched skill: {skill}"
            )

    return highlights[:5]


# =========================================================
# SEARCH ENDPOINT
# =========================================================

@router.post(
    "/",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK
)
async def search_candidates(
    request: SearchRequest,
    retriever=Depends(get_retriever),
    candidates_df=Depends(get_candidates_data)
):
    """
    Search candidates using semantic or hybrid retrieval
    """

    start_time = time.perf_counter()

    try:

        logger.info(
            "Candidate search started",
            extra={
                "query": request.query,
                "top_k": request.top_k,
                "hybrid": request.use_hybrid
            }
        )

        # =====================================================
        # RUN RETRIEVAL
        # =====================================================

        retrieval_results = await retriever.search(
            query=request.query,
            top_k=request.top_k,
            filters=(
                request.filters.model_dump()
                if request.filters
                else None
            ),
            use_hybrid=request.use_hybrid
        )

        if not retrieval_results:

            return SearchResponse(
                timestamp=datetime.utcnow().isoformat(),
                query=request.query,
                total_results=0,
                results=[],
                metadata=SearchMetadata(
                    search_time_ms=0,
                    hybrid_search=request.use_hybrid,
                    filters_applied=(
                        request.filters.model_dump()
                        if request.filters
                        else None
                    ),
                    page=request.page,
                    page_size=request.page_size
                )
            )

        # =====================================================
        # FORMAT RESULTS
        # =====================================================

        formatted_results: List[SearchResult] = []

        for result in retrieval_results:

            candidate_id = (
                result.get("candidate_id")
                or result.get("id")
            )

            if not candidate_id:
                continue

            candidate_rows = candidates_df[
                candidates_df["candidate_id"]
                == candidate_id
            ]

            if candidate_rows.empty:
                continue

            candidate = (
                candidate_rows.iloc[0].to_dict()
            )

            skills = candidate.get(
                "skills",
                []
            )

            if not isinstance(skills, list):
                skills = []

            formatted_results.append(
                SearchResult(
                    candidate_id=candidate_id,
                    name=candidate.get(
                        "name",
                        "Unknown"
                    ),
                    current_role=candidate.get(
                        "current_role",
                        "N/A"
                    ),
                    years_experience=safe_float(
                        candidate.get(
                            "years_experience",
                            0
                        )
                    ),
                    location=candidate.get(
                        "location",
                        "N/A"
                    ),
                    relevance_score=round(
                        safe_float(
                            result.get("score", 0)
                        ),
                        4
                    ),
                    matched_skills=result.get(
                        "matched_skills",
                        []
                    ),
                    highlights=extract_highlights(
                        request.query,
                        skills
                    ),
                    resume_summary=(
                        candidate.get(
                            "resume_text",
                            ""
                        )[:300]
                        if request.include_resume_text
                        else None
                    )
                )
            )

        # =====================================================
        # PAGINATION
        # =====================================================

        start_idx = (
            (request.page - 1)
            * request.page_size
        )

        end_idx = (
            start_idx + request.page_size
        )

        paginated_results = (
            formatted_results[start_idx:end_idx]
        )

        elapsed_ms = round(
            (
                time.perf_counter()
                - start_time
            ) * 1000,
            2
        )

        logger.info(
            f"Search completed in {elapsed_ms} ms"
        )

        return SearchResponse(
            timestamp=datetime.utcnow().isoformat(),
            query=request.query,
            total_results=len(formatted_results),
            results=paginated_results,
            metadata=SearchMetadata(
                search_time_ms=elapsed_ms,
                hybrid_search=request.use_hybrid,
                filters_applied=(
                    request.filters.model_dump()
                    if request.filters
                    else None
                ),
                page=request.page,
                page_size=request.page_size
            )
        )

    except Exception as exc:

        logger.exception(
            "Candidate search failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(exc)}"
        )


# =========================================================
# SEARCH SUGGESTIONS
# =========================================================

POPULAR_SUGGESTIONS = [
    "Python Developer",
    "Machine Learning Engineer",
    "Data Scientist",
    "Backend Engineer",
    "Frontend Developer",
    "DevOps Engineer",
    "Cloud Architect",
    "Full Stack Developer",
    "React Developer",
    "AI Engineer",
    "NLP Engineer",
    "Data Analyst",
    "Java Developer",
    "Golang Developer"
]


@router.get("/suggest")
async def get_search_suggestions(
    prefix: str = Query(
        ...,
        min_length=2,
        max_length=50
    ),

    limit: int = Query(
        default=10,
        ge=1,
        le=50
    )
):
    """
    Search autocomplete suggestions
    """

    try:

        prefix_lower = prefix.lower()

        matched = [
            suggestion
            for suggestion in POPULAR_SUGGESTIONS
            if prefix_lower in suggestion.lower()
        ]

        return {
            "prefix": prefix,
            "suggestions": matched[:limit],
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as exc:

        logger.exception(
            "Suggestion lookup failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# =========================================================
# ADVANCED FILTERING
# =========================================================

@router.post("/filter")
async def filter_candidates(
    filters: SearchFilters,
    candidates_df=Depends(get_candidates_data)
):
    """
    Apply advanced candidate filters
    """

    try:

        filtered_df = candidates_df.copy()

        # =====================================================
        # EXPERIENCE FILTERS
        # =====================================================

        if filters.min_experience is not None:

            filtered_df = filtered_df[
                filtered_df["years_experience"]
                >= filters.min_experience
            ]

        if filters.max_experience is not None:

            filtered_df = filtered_df[
                filtered_df["years_experience"]
                <= filters.max_experience
            ]

        # =====================================================
        # LOCATION FILTER
        # =====================================================

        if filters.location:

            filtered_df = filtered_df[
                filtered_df["location"]
                .fillna("")
                .str.contains(
                    filters.location,
                    case=False
                )
            ]

        # =====================================================
        # ROLE FILTER
        # =====================================================

        if filters.current_role:

            filtered_df = filtered_df[
                filtered_df["current_role"]
                .fillna("")
                .str.contains(
                    filters.current_role,
                    case=False
                )
            ]

        # =====================================================
        # SKILL FILTER
        # =====================================================

        if filters.skills:

            required_skills = set(
                skill.lower()
                for skill in filters.skills
            )

            def count_matches(skills):

                if not isinstance(skills, list):
                    return 0

                normalized = set(
                    s.lower()
                    for s in skills
                )

                return len(
                    required_skills.intersection(
                        normalized
                    )
                )

            filtered_df[
                "skill_match_count"
            ] = filtered_df[
                "skills"
            ].apply(count_matches)

            filtered_df = filtered_df[
                filtered_df[
                    "skill_match_count"
                ] > 0
            ]

        # =====================================================
        # RESPONSE
        # =====================================================

        return {
            "total_matches": len(filtered_df),

            "candidate_ids": filtered_df[
                "candidate_id"
            ].tolist()[:100],

            "filters_applied": (
                filters.model_dump()
            ),

            "timestamp": (
                datetime.utcnow()
                .isoformat()
            )
        }

    except Exception as exc:

        logger.exception(
            "Candidate filtering failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# =========================================================
# POPULAR SKILLS
# =========================================================

@router.get("/popular-skills")
async def get_popular_skills(
    candidates_df=Depends(get_candidates_data)
):
    """
    Get most common skills in database
    """

    try:

        from collections import Counter

        all_skills = []

        for skills in candidates_df[
            "skills"
        ].dropna():

            if isinstance(skills, list):
                all_skills.extend(skills)

        skill_counts = Counter(
            all_skills
        ).most_common(25)

        return {
            "skills": [
                {
                    "skill": skill,
                    "count": count
                }
                for skill, count
                in skill_counts
            ],
            "timestamp": (
                datetime.utcnow()
                .isoformat()
            )
        }

    except Exception as exc:

        logger.exception(
            "Popular skills lookup failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )
