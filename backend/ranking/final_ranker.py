"""
Final ranking orchestrator combining all components
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import pandas as pd
import numpy as np

from backend.ranking.score_candidates import (
    CandidateScorer,
    ScoreBreakdown
)
from backend.jd_parser.parse_jd import JDParser
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class FinalRanker:
    """
    Final orchestrator for candidate ranking.

    Responsibilities:
    - Parse JD
    - Score candidates
    - Normalize & sort scores
    - Apply optional LLM reranking
    - Generate explainability insights
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        use_llm_reranking: bool = False,
        normalize_scores: bool = True,
        max_concurrent_tasks: int = 25
    ):

        self.scorer = CandidateScorer(weights)
        self.jd_parser = JDParser()

        self.use_llm_reranking = use_llm_reranking
        self.normalize_scores = normalize_scores
        self.max_concurrent_tasks = max_concurrent_tasks

        self.llm_reranker = None

        if self.use_llm_reranking:
            try:
                from backend.llm.reranker import LLMReranker
                self.llm_reranker = LLMReranker()

            except Exception as e:
                logger.warning(
                    f"LLM reranker unavailable: {str(e)}"
                )

    # =========================================================
    # MAIN RANK METHOD
    # =========================================================

    async def rank(
        self,
        job_description: str,
        candidate_ids: List[str],
        candidates_df: pd.DataFrame,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank candidates for a job description.
        """

        logger.info(
            f"Starting ranking for {len(candidate_ids)} candidates"
        )

        if not candidate_ids:
            logger.warning("No candidate IDs provided")
            return []

        if candidates_df.empty:
            logger.warning("Candidates dataframe is empty")
            return []

        # Update weights dynamically
        if custom_weights:
            self.scorer.update_weights(custom_weights)

        # Parse JD
        job_requirements = await self.jd_parser.parse(
            job_description
        )

        logger.info(
            f"Parsed JD successfully | "
            f"Required skills: "
            f"{len(job_requirements.get('required_skills', []))}"
        )

        # Build lookup map for performance
        candidate_lookup = {
            str(row["candidate_id"]): row.to_dict()
            for _, row in candidates_df.iterrows()
        }

        semaphore = asyncio.Semaphore(
            self.max_concurrent_tasks
        )

        async def process_candidate(candidate_id: str):

            async with semaphore:

                try:

                    candidate = candidate_lookup.get(
                        str(candidate_id)
                    )

                    if not candidate:
                        logger.warning(
                            f"Candidate not found: {candidate_id}"
                        )
                        return None

                    total_score, breakdown, details = (
                        await self.scorer.calculate_score(
                            candidate,
                            job_requirements
                        )
                    )

                    return {
                        "candidate_id": candidate_id,
                        "name": candidate.get("name", "Unknown"),
                        "total_score": float(total_score),
                        "score_breakdown": breakdown.to_dict(),
                        "matched_skills": details.get(
                            "matched_skills",
                            []
                        ),
                        "missing_skills": details.get(
                            "missing_skills",
                            []
                        ),
                        "strengths": details.get(
                            "strengths",
                            []
                        ),
                        "weaknesses": details.get(
                            "weaknesses",
                            []
                        ),
                        "candidate_data": candidate
                    }

                except Exception as e:

                    logger.exception(
                        f"Failed scoring candidate "
                        f"{candidate_id}: {str(e)}"
                    )

                    return None

        # Parallel scoring
        tasks = [
            process_candidate(candidate_id)
            for candidate_id in candidate_ids
        ]

        ranked_candidates = await asyncio.gather(*tasks)

        # Remove failed results
        ranked_candidates = [
            r for r in ranked_candidates
            if r is not None
        ]

        logger.info(
            f"Successfully scored "
            f"{len(ranked_candidates)} candidates"
        )

        if not ranked_candidates:
            return []

        # Normalize scores
        if self.normalize_scores:
            ranked_candidates = self._normalize_scores(
                ranked_candidates
            )

        # Sort descending
        ranked_candidates.sort(
            key=lambda x: x["total_score"],
            reverse=True
        )

        # Optional LLM reranking
        if (
            self.use_llm_reranking
            and self.llm_reranker
        ):

            try:

                rerank_limit = min(
                    25,
                    len(ranked_candidates)
                )

                reranked = await self.llm_reranker.rerank(
                    job_description,
                    ranked_candidates[:rerank_limit]
                )

                ranked_candidates[:rerank_limit] = reranked

                logger.info(
                    "LLM reranking completed successfully"
                )

            except Exception as e:

                logger.warning(
                    f"LLM reranking failed: {str(e)}"
                )

        # Final sorting after reranking
        ranked_candidates.sort(
            key=lambda x: x["total_score"],
            reverse=True
        )

        # Assign ranks
        for idx, candidate in enumerate(
            ranked_candidates,
            start=1
        ):
            candidate["rank"] = idx

        logger.info("Ranking pipeline completed")

        return ranked_candidates

    # =========================================================
    # SCORE NORMALIZATION
    # =========================================================

    def _normalize_scores(
        self,
        ranked_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Normalize scores between 0 and 1.
        """

        scores = [
            c["total_score"]
            for c in ranked_candidates
        ]

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return ranked_candidates

        for candidate in ranked_candidates:

            normalized = (
                candidate["total_score"] - min_score
            ) / (max_score - min_score)

            candidate["total_score"] = round(
                normalized,
                4
            )

        return ranked_candidates

    # =========================================================
    # DETAILED SCORE EXPLAINABILITY
    # =========================================================

    async def get_detailed_scores(
        self,
        job_description: str,
        candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get full explainability report for a candidate.
        """

        job_requirements = await self.jd_parser.parse(
            job_description
        )

        total_score, breakdown, details = (
            await self.scorer.calculate_score(
                candidate,
                job_requirements
            )
        )

        explanations = await self._generate_explanations(
            breakdown,
            details
        )

        feature_scores = {}

        for key, score in breakdown.to_dict().items():

            weight = self.scorer.weights.get(key, 0)

            feature_scores[key] = {
                "score": score,
                "weight": weight,
                "weighted_score": round(
                    score * weight,
                    4
                ),
                "explanation": explanations.get(
                    key,
                    ""
                )
            }

        return {
            "candidate_id": candidate.get(
                "candidate_id"
            ),
            "candidate_name": candidate.get(
                "name"
            ),
            "total_score": round(total_score, 4),
            "features": feature_scores,
            "matched_skills": details.get(
                "matched_skills",
                []
            ),
            "missing_skills": details.get(
                "missing_skills",
                []
            ),
            "strengths": await self._identify_strengths(
                breakdown,
                details
            ),
            "weaknesses": await self._identify_weaknesses(
                breakdown,
                details
            )
        }

    # =========================================================
    # EXPLANATIONS
    # =========================================================

    async def _generate_explanations(
        self,
        breakdown: ScoreBreakdown,
        details: Dict[str, Any]
    ) -> Dict[str, str]:

        explanations = {}

        matched = len(
            details.get("matched_skills", [])
        )

        missing = len(
            details.get("missing_skills", [])
        )

        explanations["skill_match"] = (
            f"Matched {matched} skills"
        )

        if missing:
            explanations["skill_match"] += (
                f" | Missing {missing} skills"
            )

        explanations["experience"] = (
            "Experience alignment based on role "
            "seniority and years of experience."
        )

        explanations["recruiter_signal"] = (
            "Measures recruiter engagement, "
            "profile completeness, and activity."
        )

        explanations["availability"] = (
            "Based on candidate notice period."
        )

        explanations["salary_fit"] = (
            "Measures salary expectation alignment."
        )

        return explanations

    # =========================================================
    # STRENGTHS
    # =========================================================

    async def _identify_strengths(
        self,
        breakdown: ScoreBreakdown,
        details: Dict[str, Any]
    ) -> List[str]:

        strengths = []

        if breakdown.skill_match >= 0.75:
            strengths.append(
                "Excellent skill alignment"
            )

        if breakdown.experience >= 0.75:
            strengths.append(
                "Strong experience fit"
            )

        if breakdown.recruiter_signal >= 0.70:
            strengths.append(
                "High recruiter confidence signals"
            )

        if breakdown.availability >= 0.80:
            strengths.append(
                "Can join quickly"
            )

        if len(details.get("matched_skills", [])) >= 5:
            strengths.append(
                "Broad technical skill coverage"
            )

        return strengths

    # =========================================================
    # WEAKNESSES
    # =========================================================

    async def _identify_weaknesses(
        self,
        breakdown: ScoreBreakdown,
        details: Dict[str, Any]
    ) -> List[str]:

        weaknesses = []

        if breakdown.skill_match < 0.50:
            weaknesses.append(
                "Insufficient required skill match"
            )

        if breakdown.experience < 0.40:
            weaknesses.append(
                "Experience mismatch"
            )

        if breakdown.salary_fit < 0.40:
            weaknesses.append(
                "Salary expectations exceed range"
            )

        if breakdown.availability < 0.40:
            weaknesses.append(
                "Long notice period"
            )

        missing = details.get(
            "missing_skills",
            []
        )

        if missing:
            weaknesses.append(
                f"Missing skills: "
                f"{', '.join(missing[:3])}"
            )

        return weaknesses
