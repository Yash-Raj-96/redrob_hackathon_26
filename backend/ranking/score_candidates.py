"""
Core scoring logic for candidates
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

import numpy as np

from backend.app.constants import (
    SCORING_WEIGHTS,
    REQUIRED_SKILLS_THRESHOLD
)

from backend.ranking.skill_matcher import SkillMatcher
from backend.ranking.experience_score import ExperienceScorer
from backend.ranking.recruiter_signal_score import RecruiterSignalScorer
from backend.ranking.salary_fit import SalaryFitScorer
from backend.ranking.availability_score import AvailabilityScorer

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


# =========================================================
# Score Breakdown Model
# =========================================================

@dataclass
class ScoreBreakdown:
    """
    Detailed score breakdown for explainability
    """

    skill_match: float = 0.0
    experience: float = 0.0
    recruiter_signal: float = 0.0
    availability: float = 0.0
    salary_fit: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """
        Convert score breakdown to dictionary
        """
        return asdict(self)


# =========================================================
# Candidate Scorer
# =========================================================

class CandidateScorer:
    """
    Calculate scores for candidates using multiple ranking signals
    """

    def __init__(self, weights: Dict[str, float] = None):

        self.weights = weights or SCORING_WEIGHTS

        self.skill_matcher = SkillMatcher()
        self.exp_scorer = ExperienceScorer()
        self.signal_scorer = RecruiterSignalScorer()
        self.salary_scorer = SalaryFitScorer()
        self.avail_scorer = AvailabilityScorer()

        logger.info("CandidateScorer initialized")

    # =====================================================
    # Main Score Calculation
    # =====================================================

    async def calculate_score(
        self,
        candidate: Dict[str, Any],
        job_requirements: Dict[str, Any]
    ) -> Tuple[float, ScoreBreakdown, Dict[str, Any]]:
        """
        Calculate comprehensive candidate score
        """

        try:

            # =================================================
            # Extract candidate info safely
            # =================================================

            candidate_skills = candidate.get("skills", [])
            candidate_exp = candidate.get("years_experience", 0)
            candidate_role = candidate.get("current_role", "")
            notice_period = candidate.get("notice_period_days", 30)
            salary_expectation = candidate.get("salary_expectation", 0)

            required_skills = job_requirements.get(
                "required_skills",
                []
            )

            preferred_skills = job_requirements.get(
                "preferred_skills",
                []
            )

            salary_range = job_requirements.get(
                "salary_range",
                [0, float("inf")]
            )

            # =================================================
            # Skill Matching Score
            # =================================================

            (
                skill_score,
                matched_skills,
                missing_skills
            ) = await self.skill_matcher.calculate_score(
                candidate_skills,
                required_skills,
                preferred_skills
            )

            # =================================================
            # Experience Score
            # =================================================

            exp_score = await self.exp_scorer.calculate_score(
                candidate_exp,
                candidate_role,
                job_requirements
            )

            # =================================================
            # Recruiter Signal Score
            # =================================================

            signal_score = await self.signal_scorer.calculate_score(
                candidate
            )

            # =================================================
            # Availability Score
            # =================================================

            avail_score = self.avail_scorer.calculate_score(
                notice_period
            )

            # =================================================
            # Salary Fit Score
            # =================================================

            salary_score = await self.salary_scorer.calculate_score(
                salary_expectation,
                salary_range
            )

            # =================================================
            # Required Skills Penalty
            # =================================================

            if required_skills:

                required_matches = len(
                    [
                        skill for skill in matched_skills
                        if skill in required_skills
                    ]
                )

                required_match_ratio = (
                    required_matches / len(required_skills)
                )

                if required_match_ratio < REQUIRED_SKILLS_THRESHOLD:

                    logger.info(
                        f"Applying penalty to candidate "
                        f"{candidate.get('candidate_id', 'unknown')} "
                        f"(required match ratio="
                        f"{required_match_ratio:.2f})"
                    )

                    skill_score *= 0.7

            # =================================================
            # Weighted Final Score
            # =================================================

            total_score = (
                self.weights.get("skill_match", 0) * skill_score +
                self.weights.get("experience", 0) * exp_score +
                self.weights.get("recruiter_signal", 0) * signal_score +
                self.weights.get("availability", 0) * avail_score +
                self.weights.get("salary_fit", 0) * salary_score
            )

            # Ensure score stays within bounds
            total_score = max(0.0, min(1.0, total_score))

            # =================================================
            # Breakdown Object
            # =================================================

            breakdown = ScoreBreakdown(
                skill_match=round(skill_score, 4),
                experience=round(exp_score, 4),
                recruiter_signal=round(signal_score, 4),
                availability=round(avail_score, 4),
                salary_fit=round(salary_score, 4)
            )

            # =================================================
            # Explainability Details
            # =================================================

            details = {
                "candidate_id": candidate.get("candidate_id"),

                "total_score": round(total_score, 4),

                "features": {

                    "skill_match": {
                        "score": round(skill_score, 4),
                        "weight": self.weights.get(
                            "skill_match",
                            0
                        ),
                        "matched": matched_skills,
                        "missing": missing_skills
                    },

                    "experience": {
                        "score": round(exp_score, 4),
                        "weight": self.weights.get(
                            "experience",
                            0
                        )
                    },

                    "recruiter_signal": {
                        "score": round(signal_score, 4),
                        "weight": self.weights.get(
                            "recruiter_signal",
                            0
                        )
                    },

                    "availability": {
                        "score": round(avail_score, 4),
                        "weight": self.weights.get(
                            "availability",
                            0
                        )
                    },

                    "salary_fit": {
                        "score": round(salary_score, 4),
                        "weight": self.weights.get(
                            "salary_fit",
                            0
                        )
                    }
                },

                "matched_skills": matched_skills,
                "missing_skills": missing_skills,

                "metadata": {
                    "required_skills_count": len(required_skills),
                    "matched_required_skills": len(
                        [
                            s for s in matched_skills
                            if s in required_skills
                        ]
                    ),
                    "candidate_experience": candidate_exp,
                    "notice_period_days": notice_period
                }
            }

            logger.debug(
                f"Candidate scored successfully: "
                f"{candidate.get('candidate_id')} "
                f"-> {total_score:.4f}"
            )

            return total_score, breakdown, details

        except Exception as e:

            logger.exception(
                f"Failed scoring candidate "
                f"{candidate.get('candidate_id', 'unknown')}"
            )

            # Fail gracefully
            empty_breakdown = ScoreBreakdown()

            return 0.0, empty_breakdown, {
                "error": str(e),
                "matched_skills": [],
                "missing_skills": []
            }