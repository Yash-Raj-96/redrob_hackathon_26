"""
Generate human-readable explanations for candidate rankings
Enhanced production-ready implementation
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path

from backend.llm.llm_client import LLMClient
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class ExplanationGenerator:
    """
    Generate natural language explanations for candidate rankings.
    Supports:
    - LLM-based explanations
    - Deterministic fallback explanations
    - Recruiter-friendly summaries
    - Improvement suggestions
    """

    DEFAULT_PROMPT = """
You are an expert AI recruiting assistant.

Explain why the following candidate received their ranking score.

Candidate Name: {candidate_name}
Overall Match Score: {total_score}%

Feature Scores:
- Skill Match: {skill_score}%
- Experience Match: {exp_score}%
- Recruiter Signal: {signal_score}%
- Availability: {avail_score}%
- Salary Fit: {salary_score}%

Matched Skills:
{matched_skills}

Missing Skills:
{missing_skills}

Instructions:
1. Write a concise recruiter-friendly explanation.
2. Highlight strengths first.
3. Mention important skill gaps.
4. Explain hiring risks if any.
5. End with a recommendation.
6. Keep response under 180 words.
"""

    def __init__(self) -> None:
        self.llm = LLMClient()

    # =========================================================
    # PUBLIC METHODS
    # =========================================================

    async def generate_explanation(
        self,
        candidate: Dict[str, Any],
        job_description: str,
        score_breakdown: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable explanation.

        Priority:
        1. LLM explanation
        2. Template-based explanation fallback
        """

        try:
            explanation = await self._generate_llm_explanation(
                candidate=candidate,
                job_description=job_description,
                score_breakdown=score_breakdown
            )

            if explanation:
                return explanation.strip()

        except Exception as e:
            logger.warning(
                f"LLM explanation generation failed: {str(e)}"
            )

        logger.info("Using fallback template explanation")

        return self._generate_template_explanation(
            candidate=candidate,
            score_breakdown=score_breakdown
        )

    async def generate_short_summary(
        self,
        candidate: Dict[str, Any],
        score_breakdown: Dict[str, Any]
    ) -> str:
        """
        Generate concise one-line recruiter summary.
        """

        total_score = int(
            score_breakdown.get("total_score", 0) * 100
        )

        matched_skills = score_breakdown.get(
            "matched_skills",
            []
        )

        years = candidate.get(
            "years_experience",
            candidate.get(
                "candidate_data",
                {}
            ).get("years_experience", 0)
        )

        return (
            f"{candidate.get('name', 'Candidate')} is a "
            f"{total_score}% match with "
            f"{years} years of experience and expertise in "
            f"{', '.join(matched_skills[:3]) if matched_skills else 'relevant technologies'}."
        )

    def generate_improvement_suggestions(
        self,
        score_breakdown: Dict[str, Any]
    ) -> List[str]:
        """
        Generate actionable candidate improvement suggestions.
        """

        suggestions: List[str] = []

        missing_skills = score_breakdown.get(
            "missing_skills",
            []
        )

        if missing_skills:
            suggestions.append(
                f"Develop experience in {', '.join(missing_skills[:3])}"
            )

        features = score_breakdown.get("features", {})

        exp_score = (
            features.get("experience", {})
            .get("score", 1.0)
        )

        if exp_score < 0.5:
            suggestions.append(
                "Highlight more role-relevant projects and achievements"
            )

        signal_score = (
            features.get("recruiter_signal", {})
            .get("score", 1.0)
        )

        if signal_score < 0.5:
            suggestions.append(
                "Improve LinkedIn/GitHub/profile completeness"
            )

        avail_score = (
            features.get("availability", {})
            .get("score", 1.0)
        )

        if avail_score < 0.5:
            suggestions.append(
                "Consider reducing notice period if possible"
            )

        return suggestions[:5]

    # =========================================================
    # INTERNAL METHODS
    # =========================================================

    async def _generate_llm_explanation(
        self,
        candidate: Dict[str, Any],
        job_description: str,
        score_breakdown: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate explanation using LLM.
        """

        prompt_template = self._load_prompt_template()

        features = score_breakdown.get("features", {})

        prompt = prompt_template.format(
            candidate_name=candidate.get(
                "name",
                "Candidate"
            ),
            total_score=int(
                score_breakdown.get(
                    "total_score",
                    0
                ) * 100
            ),
            skill_score=int(
                features.get(
                    "skill_match",
                    {}
                ).get("score", 0) * 100
            ),
            exp_score=int(
                features.get(
                    "experience",
                    {}
                ).get("score", 0) * 100
            ),
            signal_score=int(
                features.get(
                    "recruiter_signal",
                    {}
                ).get("score", 0) * 100
            ),
            avail_score=int(
                features.get(
                    "availability",
                    {}
                ).get("score", 0) * 100
            ),
            salary_score=int(
                features.get(
                    "salary_fit",
                    {}
                ).get("score", 0) * 100
            ),
            matched_skills=", ".join(
                score_breakdown.get(
                    "matched_skills",
                    []
                )[:8]
            ) or "None",
            missing_skills=", ".join(
                score_breakdown.get(
                    "missing_skills",
                    []
                )[:5]
            ) or "None"
        )

        logger.info(
            f"Generating LLM explanation for "
            f"{candidate.get('name', 'Candidate')}"
        )

        response = await self.llm.generate(
            prompt=prompt,
            max_tokens=220,
            temperature=0.4
        )

        return response

    def _generate_template_explanation(
        self,
        candidate: Dict[str, Any],
        score_breakdown: Dict[str, Any]
    ) -> str:
        """
        Deterministic fallback explanation.
        """

        features = score_breakdown.get("features", {})

        strengths = []
        concerns = []

        # =====================================================
        # Skill Match
        # =====================================================

        skill_score = (
            features.get("skill_match", {})
            .get("score", 0)
        )

        if skill_score >= 0.8:
            strengths.append(
                "excellent alignment with required technical skills"
            )
        elif skill_score >= 0.5:
            strengths.append(
                "moderate alignment with required skills"
            )
        else:
            concerns.append(
                "significant technical skill gaps"
            )

        # =====================================================
        # Experience
        # =====================================================

        exp_score = (
            features.get("experience", {})
            .get("score", 0)
        )

        if exp_score >= 0.75:
            strengths.append(
                "strong relevant experience"
            )
        elif exp_score < 0.4:
            concerns.append(
                "experience mismatch for the role"
            )

        # =====================================================
        # Recruiter Signal
        # =====================================================

        signal_score = (
            features.get("recruiter_signal", {})
            .get("score", 0)
        )

        if signal_score >= 0.7:
            strengths.append(
                "strong recruiter/platform engagement"
            )

        # =====================================================
        # Availability
        # =====================================================

        avail_score = (
            features.get("availability", {})
            .get("score", 0)
        )

        if avail_score >= 0.7:
            strengths.append(
                "good joining availability"
            )

        # =====================================================
        # Build Narrative
        # =====================================================

        candidate_name = candidate.get(
            "name",
            "This candidate"
        )

        explanation = (
            f"{candidate_name} demonstrates "
        )

        if strengths:
            explanation += ", ".join(strengths)

        if concerns:
            explanation += (
                ". However, there are concerns regarding "
                + ", ".join(concerns)
            )

        missing_skills = score_breakdown.get(
            "missing_skills",
            []
        )

        if missing_skills:
            explanation += (
                f". Key missing skills include "
                f"{', '.join(missing_skills[:3])}"
            )

        total_score = int(
            score_breakdown.get(
                "total_score",
                0
            ) * 100
        )

        # Hiring recommendation
        if total_score >= 80:
            recommendation = (
                "Recommended for immediate shortlist."
            )
        elif total_score >= 60:
            recommendation = (
                "Worth considering after further evaluation."
            )
        else:
            recommendation = (
                "May require significant upskilling for this role."
            )

        explanation += f". {recommendation}"

        return explanation

    def _load_prompt_template(self) -> str:
        """
        Load prompt template from file if available.
        """

        prompt_path = Path(
            "backend/llm/prompts/explain_prompt.txt"
        )

        try:
            if prompt_path.exists():
                return prompt_path.read_text(
                    encoding="utf-8"
                )

        except Exception as e:
            logger.warning(
                f"Failed to load prompt template: {str(e)}"
            )

        return self.DEFAULT_PROMPT
