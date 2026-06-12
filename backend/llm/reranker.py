
"""
LLM-based reranking of top candidates
Production-grade implementation
"""

from typing import List, Dict, Any, Optional
import asyncio
import json
import re

from backend.llm.llm_client import LLMClient
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class LLMReranker:
    """
    LLM-powered reranker for top candidates.

    Features:
    - Intelligent semantic reranking
    - Batched evaluation
    - Bias-aware prompts
    - Robust score extraction
    - Configurable blending
    - Explainability metadata
    """

    DEFAULT_TOP_K = 20

    def __init__(
        self,
        blend_weight_original: float = 0.7,
        blend_weight_llm: float = 0.3,
        top_k: int = DEFAULT_TOP_K
    ):

        self.llm = LLMClient()

        self.blend_weight_original = blend_weight_original
        self.blend_weight_llm = blend_weight_llm

        self.top_k = top_k

    # =========================================================
    # Public API
    # =========================================================

    async def rerank(
        self,
        job_description: str,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank top candidates using LLM intelligence.
        """

        logger.info(
            f"Starting LLM reranking for {len(candidates)} candidates"
        )

        if not candidates:
            return []

        candidates_to_rerank = candidates[: self.top_k]

        try:

            scored_candidates = await asyncio.gather(
                *[
                    self._score_candidate(
                        job_description,
                        candidate
                    )
                    for candidate in candidates_to_rerank
                ]
            )

            reranked = []

            for candidate, llm_result in zip(
                candidates_to_rerank,
                scored_candidates
            ):

                llm_score = llm_result["score"]

                original_score = candidate.get(
                    "total_score",
                    0.0
                )

                blended_score = (
                    self.blend_weight_original * original_score
                    + self.blend_weight_llm * llm_score
                )

                candidate["original_score"] = original_score
                candidate["llm_score"] = llm_score
                candidate["llm_reasoning"] = llm_result.get(
                    "reasoning",
                    ""
                )

                candidate["total_score"] = round(
                    blended_score,
                    4
                )

                reranked.append(candidate)

            reranked.sort(
                key=lambda x: x["total_score"],
                reverse=True
            )

            # Update ranks
            for idx, candidate in enumerate(reranked, start=1):
                candidate["rank"] = idx

            logger.info("LLM reranking completed successfully")

            # Append untouched candidates
            if len(candidates) > self.top_k:
                reranked.extend(candidates[self.top_k:])

            return reranked

        except Exception as e:

            logger.exception("LLM reranking failed")

            return candidates

    # =========================================================
    # Candidate Scoring
    # =========================================================

    async def _score_candidate(
        self,
        job_description: str,
        candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score candidate using LLM.
        """

        try:

            prompt = self._build_prompt(
                job_description,
                candidate
            )

            response = await self.llm.generate(
                prompt=prompt,
                max_tokens=120,
                temperature=0.2
            )

            parsed = self._parse_llm_response(response)

            logger.debug(
                f"LLM score for "
                f"{candidate.get('name')}: "
                f"{parsed['score']}"
            )

            return parsed

        except Exception as e:

            logger.error(
                f"Failed to score candidate "
                f"{candidate.get('candidate_id')}: {str(e)}"
            )

            return {
                "score": candidate.get("total_score", 0.0),
                "reasoning": "Fallback to original ranking score"
            }

    # =========================================================
    # Prompt Engineering
    # =========================================================

    def _build_prompt(
        self,
        job_description: str,
        candidate: Dict[str, Any]
    ) -> str:
        """
        Construct reranking prompt.
        """

        candidate_data = candidate.get(
            "candidate_data",
            {}
        )

        matched_skills = candidate.get(
            "matched_skills",
            []
        )[:10]

        missing_skills = candidate.get(
            "missing_skills",
            []
        )[:5]

        return f"""
You are an expert AI recruiting evaluator.

Your task:
Evaluate how well this candidate fits the role.

IMPORTANT RULES:
- Focus ONLY on job relevance
- Ignore gender, ethnicity, age, nationality
- Prioritize skills, experience, and role alignment
- Score from 0.0 to 1.0

Job Description:
{job_description[:3000]}

Candidate Profile:
Name: {candidate.get('name', 'Unknown')}

Current Role:
{candidate_data.get('current_role', 'N/A')}

Experience:
{candidate_data.get('years_experience', 0)} years

Location:
{candidate_data.get('location', 'N/A')}

Matched Skills:
{', '.join(matched_skills)}

Missing Skills:
{', '.join(missing_skills)}

Current System Score:
{candidate.get('total_score', 0):.2f}

Respond ONLY in JSON format:

{{
  "score": 0.85,
  "reasoning": "Excellent backend and cloud alignment..."
}}
"""

    # =========================================================
    # Response Parsing
    # =========================================================

    def _parse_llm_response(
        self,
        response: str
    ) -> Dict[str, Any]:
        """
        Parse structured LLM response safely.
        """

        try:

            json_match = re.search(
                r'\{.*\}',
                response,
                re.DOTALL
            )

            if json_match:

                parsed = json.loads(
                    json_match.group(0)
                )

                score = float(
                    parsed.get("score", 0.5)
                )

                score = max(0.0, min(score, 1.0))

                return {
                    "score": score,
                    "reasoning": parsed.get(
                        "reasoning",
                        ""
                    )
                }

        except Exception as e:

            logger.warning(
                f"JSON parse failed: {str(e)}"
            )

        # ---------------------------------------------
        # Fallback numeric extraction
        # ---------------------------------------------

        numbers = re.findall(
            r"0?\.\d+|\d+",
            response
        )

        if numbers:

            try:

                value = float(numbers[0])

                # Convert 0-100 → 0-1
                if value > 1:
                    value /= 100

                value = max(0.0, min(value, 1.0))

                return {
                    "score": value,
                    "reasoning": "Parsed numeric fallback"
                }

            except Exception:
                pass

        logger.warning("Could not parse LLM score")

        return {
            "score": 0.5,
            "reasoning": "Default fallback score"
        }

    # =========================================================
    # Diagnostics
    # =========================================================

    def get_config(self) -> Dict[str, Any]:
        """
        Return reranker configuration.
        """

        return {
            "blend_weight_original": self.blend_weight_original,
            "blend_weight_llm": self.blend_weight_llm,
            "top_k": self.top_k
        }
