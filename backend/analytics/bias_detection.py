"""
Advanced Bias Detection Engine
Enterprise-grade fairness analytics for candidate ranking
"""

from typing import Dict, Any, List, Optional
from collections import Counter

import numpy as np
import pandas as pd

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class BiasDetector:
    """
    Detect and analyze ranking bias across candidate populations.

    Supported fairness checks:
    - Location bias
    - Experience bias
    - Education pedigree bias
    - Gender representation
    - Diversity concentration
    - Representation imbalance
    """

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    HIGH_BIAS_THRESHOLD = 0.65
    MODERATE_BIAS_THRESHOLD = 0.35

    LOCATION_DOMINANCE_HIGH = 0.50
    LOCATION_DOMINANCE_MODERATE = 0.30

    EXPERIENCE_DOMINANCE_HIGH = 0.60
    EXPERIENCE_DOMINANCE_MODERATE = 0.40

    EDUCATION_DOMINANCE_HIGH = 0.40
    EDUCATION_DOMINANCE_MODERATE = 0.20

    TIER1_KEYWORDS = [
        "iit",
        "nit",
        "iiit",
        "bits",
        "iim",
        "isb"
    ]

    # ---------------------------------------------------------
    # Public Analysis API
    # ---------------------------------------------------------

    async def analyze(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run comprehensive bias analysis.
        """

        logger.info("Starting bias analysis")

        if df.empty:
            logger.warning("Empty dataframe received")

            return self._empty_response()

        location_bias = await self._check_location_bias(df)
        experience_bias = await self._check_experience_bias(df)
        education_bias = await self._check_education_bias(df)
        gender_balance = await self._check_gender_balance(df)

        diversity_metrics = await self._calculate_diversity_metrics(df)

        component_scores = [
            location_bias["score"],
            experience_bias["score"],
            education_bias["score"]
        ]

        overall_bias_score = round(
            float(np.mean(component_scores)),
            4
        )

        fairness_rating = self._get_fairness_rating(
            overall_bias_score
        )

        results = {
            "bias_score": overall_bias_score,
            "fairness_rating": fairness_rating,
            "location_bias": location_bias,
            "experience_bias": experience_bias,
            "education_bias": education_bias,
            "gender_balance": gender_balance,
            "diversity_metrics": diversity_metrics,
            "risk_level": self._calculate_risk_level(
                overall_bias_score
            ),
            "recommendations": self._generate_recommendations(
                location_bias,
                experience_bias,
                education_bias,
                gender_balance
            )
        }

        logger.info(
            f"Bias analysis completed | "
            f"Overall score: {overall_bias_score}"
        )

        return results

    # ---------------------------------------------------------
    # Location Bias
    # ---------------------------------------------------------

    async def _check_location_bias(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        if "location" not in df.columns:
            return self._missing_column_response(
                "location"
            )

        location_series = (
            df["location"]
            .fillna("Unknown")
            .astype(str)
        )

        distribution = (
            location_series
            .value_counts(normalize=True)
        )

        if distribution.empty:
            return self._neutral_metric(
                "No location data available"
            )

        dominant_location = distribution.index[0]
        dominant_pct = float(distribution.iloc[0])

        score, severity = self._calculate_bias_level(
            dominant_pct,
            self.LOCATION_DOMINANCE_HIGH,
            self.LOCATION_DOMINANCE_MODERATE
        )

        return {
            "score": score,
            "severity": severity,
            "dominant_group": dominant_location,
            "dominant_percentage": round(
                dominant_pct,
                4
            ),
            "group_count": int(len(distribution)),
            "distribution": (
                distribution.head(10)
                .round(4)
                .to_dict()
            ),
            "message": (
                f"{dominant_location} represents "
                f"{dominant_pct:.1%} of candidates"
            )
        }

    # ---------------------------------------------------------
    # Experience Bias
    # ---------------------------------------------------------

    async def _check_experience_bias(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        if "years_experience" not in df.columns:
            return self._missing_column_response(
                "years_experience"
            )

        exp_bins = [0, 2, 5, 8, 12, 100]

        exp_labels = [
            "0-2",
            "2-5",
            "5-8",
            "8-12",
            "12+"
        ]

        experience_distribution = (
            pd.cut(
                df["years_experience"],
                bins=exp_bins,
                labels=exp_labels
            )
            .value_counts(normalize=True)
        )

        if experience_distribution.empty:
            return self._neutral_metric(
                "No experience data available"
            )

        dominant_group = (
            experience_distribution.index[0]
        )

        dominant_pct = float(
            experience_distribution.iloc[0]
        )

        score, severity = self._calculate_bias_level(
            dominant_pct,
            self.EXPERIENCE_DOMINANCE_HIGH,
            self.EXPERIENCE_DOMINANCE_MODERATE
        )

        return {
            "score": score,
            "severity": severity,
            "dominant_group": str(dominant_group),
            "dominant_percentage": round(
                dominant_pct,
                4
            ),
            "distribution": (
                experience_distribution
                .round(4)
                .to_dict()
            ),
            "message": (
                f"Highest concentration in "
                f"{dominant_group} years bucket"
            )
        }

    # ---------------------------------------------------------
    # Education Bias
    # ---------------------------------------------------------

    async def _check_education_bias(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        if "education" not in df.columns:
            return self._missing_column_response(
                "education"
            )

        education_series = (
            df["education"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        tier1_mask = education_series.apply(
            self._is_tier1_institution
        )

        tier1_percentage = float(
            tier1_mask.mean()
        )

        score, severity = self._calculate_bias_level(
            tier1_percentage,
            self.EDUCATION_DOMINANCE_HIGH,
            self.EDUCATION_DOMINANCE_MODERATE
        )

        return {
            "score": score,
            "severity": severity,
            "tier1_percentage": round(
                tier1_percentage,
                4
            ),
            "message": (
                f"{tier1_percentage:.1%} of candidates "
                f"come from Tier-1 institutions"
            )
        }

    # ---------------------------------------------------------
    # Gender Representation
    # ---------------------------------------------------------

    async def _check_gender_balance(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        if "gender" not in df.columns:
            return {
                "available": False,
                "message": "Gender data unavailable"
            }

        gender_distribution = (
            df["gender"]
            .fillna("Unknown")
            .value_counts(normalize=True)
        )

        if len(gender_distribution) <= 1:

            imbalance_ratio = 1.0

        else:

            imbalance_ratio = (
                gender_distribution.max()
                / gender_distribution.min()
            )

        balance_score = round(
            min(imbalance_ratio / 3.0, 1.0),
            4
        )

        return {
            "available": True,
            "balance_score": balance_score,
            "distribution": (
                gender_distribution
                .round(4)
                .to_dict()
            ),
            "message": (
                "Gender representation analyzed"
            )
        }

    # ---------------------------------------------------------
    # Diversity Metrics
    # ---------------------------------------------------------

    async def _calculate_diversity_metrics(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        metrics = {}

        if "location" in df.columns:

            location_counts = (
                df["location"]
                .fillna("Unknown")
                .value_counts()
            )

            metrics["location_entropy"] = round(
                self._shannon_entropy(
                    location_counts
                ),
                4
            )

        if "education" in df.columns:

            metrics["unique_education_entries"] = int(
                df["education"]
                .nunique()
            )

        metrics["candidate_count"] = int(len(df))

        return metrics

    # ---------------------------------------------------------
    # Report Generation
    # ---------------------------------------------------------

    async def generate_report(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:

        metrics = await self.analyze(df)

        report = {
            "summary": {
                "overall_bias_score": metrics[
                    "bias_score"
                ],
                "risk_level": metrics[
                    "risk_level"
                ],
                "fairness_rating": metrics[
                    "fairness_rating"
                ]
            },
            "metrics": metrics,
            "mitigation_strategies": (
                self._generate_mitigation_strategies(
                    metrics
                )
            ),
            "audit_timestamp": (
                pd.Timestamp.utcnow()
                .isoformat()
            )
        }

        return report

    # ---------------------------------------------------------
    # Recommendation Engine
    # ---------------------------------------------------------

    def _generate_recommendations(
        self,
        location_bias: Dict,
        experience_bias: Dict,
        education_bias: Dict,
        gender_balance: Dict
    ) -> List[str]:

        recommendations = []

        if location_bias["score"] > 0.5:
            recommendations.append(
                "Expand sourcing to additional geographic regions"
            )

        if experience_bias["score"] > 0.5:
            recommendations.append(
                "Reduce over-filtering by experience thresholds"
            )

        if education_bias["score"] > 0.5:
            recommendations.append(
                "Prioritize skills-based evaluation over institutional pedigree"
            )

        if gender_balance.get(
            "balance_score",
            0
        ) > 0.7:

            recommendations.append(
                "Review sourcing channels for gender diversity"
            )

        if not recommendations:
            recommendations.append(
                "Current diversity metrics are within acceptable thresholds"
            )

        return recommendations

    # ---------------------------------------------------------
    # Mitigation Strategies
    # ---------------------------------------------------------

    def _generate_mitigation_strategies(
        self,
        metrics: Dict[str, Any]
    ) -> List[str]:

        return [
            "Use blind resume screening where possible",
            "Balance sourcing across multiple regions",
            "Reduce pedigree-heavy filtering",
            "Continuously audit ranking outputs",
            "Track fairness metrics over time",
            "Incorporate skill-based assessments"
        ]

    # ---------------------------------------------------------
    # Utility Functions
    # ---------------------------------------------------------

    def _is_tier1_institution(
        self,
        education_text: str
    ) -> bool:

        return any(
            keyword in education_text
            for keyword in self.TIER1_KEYWORDS
        )

    def _calculate_bias_level(
        self,
        percentage: float,
        high_threshold: float,
        moderate_threshold: float
    ):

        if percentage >= high_threshold:
            return 0.8, "High"

        if percentage >= moderate_threshold:
            return 0.4, "Moderate"

        return 0.1, "Low"

    def _calculate_risk_level(
        self,
        score: float
    ) -> str:

        if score >= self.HIGH_BIAS_THRESHOLD:
            return "High"

        if score >= self.MODERATE_BIAS_THRESHOLD:
            return "Moderate"

        return "Low"

    def _get_fairness_rating(
        self,
        score: float
    ) -> str:

        if score < 0.25:
            return "Fair"

        if score < 0.50:
            return "Needs Monitoring"

        return "High Bias Risk"

    def _shannon_entropy(
        self,
        counts: pd.Series
    ) -> float:

        probabilities = (
            counts / counts.sum()
        )

        return float(
            -np.sum(
                probabilities
                * np.log2(probabilities)
            )
        )

    def _neutral_metric(
        self,
        message: str
    ) -> Dict[str, Any]:

        return {
            "score": 0.0,
            "severity": "None",
            "message": message
        }

    def _missing_column_response(
        self,
        column_name: str
    ) -> Dict[str, Any]:

        return {
            "score": 0.0,
            "severity": "Unknown",
            "message": (
                f"Column '{column_name}' not available"
            )
        }

    def _empty_response(self) -> Dict[str, Any]:

        return {
            "bias_score": 0.0,
            "fairness_rating": "Unknown",
            "risk_level": "Unknown",
            "message": "No candidate data available"
        }
