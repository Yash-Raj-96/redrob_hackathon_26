"""
Metrics for evaluating ranking quality
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class RankingMetrics:
    """
    Calculate ranking evaluation metrics for candidate retrieval/ranking systems.
    """

    # =========================================================
    # Core IR Metrics
    # =========================================================

    def calculate_ndcg(
        self,
        relevance_scores: List[float],
        k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).

        Measures ranking quality with position-aware relevance weighting.
        """

        if not relevance_scores:
            return 0.0

        relevance_scores = relevance_scores[:k]

        dcg = self._calculate_dcg(relevance_scores)

        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = self._calculate_dcg(ideal_scores)

        return round(dcg / idcg, 4) if idcg > 0 else 0.0

    def calculate_mrr(
        self,
        relevant_ranks: List[int]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        Focuses on the first relevant result.
        """

        if not relevant_ranks:
            return 0.0

        reciprocal_ranks = [
            1 / rank
            for rank in relevant_ranks
            if rank > 0
        ]

        if not reciprocal_ranks:
            return 0.0

        return round(float(np.mean(reciprocal_ranks)), 4)

    def calculate_precision_at_k(
        self,
        relevance_scores: List[int],
        k: int = 10
    ) -> float:
        """
        Calculate Precision@K.
        """

        if not relevance_scores or k <= 0:
            return 0.0

        top_k = relevance_scores[:k]

        precision = sum(top_k) / k

        return round(float(precision), 4)

    def calculate_recall_at_k(
        self,
        relevance_scores: List[int],
        total_relevant: int,
        k: int = 10
    ) -> float:
        """
        Calculate Recall@K.
        """

        if total_relevant <= 0:
            return 0.0

        retrieved_relevant = sum(relevance_scores[:k])

        recall = retrieved_relevant / total_relevant

        return round(float(recall), 4)

    def calculate_f1_at_k(
        self,
        relevance_scores: List[int],
        total_relevant: int,
        k: int = 10
    ) -> float:
        """
        Calculate F1@K.
        """

        precision = self.calculate_precision_at_k(
            relevance_scores,
            k
        )

        recall = self.calculate_recall_at_k(
            relevance_scores,
            total_relevant,
            k
        )

        if precision + recall == 0:
            return 0.0

        f1 = 2 * ((precision * recall) / (precision + recall))

        return round(float(f1), 4)

    def calculate_hit_rate_at_k(
        self,
        relevance_scores: List[int],
        k: int = 10
    ) -> float:
        """
        Calculate Hit Rate@K.

        Returns 1 if at least one relevant item exists in top-k.
        """

        if not relevance_scores:
            return 0.0

        return 1.0 if any(relevance_scores[:k]) else 0.0

    def calculate_average_precision(
        self,
        relevance_scores: List[int]
    ) -> float:
        """
        Calculate Average Precision (AP).
        """

        if not relevance_scores:
            return 0.0

        precisions = []

        relevant_so_far = 0

        for idx, rel in enumerate(relevance_scores, start=1):

            if rel:

                relevant_so_far += 1

                precisions.append(
                    relevant_so_far / idx
                )

        if not precisions:
            return 0.0

        return round(float(np.mean(precisions)), 4)

    def calculate_map(
        self,
        query_ap_scores: List[float]
    ) -> float:
        """
        Calculate Mean Average Precision (MAP).
        """

        if not query_ap_scores:
            return 0.0

        return round(float(np.mean(query_ap_scores)), 4)

    # =========================================================
    # System-Level Evaluation
    # =========================================================

    async def evaluate_ranking(
        self,
        ranked_candidates: List[Dict[str, Any]],
        ground_truth: List[str],
        k_values: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate ranking against ground truth labels.
        """

        if k_values is None:
            k_values = [5, 10, 20]

        metrics: Dict[str, Any] = {}

        try:

            if not ranked_candidates:
                logger.warning("No ranked candidates provided")
                return self._empty_metrics()

            if not ground_truth:
                logger.warning("No ground truth provided")
                return self._empty_metrics()

            # ======================================
            # Build relevance vector
            # ======================================

            relevance_scores = []
            relevant_ranks = []

            ground_truth_set = set(ground_truth)

            for idx, candidate in enumerate(ranked_candidates):

                candidate_id = candidate.get("candidate_id")

                is_relevant = int(
                    candidate_id in ground_truth_set
                )

                relevance_scores.append(is_relevant)

                if is_relevant:
                    relevant_ranks.append(idx + 1)

            # ======================================
            # Standard Metrics
            # ======================================

            for k in k_values:

                metrics[f"ndcg@{k}"] = self.calculate_ndcg(
                    relevance_scores,
                    k
                )

                metrics[f"precision@{k}"] = (
                    self.calculate_precision_at_k(
                        relevance_scores,
                        k
                    )
                )

                metrics[f"recall@{k}"] = (
                    self.calculate_recall_at_k(
                        relevance_scores,
                        len(ground_truth),
                        k
                    )
                )

                metrics[f"f1@{k}"] = (
                    self.calculate_f1_at_k(
                        relevance_scores,
                        len(ground_truth),
                        k
                    )
                )

                metrics[f"hit_rate@{k}"] = (
                    self.calculate_hit_rate_at_k(
                        relevance_scores,
                        k
                    )
                )

            # ======================================
            # Global Metrics
            # ======================================

            metrics["mrr"] = self.calculate_mrr(
                relevant_ranks
            )

            metrics["average_precision"] = (
                self.calculate_average_precision(
                    relevance_scores
                )
            )

            metrics["coverage"] = self.calculate_coverage(
                ranked_candidates,
                ground_truth
            )

            metrics["relevant_candidates_found"] = int(
                sum(relevance_scores)
            )

            metrics["total_ground_truth"] = len(
                ground_truth
            )

            metrics["ranking_size"] = len(
                ranked_candidates
            )

            # ======================================
            # Diagnostic Insights
            # ======================================

            metrics["ranking_quality"] = (
                self._assess_ranking_quality(metrics)
            )

            metrics["diagnostics"] = (
                self._generate_diagnostics(
                    metrics,
                    relevant_ranks
                )
            )

            logger.info(
                "Ranking evaluation completed successfully"
            )

            return metrics

        except Exception as e:

            logger.exception(
                "Ranking evaluation failed"
            )

            return {
                "error": str(e),
                **self._empty_metrics()
            }

    # =========================================================
    # Additional Metrics
    # =========================================================

    def calculate_coverage(
        self,
        ranked_candidates: List[Dict[str, Any]],
        ground_truth: List[str]
    ) -> float:
        """
        Calculate coverage of relevant candidates.
        """

        if not ground_truth:
            return 0.0

        retrieved_ids = {
            candidate.get("candidate_id")
            for candidate in ranked_candidates
        }

        matched = len(
            retrieved_ids.intersection(set(ground_truth))
        )

        return round(
            matched / len(ground_truth),
            4
        )

    # =========================================================
    # Internal Helpers
    # =========================================================

    def _calculate_dcg(
        self,
        scores: List[float]
    ) -> float:
        """
        Calculate Discounted Cumulative Gain.
        """

        dcg = 0.0

        for idx, score in enumerate(scores):
            dcg += (
                (2**score - 1)
                / np.log2(idx + 2)
            )

        return dcg

    def _assess_ranking_quality(
        self,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable quality assessment.
        """

        ndcg_10 = metrics.get("ndcg@10", 0)

        if ndcg_10 >= 0.9:
            return "Excellent"

        elif ndcg_10 >= 0.75:
            return "Very Good"

        elif ndcg_10 >= 0.6:
            return "Good"

        elif ndcg_10 >= 0.4:
            return "Moderate"

        return "Poor"

    def _generate_diagnostics(
        self,
        metrics: Dict[str, Any],
        relevant_ranks: List[int]
    ) -> List[str]:
        """
        Generate ranking diagnostics.
        """

        diagnostics = []

        if metrics.get("precision@10", 0) < 0.3:
            diagnostics.append(
                "Low precision in top results"
            )

        if metrics.get("recall@10", 0) < 0.5:
            diagnostics.append(
                "Many relevant candidates missing from top-10"
            )

        if metrics.get("mrr", 0) < 0.5:
            diagnostics.append(
                "Relevant candidates appear too low in ranking"
            )

        if not relevant_ranks:
            diagnostics.append(
                "No relevant candidates retrieved"
            )

        if not diagnostics:
            diagnostics.append(
                "Ranking performance looks healthy"
            )

        return diagnostics

    def _empty_metrics(self) -> Dict[str, Any]:
        """
        Empty fallback metrics response.
        """

        return {
            "ndcg@5": 0.0,
            "ndcg@10": 0.0,
            "ndcg@20": 0.0,
            "precision@5": 0.0,
            "precision@10": 0.0,
            "precision@20": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "recall@20": 0.0,
            "f1@5": 0.0,
            "f1@10": 0.0,
            "f1@20": 0.0,
            "hit_rate@5": 0.0,
            "hit_rate@10": 0.0,
            "hit_rate@20": 0.0,
            "mrr": 0.0,
            "average_precision": 0.0,
            "coverage": 0.0,
            "ranking_quality": "Unknown",
            "diagnostics": []
        }