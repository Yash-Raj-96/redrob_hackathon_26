"""
Ranking evaluation metrics
"""

from typing import List, Dict, Any
import numpy as np

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class RankingMetrics:
    """
    Evaluate ranking quality.
    """

    def calculate_ndcg(
        self,
        relevance_scores: List[float],
        k: int = 10
    ) -> float:

        if not relevance_scores:
            return 0.0

        dcg = self._dcg(relevance_scores[:k])

        ideal_scores = sorted(
            relevance_scores,
            reverse=True
        )[:k]

        idcg = self._dcg(ideal_scores)

        return round(dcg / idcg, 4) if idcg > 0 else 0.0

    def calculate_mrr(
        self,
        relevant_ranks: List[int]
    ) -> float:

        if not relevant_ranks:
            return 0.0

        reciprocal_ranks = [
            1 / rank
            for rank in relevant_ranks
            if rank > 0
        ]

        return round(
            float(np.mean(reciprocal_ranks)),
            4
        ) if reciprocal_ranks else 0.0

    def calculate_precision_at_k(
        self,
        relevance_scores: List[int],
        k: int = 10
    ) -> float:

        if not relevance_scores:
            return 0.0

        top_k = relevance_scores[:k]

        return round(sum(top_k) / k, 4)

    def calculate_recall_at_k(
        self,
        relevance_scores: List[int],
        total_relevant: int,
        k: int = 10
    ) -> float:

        if total_relevant == 0:
            return 0.0

        retrieved_relevant = sum(relevance_scores[:k])

        return round(
            retrieved_relevant / total_relevant,
            4
        )

    async def evaluate_ranking(
        self,
        ranked_candidates: List[Dict],
        ground_truth: List[str]
    ) -> Dict[str, float]:

        relevance = []
        relevant_ranks = []

        for idx, candidate in enumerate(ranked_candidates):

            is_relevant = (
                candidate.get("candidate_id")
                in ground_truth
            )

            relevance.append(1 if is_relevant else 0)

            if is_relevant:
                relevant_ranks.append(idx + 1)

        return {
            "ndcg@10": self.calculate_ndcg(relevance, 10),
            "ndcg@20": self.calculate_ndcg(relevance, 20),
            "mrr": self.calculate_mrr(relevant_ranks),
            "precision@10": self.calculate_precision_at_k(relevance, 10),
            "recall@10": self.calculate_recall_at_k(
                relevance,
                len(ground_truth),
                10
            )
        }

    def _dcg(
        self,
        scores: List[float]
    ) -> float:

        return sum(
            (
                (2 ** score - 1)
                / np.log2(idx + 2)
            )
            for idx, score in enumerate(scores)
        )
