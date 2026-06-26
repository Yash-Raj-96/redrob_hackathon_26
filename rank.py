"""
rank.py

Main entry point.

Pipeline:

1. Load candidates
2. Apply hard filters
3. Compute scores
4. Apply final reranking
5. Generate reasoning
6. Export final_candidates.csv
"""

import argparse

from src.loader import load_candidates
from src.hard_filter import passes_hard_filter
from src.ranking_engine import rank_candidates
from src.llm_reranker import rerank
from src.reasoning_generator import generate_reasoning
from src.submission_builder import build_submission


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates JSONL file"
    )

    parser.add_argument(
        "--out",
        default="final_candidates.csv",
        help="Output CSV file"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading candidates...")
    candidates = load_candidates(args.candidates)

    print(f"Loaded {len(candidates):,} candidates")

    # --------------------------------------------------
    # Hard filter
    # --------------------------------------------------
    filtered = [
        c
        for c in candidates
        if passes_hard_filter(c)
    ]

    print(
        f"{len(filtered):,} candidates "
        f"passed hard filtering"
    )

    if len(filtered) < 100:
        raise RuntimeError(
            f"Only {len(filtered)} candidates remain. "
            f"Need at least 100 for submission."
        )

    # --------------------------------------------------
    # Score + rank
    # --------------------------------------------------
    ranked = rank_candidates(filtered)

    # --------------------------------------------------
    # Final reranking
    # --------------------------------------------------
    ranked = rerank(ranked)

    # --------------------------------------------------
    # Generate reasoning
    # --------------------------------------------------
    for c in ranked:
        c["reasoning"] = generate_reasoning(c)

    # --------------------------------------------------
    # Build submission
    # --------------------------------------------------
    build_submission(
        ranked,
        args.out
    )

    print(
        f"Submission written to {args.out}"
    )

    print(
        f"Top candidate: "
        f"{ranked[0]['candidate_id']} "
        f"(score={ranked[0]['_final_score']:.6f})"
    )


if __name__ == "__main__":
    main()