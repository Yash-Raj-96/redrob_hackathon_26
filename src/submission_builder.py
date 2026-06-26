"""
submission_builder.py

Builds the final challenge submission CSV.

Output format:

candidate_id,rank,score,reasoning

Requirements:
- Exactly 100 rows
- Rank 1..100
- Scores non-increasing
- UTF-8 CSV
- Compatible with validate_submission.py
"""

import csv


HEADER = [
    "candidate_id",
    "rank",
    "score",
    "reasoning",
]


def build_submission(candidates, output_path):
    """
    Parameters
    ----------
    candidates : list[dict]
        Ranked candidates.
        Must already contain:
            candidate_id
            _final_score
            reasoning

    output_path : str
        Destination CSV path.
    """

    if len(candidates) < 100:
        raise ValueError(
            f"Need at least 100 ranked candidates, got {len(candidates)}"
        )

    top100 = candidates[:100]

    rows = []

    previous_score = float("inf")

    for rank, c in enumerate(top100, start=1):

        score = round(
            float(c.get("_final_score", 0.0)),
            6
        )

        # Ensure validator rule:
        # scores must be non-increasing by rank
        if score > previous_score:
            score = previous_score

        previous_score = score

        rows.append({
            "candidate_id": c["candidate_id"],
            "rank": rank,
            "score": f"{score:.6f}",
            "reasoning": c.get("reasoning", "")
        })

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=HEADER
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


def prepare_submission_rows(candidates):
    """
    Creates rows without writing a file.

    Useful for testing.
    """

    rows = []

    for rank, c in enumerate(candidates[:100], start=1):

        rows.append({
            "candidate_id": c["candidate_id"],
            "rank": rank,
            "score": f"{c['_final_score']:.6f}",
            "reasoning": c.get("reasoning", "")
        })

    return rows