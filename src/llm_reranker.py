"""
llm_reranker.py

Final reranking layer.

Despite the name, this implementation does NOT call an external LLM.
Instead it applies lightweight deterministic adjustments using
signals that often separate similarly qualified candidates.

This keeps results reproducible and challenge-safe.
"""


def rerank(candidates):
    """
    Input:
        candidates = list of candidate dicts
        each candidate already has:
            _final_score

    Output:
        same list with updated _final_score
    """

    for c in candidates:

        score = c.get("_final_score", 0.0)

        # ----------------------------------
        # Open To Work
        # ----------------------------------
        if c.get("open_to_work", False):
            score += 0.010

        # ----------------------------------
        # Immediate Availability
        # ----------------------------------
        notice = c.get("notice_days", 90)

        if notice <= 15:
            score += 0.010
        elif notice > 90:
            score -= 0.010

        # ----------------------------------
        # Recruiter Response Rate
        # ----------------------------------
        response_rate = c.get("response_rate", 0)

        if response_rate >= 0.80:
            score += 0.008
        elif response_rate < 0.20:
            score -= 0.008

        # ----------------------------------
        # Recent Activity
        # ----------------------------------
        days_active = c.get("days_since_active", 9999)

        if days_active <= 30:
            score += 0.005
        elif days_active > 120:
            score -= 0.010

        # ----------------------------------
        # GitHub Bonus
        # ----------------------------------
        github = c.get("github_score", -1)

        if github >= 80:
            score += 0.005

        # ----------------------------------
        # Verified Profile Bonus
        # ----------------------------------
        if (
            c.get("verified_email", False)
            and c.get("verified_phone", False)
        ):
            score += 0.003

        c["_final_score"] = score

    return sorted(
        candidates,
        key=lambda x: (
            -x["_final_score"],
            x["candidate_id"]
        )
    )