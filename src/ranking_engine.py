"""
ranking_engine.py

Combines skill, career, and behavioral scores into a final ranking score.
Stores component scores on each candidate object.

Outputs:
    c["_skill_score"]
    c["_career_score"]
    c["_behavior_score"]
    c["_final_score"]
"""

import src.skill_matcher as skill_matcher
import src.career_matcher as career_matcher
import src.behavioral_scorer as behavioral_scorer


SKILL_WEIGHT = 0.55
CAREER_WEIGHT = 0.30
BEHAVIOR_WEIGHT = 0.15


def _call_skill_score(c):
    if hasattr(skill_matcher, "skill_score"):
        return skill_matcher.skill_score(c)

    if hasattr(skill_matcher, "compute_skill_score"):
        return skill_matcher.compute_skill_score(c)

    raise AttributeError(
        "skill_matcher.py must define skill_score() "
        "or compute_skill_score()."
    )


def _call_career_score(c):
    if hasattr(career_matcher, "career_score"):
        return career_matcher.career_score(c)

    if hasattr(career_matcher, "compute_career_score"):
        return career_matcher.compute_career_score(c)

    raise AttributeError(
        "career_matcher.py must define career_score() "
        "or compute_career_score()."
    )


def _call_behavior_score(c):
    if hasattr(behavioral_scorer, "behavioral_score"):
        return behavioral_scorer.behavioral_score(c)

    if hasattr(behavioral_scorer, "compute_behavioral_score"):
        return behavioral_scorer.compute_behavioral_score(c)

    raise AttributeError(
        "behavioral_scorer.py must define behavioral_score() "
        "or compute_behavioral_score()."
    )


def compute_score(c):
    """
    Compute final weighted score for a candidate.
    """

    skill = float(_call_skill_score(c))
    career = float(_call_career_score(c))
    behavior = float(_call_behavior_score(c))

    final_score = (
        skill * SKILL_WEIGHT
        + career * CAREER_WEIGHT
        + behavior * BEHAVIOR_WEIGHT
    )

    c["_skill_score"] = round(skill, 6)
    c["_career_score"] = round(career, 6)
    c["_behavior_score"] = round(behavior, 6)
    c["_final_score"] = round(final_score, 6)

    return c["_final_score"]


def score_candidates(candidates):
    for c in candidates:
        compute_score(c)

    return candidates


def rank_candidates(candidates):
    score_candidates(candidates)

    return sorted(
        candidates,
        key=lambda x: (
            -x["_final_score"],
            x["candidate_id"]
        )
    )