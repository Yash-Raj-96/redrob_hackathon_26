"""
behavioral_scorer.py

Scores platform/recruiter signals that indicate candidate availability,
responsiveness, engagement, and profile quality.

Returns score in range [0.0, 1.0]
"""


def behavioral_score(c):
    score = 0.0

    # --------------------------------------------------
    # Availability / Notice Period (30%)
    # --------------------------------------------------
    notice = c.get("notice_days", 90)

    if notice <= 15:
        score += 0.30
    elif notice <= 30:
        score += 0.25
    elif notice <= 60:
        score += 0.18
    elif notice <= 90:
        score += 0.10
    else:
        score += 0.03

    # --------------------------------------------------
    # Recruiter Response Rate (20%)
    # --------------------------------------------------
    response_rate = c.get("response_rate", 0.0)

    if response_rate >= 0.90:
        score += 0.20
    elif response_rate >= 0.75:
        score += 0.16
    elif response_rate >= 0.60:
        score += 0.12
    elif response_rate >= 0.40:
        score += 0.08
    elif response_rate >= 0.20:
        score += 0.04

    # --------------------------------------------------
    # Recent Activity (15%)
    # --------------------------------------------------
    days_active = c.get("days_since_active", 9999)

    if days_active <= 7:
        score += 0.15
    elif days_active <= 30:
        score += 0.12
    elif days_active <= 60:
        score += 0.08
    elif days_active <= 90:
        score += 0.04

    # --------------------------------------------------
    # Open To Work (10%)
    # --------------------------------------------------
    if c.get("open_to_work", False):
        score += 0.10

    # --------------------------------------------------
    # Profile Completeness (10%)
    # --------------------------------------------------
    completeness = c.get("profile_completeness", 0)

    if completeness >= 90:
        score += 0.10
    elif completeness >= 75:
        score += 0.07
    elif completeness >= 60:
        score += 0.04

    # --------------------------------------------------
    # GitHub Activity (5%)
    # --------------------------------------------------
    github = c.get("github_score", -1)

    if github >= 85:
        score += 0.05
    elif github >= 70:
        score += 0.04
    elif github >= 50:
        score += 0.03
    elif github >= 30:
        score += 0.01

    # --------------------------------------------------
    # Recruiter Interest Signals (5%)
    # --------------------------------------------------
    saved = c.get("saved_by_recruiters_30d", 0)
    appearances = c.get("search_appearances_30d", 0)

    interest = min(saved * 2 + appearances / 20.0, 10)

    score += (interest / 10.0) * 0.05

    # --------------------------------------------------
    # Interview Completion (3%)
    # --------------------------------------------------
    interview_completion = c.get("interview_completion", 0)

    score += min(interview_completion, 1.0) * 0.03

    # --------------------------------------------------
    # Verification Bonus (2%)
    # --------------------------------------------------
    verified_count = 0

    if c.get("verified_email", False):
        verified_count += 1

    if c.get("verified_phone", False):
        verified_count += 1

    if c.get("linkedin_connected", False):
        verified_count += 1

    score += min(verified_count / 3.0, 1.0) * 0.02

    # --------------------------------------------------
    # Clamp
    # --------------------------------------------------
    return min(score, 1.0)