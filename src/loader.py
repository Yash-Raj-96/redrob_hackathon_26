import json
from datetime import date


REFERENCE_DATE = date(2026, 6, 16)


def _days_ago(date_str):
    if not date_str or len(date_str) < 10:
        return 9999
    try:
        # Fast direct string slicing to avoid strptime regex and locale check overheads
        y = int(date_str[:4])
        m = int(date_str[5:7])
        d = int(date_str[8:10])
        return (REFERENCE_DATE - date(y, m, d)).days
    except Exception:
        return 9999

def _skill_names(skills_list):
    names = []

    for skill in skills_list:
        name = skill.get("name")
        if name:
            names.append(name.lower())

    return names
#def _skill_names(skills_list):
 #   return [s["name"].lower() for s in skills_list]


def _all_job_text(career_history):
    parts = []
    for job in career_history:
        parts.append(job.get("title") or "")
        parts.append(job.get("company") or "")
        parts.append(job.get("description") or "")
    # Lowercase the entire joined string at once to reduce allocations
    return " ".join(parts).lower()


def _total_career_months(career_history):
    return sum(j.get("duration_months", 0) for j in career_history)


def _edu_tier(education):
    # return the best tier found
    tiers = [e.get("tier", "tier_3") for e in education]
    if "tier_1" in tiers:
        return "tier_1"
    if "tier_2" in tiers:
        return "tier_2"
    return "tier_3"


def flatten(c):
    p = c["profile"]
    sig = c["redrob_signals"]
    career = c.get("career_history", [])
    skills = c.get("skills", [])
    edu = c.get("education", [])

    skill_names_list = _skill_names(skills)

    flat = {
        "candidate_id": c["candidate_id"],
        # profile
        "name": p.get("anonymized_name", ""),
        "headline": p.get("headline", ""),
        "summary": p.get("summary", "").lower(),
        "location": p.get("location", ""),
        "country": p.get("country", ""),
        "yoe": p.get("years_of_experience", 0),
        "current_title": p.get("current_title", "").lower(),
        "current_company": p.get("current_company", ""),
        "current_company_size": p.get("current_company_size", ""),
        "current_industry": p.get("current_industry", ""),
        # derived
        "skill_names": skill_names_list,
        "skills_raw": skills,
        "skills_concat": " ".join(skill_names_list),
        "job_text": _all_job_text(career),
        "career_raw": career,
        "total_career_months": _total_career_months(career),
        "edu_tier": _edu_tier(edu),
        # signals
        "profile_completeness": sig.get("profile_completeness_score", 0),
        "days_since_active": _days_ago(sig.get("last_active_date")),
        "open_to_work": sig.get("open_to_work_flag", False),
        "profile_views_30d": sig.get("profile_views_received_30d", 0),
        "applications_30d": sig.get("applications_submitted_30d", 0),
        "response_rate": sig.get("recruiter_response_rate", 0.0),
        "avg_response_hours": sig.get("avg_response_time_hours", 999),
        "assessment_scores": sig.get("skill_assessment_scores", {}),
        "connection_count": sig.get("connection_count", 0),
        "endorsements_total": sig.get("endorsements_received", 0),
        "notice_days": sig.get("notice_period_days", 90),
        "salary_min": sig.get("expected_salary_range_inr_lpa", {}).get("min", 0),
        "salary_max": sig.get("expected_salary_range_inr_lpa", {}).get("max", 0),
        "work_mode": sig.get("preferred_work_mode", ""),
        "willing_relocate": sig.get("willing_to_relocate", False),
        "github_score": sig.get("github_activity_score", -1),
        "search_appearances_30d": sig.get("search_appearance_30d", 0),
        "saved_by_recruiters_30d": sig.get("saved_by_recruiters_30d", 0),
        "interview_completion": sig.get("interview_completion_rate", 0.0),
        "offer_acceptance": sig.get("offer_acceptance_rate", -1),
        "verified_email": sig.get("verified_email", False),
        "verified_phone": sig.get("verified_phone", False),
        "linkedin_connected": sig.get("linkedin_connected", False),
    }
    return flat


def load_candidates(path):
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            candidates.append(flatten(raw))
    return candidates
