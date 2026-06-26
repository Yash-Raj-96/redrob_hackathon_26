"""
career_matcher.py

Career relevance scoring.

Output:
    candidate["_career_score"]

Returns:
    float in [0,1]
"""

HIGH_VALUE_TITLES = {
    "senior machine learning engineer",
    "machine learning engineer",
    "ml engineer",
    "senior ml engineer",
    "ai engineer",
    "senior ai engineer",
    "nlp engineer",
    "senior nlp engineer",
    "search engineer",
    "recommendation systems engineer",
    "applied scientist",
    "senior applied scientist",
    "data scientist",
    "senior data scientist",
    "ai research engineer",
    "applied ml engineer",
}

SEARCH_KEYWORDS = {
    "search",
    "retrieval",
    "recommendation",
    "ranking",
    "semantic search",
    "information retrieval",
    "learning to rank",
    "vector search",
    "rag",
}

TOP_AI_COMPANIES = {
    "google",
    "microsoft",
    "amazon",
    "aws",
    "meta",
    "linkedin",
    "netflix",
    "uber",
    "swiggy",
    "zomato",
    "cred",
    "razorpay",
    "freshworks",
    "sarvam ai",
    "haptik",
    "zoho",
    "salesforce",
    "nykaa",
    "ola",
    "genpact ai",
    "glance",
    "mad street den",
}


def _contains_any(text, keywords):
    text = (text or "").lower()
    return any(k in text for k in keywords)


def career_score(candidate):
    """
    Main scoring function expected by ranking_engine.py
    """

    score = 0.0

    current_title = (
        candidate.get("current_title", "")
        .strip()
        .lower()
    )

    current_company = (
        candidate.get("current_company", "")
        .strip()
        .lower()
    )

    yoe = float(candidate.get("yoe", 0))

    career = candidate.get("career_raw", [])

    # ----------------------------------
    # Current title
    # ----------------------------------

    if current_title in HIGH_VALUE_TITLES:
        score += 4.0

    elif any(
        kw in current_title
        for kw in (
            "engineer",
            "scientist",
            "machine learning",
            "ml",
            "ai",
            "nlp",
        )
    ):
        score += 2.5

    # ----------------------------------
    # Company quality
    # ----------------------------------

    if any(company in current_company for company in TOP_AI_COMPANIES):
        score += 1.5

    # ----------------------------------
    # Experience
    # ----------------------------------

    if 5 <= yoe <= 10:
        score += 2.0
    elif 3 <= yoe < 5:
        score += 1.5
    elif 10 < yoe <= 15:
        score += 1.5
    elif yoe >= 2:
        score += 1.0

    # ----------------------------------
    # Career history
    # ----------------------------------

    relevant_jobs = 0
    search_jobs = 0

    for job in career:

        title = job.get("title", "").lower()
        description = job.get("description", "").lower()

        combined = f"{title} {description}"

        if any(
            kw in combined
            for kw in (
                "machine learning",
                "ml engineer",
                "ai engineer",
                "data scientist",
                "nlp",
                "applied scientist",
                "research engineer",
            )
        ):
            relevant_jobs += 1

        if _contains_any(combined, SEARCH_KEYWORDS):
            search_jobs += 1

    score += min(relevant_jobs * 0.75, 3.0)
    score += min(search_jobs * 1.0, 3.0)

    # ----------------------------------
    # Seniority bonus
    # ----------------------------------

    if any(
        kw in current_title
        for kw in (
            "senior",
            "staff",
            "principal",
            "lead",
        )
    ):
        score += 1.0

    # ----------------------------------
    # Normalize
    # ----------------------------------

    final_score = min(score / 15.0, 1.0)

    candidate["_career_score"] = round(final_score, 6)
    candidate["_relevant_jobs"] = relevant_jobs
    candidate["_search_jobs"] = search_jobs

    return candidate["_career_score"]


# Compatibility aliases
compute_career_score = career_score
score_career_match = career_score