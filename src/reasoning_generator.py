"""
reasoning_generator.py

Generates factual, rank-consistent candidate explanations.
Uses only real profile data.
No hallucinated information.
"""

TITLE_FIXES = {
    "ml engineer": "ML Engineer",
    "senior ml engineer": "Senior ML Engineer",
    "machine learning engineer": "Machine Learning Engineer",
    "senior machine learning engineer": "Senior Machine Learning Engineer",
    "ai engineer": "AI Engineer",
    "senior ai engineer": "Senior AI Engineer",
    "nlp engineer": "NLP Engineer",
    "senior nlp engineer": "Senior NLP Engineer",
    "data scientist": "Data Scientist",
    "senior data scientist": "Senior Data Scientist",
    "applied scientist": "Applied Scientist",
    "search engineer": "Search Engineer",
    "recommendation systems engineer": "Recommendation Systems Engineer",
}

PREFERRED_SKILLS = [
    "pinecone",
    "weaviate",
    "faiss",
    "milvus",
    "qdrant",
    "opensearch",
    "elasticsearch",
    "semantic search",
    "information retrieval",
    "recommendation",
    "learning to rank",
    "vector search",
    "embeddings",
    "rag",
    "lora",
    "qlora",
    "peft",
    "sentence transformers",
    "transformers",
]


def _format_title(title):
    title = (title or "").strip()
    return TITLE_FIXES.get(title.lower(), title.title())


def _yoe_str(yoe):
    try:
        yoe = float(yoe)
        return f"{yoe:.0f}" if yoe.is_integer() else f"{yoe:.1f}"
    except Exception:
        return "0"


def _location_phrase(c):
    location = c.get("location", "")
    country = c.get("country", "")

    if not country:
        return location

    if country.lower() == "india":
        return location

    if c.get("willing_relocate", False):
        return f"{location} (open to relocate)"

    return f"{location}, {country}"


def _top_relevant_skills(c, n=3):
    skills = []

    for s in c.get("skills_raw", []):
        name = s.get("name", "")
        if name:
            skills.append(name)

    selected = []

    for preferred in PREFERRED_SKILLS:
        for skill in skills:
            if preferred in skill.lower():
                if skill not in selected:
                    selected.append(skill)

    for skill in skills:
        if skill not in selected:
            selected.append(skill)

    return selected[:n]


def _signal_phrase(c):
    parts = []

    days_active = c.get("days_since_active", 9999)

    if days_active <= 7:
        parts.append("active this week")
    elif days_active <= 30:
        parts.append(f"active {days_active}d ago")

    response_rate = c.get("response_rate", 0)

    if response_rate >= 0.75:
        parts.append(f"high response rate ({response_rate:.0%})")

    notice_days = c.get("notice_days", 90)

    if notice_days <= 15:
        parts.append("immediate availability")

    github_score = c.get("github_score", -1)

    if github_score >= 60:
        parts.append(f"GitHub score {github_score:.0f}")

    return "; ".join(parts[:2])


def _concern_phrase(c):
    concerns = []

    if c.get("days_since_active", 0) > 90:
        concerns.append(
            f"last active {c['days_since_active']}d ago"
        )

    if c.get("response_rate", 1.0) < 0.20:
        concerns.append(
            f"low response rate ({c['response_rate']:.0%})"
        )

    if c.get("notice_days", 0) > 90:
        concerns.append(
            f"long notice period ({c['notice_days']} days)"
        )

    if (
        c.get("country", "").lower() != "india"
        and not c.get("willing_relocate", False)
    ):
        concerns.append(
            f"based in {c.get('country', 'unknown')}"
        )

    if c.get("_skill_score", 0) < 0.25:
        concerns.append("limited core skill match")

    if c.get("_career_score", 0) < 0.35:
        concerns.append("partially aligned career history")

    return "; ".join(concerns[:2])


def generate_reasoning(c):
    score = c.get("_final_score", 0)

    yoe = _yoe_str(c.get("yoe", 0))

    title = _format_title(
        c.get("current_title", "")
    )

    company = c.get("current_company", "")

    location = _location_phrase(c)

    skills = _top_relevant_skills(c)

    skill_text = (
        ", ".join(skills)
        if skills
        else "general ML background"
    )

    signal = _signal_phrase(c)

    concern = _concern_phrase(c)

    company_text = (
        f" at {company}"
        if company
        else ""
    )

    # Tier 1
    if score >= 0.75:

        line1 = (
            f"{title} with {yoe} yrs experience"
            f"{company_text}; relevant skills: "
            f"{skill_text}; {location}."
        )

        if signal:
            line2 = (
                f"Platform signals are strong: "
                f"{signal}."
            )
        else:
            line2 = (
                "Career history shows strong applied AI/ML experience."
            )

        return f"{line1} {line2}"

    # Tier 2
    elif score >= 0.60:

        line1 = (
            f"{yoe} yrs total; currently "
            f"{title}{company_text} "
            f"({location}); relevant skills include "
            f"{skill_text}."
        )

        if signal:
            line2 = (
                f"Behavioral signals: {signal}."
            )
        elif concern:
            line2 = (
                f"Minor concern: {concern}."
            )
        else:
            line2 = (
                "Solid fit for the core requirements."
            )

        return f"{line1} {line2}"

    # Tier 3
    elif score >= 0.45:

        line1 = (
            f"{title} ({yoe} yrs, {location}); "
            f"has {skill_text} but career history "
            f"is only partially aligned to the target role."
        )

        if concern:
            line2 = (
                f"Additional concern: {concern}."
            )
        else:
            line2 = (
                "Included as a marginal match due to technical overlap."
            )

        return f"{line1} {line2}"

    # Tier 4
    else:

        line1 = (
            f"Borderline candidate: "
            f"{title} ({yoe} yrs, {location}); "
            f"limited direct match to the role requirements."
        )

        if concern:
            line2 = (
                f"Key gaps: {concern}."
            )
        else:
            line2 = (
                "Skill and career alignment are below the primary candidate cohort."
            )

        return f"{line1} {line2}"