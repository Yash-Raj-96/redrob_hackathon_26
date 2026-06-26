"""
hard_filter.py

Minimal knockout filters.

Purpose:
- Remove obviously irrelevant candidates
- Keep potentially good candidates for downstream scoring
- Avoid over-filtering strong ML/AI profiles
"""

AI_TITLE_KEYWORDS = {
    "machine learning",
    "ml engineer",
    "ai engineer",
    "artificial intelligence",
    "data scientist",
    "nlp",
    "research engineer",
    "applied scientist",
    "search engineer",
    "ranking engineer",
    "recommendation",
    "recsys",
    "retrieval",
    "llm",
    "deep learning",
    "computer vision",
    "speech"
}

CORE_AI_SKILLS = {
    "machine learning",
    "deep learning",
    "nlp",
    "llm",
    "rag",
    "embedding",
    "embeddings",
    "vector search",
    "vector database",
    "retrieval",
    "recommendation",
    "ranking",
    "learning to rank",
    "semantic search",
    "information retrieval",
    "faiss",
    "pinecone",
    "weaviate",
    "milvus",
    "qdrant",
    "opensearch",
    "elasticsearch",
    "transformers",
    "sentence transformers",
    "bert",
    "lora",
    "qlora",
    "peft",
    "fine-tuning",
    "fine tuning",
    "hugging face",
    "pytorch",
    "tensorflow"
}

CLEARLY_WRONG_TITLES = {
    "accountant",
    "hr manager",
    "human resources",
    "marketing manager",
    "content writer",
    "graphic designer",
    "social media",
    "seo",
    "sales manager",
    "business development",
    "finance manager",
    "operations manager",
    "supply chain",
    "procurement",
    "customer support",
    "civil engineer",
    "mechanical engineer",
    "electrical engineer",
    "architect",
    "interior designer",
    "teacher",
    "professor",
    "doctor",
    "nurse",
    "pharmacist",
    "lawyer",
    "legal"
}


def _has_any_tech_signal(c):
    """
    Quick technical relevance check.
    """

    title = c.get("current_title", "").lower()

    for kw in AI_TITLE_KEYWORDS:
        if kw in title:
            return True

    skills_concat = c.get("skills_concat", "").lower()

    for skill in CORE_AI_SKILLS:
        if skill in skills_concat:
            return True

    job_text = c.get("job_text", "").lower()

    tech_words = (
        "machine learning",
        "artificial intelligence",
        "data science",
        "ml",
        "nlp",
        "llm",
        "retrieval",
        "recommendation",
        "ranking",
        "search"
    )

    for word in tech_words:
        if word in job_text:
            return True

    return False


def _has_meaningful_ai_exp(c):
    """
    Require at least 2 AI-relevant skills.
    """

    skills = c.get("skill_names", [])

    count = 0

    for sk in skills:
        sk = sk.lower()

        for ai_skill in CORE_AI_SKILLS:
            if ai_skill in sk:
                count += 1
                break

    return count >= 2


def _is_clearly_wrong_title(c):
    """
    Reject clearly unrelated professions unless
    candidate shows strong AI evidence.
    """

    title = c.get("current_title", "").lower()

    matched_wrong_title = False

    for bad in CLEARLY_WRONG_TITLES:
        if bad in title:
            matched_wrong_title = True
            break

    if not matched_wrong_title:
        return False

    strong_ai_count = 0

    for sk in c.get("skill_names", []):

        sk = sk.lower()

        if any(
            keyword in sk
            for keyword in (
                "machine learning",
                "deep learning",
                "nlp",
                "llm",
                "embedding",
                "vector",
                "retrieval",
                "recommendation",
                "ranking",
                "faiss",
                "pinecone",
                "weaviate",
                "milvus",
                "qdrant",
                "opensearch",
                "elasticsearch",
                "transformers",
                "lora",
                "qlora",
                "peft",
                "fine-tuning"
            )
        ):
            strong_ai_count += 1

    return strong_ai_count < 3


def passes_hard_filter(c):
    """
    Minimal filtering.

    Keep candidate if:
    - Has at least 2 years experience
    - Shows technical AI/ML signals
    - Has at least 2 AI-relevant skills
    - Is not clearly unrelated
    """

    if c.get("yoe", 0) < 2:
        return False

    if not _has_any_tech_signal(c):
        return False

    if not _has_meaningful_ai_exp(c):
        return False

    if _is_clearly_wrong_title(c):
        return False

    return True