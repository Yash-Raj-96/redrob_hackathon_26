"""
skill_matcher.py

Computes skill relevance score for Senior AI Engineer /
Search / Retrieval / Recommendation candidates.

Output:
    score in [0,1]

Stores:
    candidate["_skill_score"]
"""

TIER_A = {
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "faiss",
    "vector search",
    "semantic search",
    "information retrieval",
    "learning to rank",
    "recommendation systems",
    "recommendation",
    "retrieval",
    "search",
    "rag",
}

TIER_B = {
    "embeddings",
    "sentence transformers",
    "transformers",
    "opensearch",
    "elasticsearch",
    "bm25",
    "lora",
    "qlora",
    "peft",
    "fine tuning",
    "fine-tuning",
}

TIER_C = {
    "machine learning",
    "deep learning",
    "nlp",
    "python",
    "tensorflow",
    "pytorch",
    "llm",
    "generative ai",
    "data science",
    "mlops",
    "docker",
    "kubernetes",
    "aws",
    "gcp",
    "azure",
}


def _extract_skills(candidate):
    skills = []

    for s in candidate.get("skills_raw", []):
        if isinstance(s, dict):
            name = s.get("name", "")
        else:
            name = str(s)

        name = name.strip().lower()

        if name:
            skills.append(name)

    return skills


def skill_score(candidate):
    """
    Returns normalized skill score [0,1].
    """

    skills = _extract_skills(candidate)

    if not skills:
        candidate["_skill_score"] = 0.0
        return 0.0

    matched_a = set()
    matched_b = set()
    matched_c = set()

    for skill in skills:

        for kw in TIER_A:
            if kw in skill:
                matched_a.add(kw)

        for kw in TIER_B:
            if kw in skill:
                matched_b.add(kw)

        for kw in TIER_C:
            if kw in skill:
                matched_c.add(kw)

    score = 0.0

    score += len(matched_a) * 1.0
    score += len(matched_b) * 0.75
    score += len(matched_c) * 0.40

    # Retrieval/Search bonus
    if len(matched_a) >= 3:
        score += 1.5

    if len(matched_a) >= 5:
        score += 1.5

    if len(matched_a) >= 7:
        score += 1.0

    # Modern LLM stack bonus
    llm_terms = {"lora", "qlora", "peft", "transformers"}

    if len(matched_b.intersection(llm_terms)) >= 2:
        score += 1.0

    # Normalize
    final_score = min(score / 15.0, 1.0)

    candidate["_skill_score"] = round(final_score, 6)

    candidate["_matched_tier_a"] = len(matched_a)
    candidate["_matched_tier_b"] = len(matched_b)
    candidate["_matched_tier_c"] = len(matched_c)

    return candidate["_skill_score"]


# Backward compatibility
score_skill_match = skill_score