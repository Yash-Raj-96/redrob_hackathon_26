"""
Helper utility functions
"""

import hashlib
import math
import random
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def generate_candidate_id(
    name: str,
    email: str,
    use_uuid: bool = False
) -> str:
    """
    Generate unique candidate ID
    """

    if use_uuid:
        return str(uuid.uuid4())

    unique_string = (
        f"{name.strip().lower()}_"
        f"{email.strip().lower()}_"
        f"{datetime.now(timezone.utc).timestamp()}_"
        f"{random.randint(1000, 9999)}"
    )

    return hashlib.md5(unique_string.encode()).hexdigest()[:12]


def extract_years_from_string(text: str) -> float:
    """
    Extract years of experience from text
    """

    if not text:
        return 0.0

    text = str(text).lower()

    patterns = [
        r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)',
        r'(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)',
        r'experience\s*(?:of)?\s*(\d+(?:\.\d+)?)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return 0.0


def normalize_text(text: Optional[str]) -> str:
    """
    Normalize text for matching/searching
    """

    if not text:
        return ""

    text = str(text).lower().strip()

    # Remove special chars except spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def chunk_list(
    lst: List[Any],
    chunk_size: int
) -> List[List[Any]]:
    """
    Split list into chunks
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    return [
        lst[i:i + chunk_size]
        for i in range(0, len(lst), chunk_size)
    ]


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """
    Flatten nested list
    """

    return [
        item
        for sublist in nested_list
        for item in sublist
    ]


def deduplicate_list(items: List[Any]) -> List[Any]:
    """
    Remove duplicates while preserving order
    """

    seen = set()

    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float
    """

    try:
        if value is None:
            return default

        if isinstance(value, str) and not value.strip():
            return default

        return float(value)

    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int
    """

    try:
        if value is None:
            return default

        return int(float(value))

    except (ValueError, TypeError):
        return default


def format_currency(amount: float) -> str:
    """
    Format currency in Indian Rupees
    """

    amount = safe_float(amount)

    if amount >= 10000000:
        return f"₹{amount / 10000000:.1f}Cr"

    if amount >= 100000:
        return f"₹{amount / 100000:.1f}L"

    if amount >= 1000:
        return f"₹{amount / 1000:.1f}K"

    return f"₹{amount:,.0f}"


def calculate_percentile(
    scores: List[float],
    score: float
) -> float:
    """
    Calculate percentile rank
    """

    if not scores:
        return 0.0

    count = sum(1 for s in scores if s <= score)

    return round((count / len(scores)) * 100, 2)


def cosine_similarity(
    vec1: List[float],
    vec2: List[float]
) -> float:
    """
    Compute cosine similarity
    """

    if not vec1 or not vec2:
        return 0.0

    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def normalize_score(
    score: float,
    min_score: float,
    max_score: float
) -> float:
    """
    Normalize score to range [0, 1]
    """

    score = safe_float(score)

    if max_score == min_score:
        return 0.0

    normalized = (score - min_score) / (max_score - min_score)

    return max(0.0, min(1.0, normalized))


def timestamp_now() -> str:
    """
    Get UTC timestamp ISO format
    """

    return datetime.now(timezone.utc).isoformat()


def truncate_text(
    text: str,
    max_length: int = 200
) -> str:
    """
    Truncate text safely
    """

    if not text:
        return ""

    text = str(text)

    if len(text) <= max_length:
        return text

    return text[:max_length].rstrip() + "..."


def extract_skills_from_text(text: str) -> List[str]:
    """
    Basic skill extraction utility
    """

    if not text:
        return []

    common_skills = [
        "python",
        "java",
        "javascript",
        "typescript",
        "sql",
        "aws",
        "docker",
        "kubernetes",
        "react",
        "node.js",
        "fastapi",
        "django",
        "flask",
        "machine learning",
        "deep learning",
        "tensorflow",
        "pytorch",
        "nlp",
        "data science",
        "mongodb",
        "postgresql",
        "redis",
        "git",
        "linux"
    ]

    text_lower = normalize_text(text)

    found_skills = []

    for skill in common_skills:
        if skill.lower() in text_lower:
            found_skills.append(skill)

    return deduplicate_list(found_skills)


def build_api_response(
    success: bool,
    message: str,
    data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Standard API response builder
    """

    return {
        "success": success,
        "message": message,
        "data": data or {},
        "timestamp": timestamp_now()
    }