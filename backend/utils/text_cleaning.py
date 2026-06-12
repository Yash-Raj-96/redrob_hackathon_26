"""
Text cleaning utilities
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and unwanted special characters
    """

    if not text:
        return ""

    text = str(text)

    # Normalize whitespace first
    text = re.sub(r'\s+', ' ', text)

    # Remove unwanted special characters (keep useful punctuation for resumes/JDs)
    text = re.sub(r'[^\w\s.,\-\/@#\+\(\)]', '', text)

    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalize text for matching/search
    """

    if not text:
        return ""

    text = str(text).lower().strip()

    # Replace multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text


def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text
    """

    if not text:
        return []

    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'

    emails = re.findall(pattern, text)

    return list(set(emails))


def extract_phone_numbers(text: str) -> List[str]:
    """
    Extract phone numbers from text (basic international support)
    """

    if not text:
        return []

    # Matches:
    # +91 9876543210
    # (123) 456-7890
    # 98765 43210
    pattern = r'(\+?\d[\d\s\-\(\)]{8,}\d)'

    phones = re.findall(pattern, text)

    # Clean formatting
    cleaned = [
        re.sub(r'\s+', ' ', p).strip()
        for p in phones
        if len(re.sub(r'\D', '', p)) >= 10
    ]

    return list(set(cleaned))


def remove_extra_spaces(text: str) -> str:
    """
    Remove redundant whitespace
    """

    if not text:
        return ""

    return re.sub(r'\s+', ' ', str(text)).strip()


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text
    """

    if not text:
        return []

    pattern = r'https?://[^\s]+'

    return list(set(re.findall(pattern, text)))


def tokenize(text: str) -> List[str]:
    """
    Simple word tokenizer
    """

    if not text:
        return []

    text = normalize_text(text)

    return [t for t in text.split(" ") if t]


def remove_stop_chars(text: str) -> str:
    """
    Remove excessive punctuation noise
    """

    if not text:
        return ""

    # Keep meaningful punctuation, remove noise
    text = re.sub(r'[^\w\s.,\-@]', ' ', text)

    return remove_extra_spaces(text)