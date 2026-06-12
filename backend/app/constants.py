"""
Global constants and scoring configuration
"""

from typing import Dict, List, Any


# =========================================================
# Application Metadata
# =========================================================

APP_NAME: str = "INDIA RUNS Candidate Engine"
APP_VERSION: str = "1.0.0"
DEFAULT_TIMEZONE: str = "UTC"


# =========================================================
# Core Ranking Weights
# =========================================================

SCORING_WEIGHTS: Dict[str, float] = {
    "skill_match": 0.40,
    "experience": 0.25,
    "recruiter_signal": 0.15,
    "availability": 0.10,
    "salary_fit": 0.10,
}

# ✅ BACKWARD COMPATIBILITY (IMPORTANT FIX)
# Your old code expects this name
REQUIRED_SKILLS_THRESHOLD: float = 0.60
REQUIRED_SKILLS_PENALTY: float = 0.30


# =========================================================
# Skill Matching Configuration
# =========================================================

SKILL_MATCH_WEIGHTS: Dict[str, float] = {
    "required_skills": 1.00,
    "preferred_skills": 0.70,
    "domain_knowledge": 0.50,
    "tools_technologies": 0.60,
    "soft_skills": 0.30,
    "certifications": 0.25,
}

SKILL_MATCH_CONFIG: Dict[str, Any] = {
    "minimum_required_skill_match": 0.60,
    "required_skill_penalty": 0.30,
    "max_missing_required_skills": 5,
    "boost_for_exact_match": 1.15,
    "partial_match_threshold": 0.75,
}


# =========================================================
# Experience Configuration
# =========================================================

EXPERIENCE_CONFIG: Dict[str, Any] = {
    "max_years": 20,
    "min_years": 0,
    "optimal_range": (3, 8),
    "senior_range": (8, 15),
    "leadership_bonus_threshold": 10,
    "decay_rate": 0.8,
    "role_relevance_weight": 0.5,
    "company_reputation_weight": 0.2,
}


# =========================================================
# Recruiter Signals
# =========================================================

RECRUITER_SIGNALS: Dict[str, float] = {
    "github_activity": 0.25,
    "linkedin_endorsements": 0.20,
    "platform_activity": 0.15,
    "profile_completeness": 0.15,
    "certifications": 0.10,
    "portfolio_quality": 0.10,
    "open_source_contributions": 0.05,
}


# =========================================================
# Salary Configuration
# =========================================================

SALARY_CONFIG: Dict[str, Any] = {
    "ideal_variance_pct": 0.10,
    "acceptable_variance_pct": 0.25,
    "max_penalty_pct": 0.50,
}


# =========================================================
# Availability / Notice Period
# =========================================================

NOTICE_PERIOD_SCORES: Dict[str, float] = {
    "immediate": 1.00,
    "7 days": 0.95,
    "15 days": 0.90,
    "30 days": 0.75,
    "45 days": 0.55,
    "60 days": 0.35,
    "90 days": 0.15,
}

DEFAULT_NOTICE_SCORE: float = 0.50


# =========================================================
# Candidate Schema Fields
# =========================================================

CANDIDATE_FIELDS: List[str] = [
    "candidate_id",
    "name",
    "email",
    "phone",
    "location",
    "current_role",
    "years_experience",
    "skills",
    "skills_normalized",
    "education",
    "previous_companies",
    "certifications",
    "salary_expectation",
    "notice_period",
    "github_url",
    "linkedin_url",
    "portfolio_url",
    "last_active",
    "profile_score",
    "languages",
    "achievements",
    "projects",
]


# =========================================================
# Vector Search Configuration
# =========================================================

VECTOR_SEARCH_CONFIG: Dict[str, Any] = {
    "embedding_dimension": 1024,
    "similarity_metric": "cosine",
    "hybrid_search_weight": 0.60,
    "semantic_search_weight": 0.70,
    "keyword_search_weight": 0.30,
    "top_k_retrieval": 100,
    "top_k_rerank": 50,
    "max_candidates_for_llm": 20,
}


# =========================================================
# Retrieval Thresholds
# =========================================================

RETRIEVAL_THRESHOLDS: Dict[str, float] = {
    "minimum_similarity_score": 0.65,
    "high_confidence_score": 0.85,
    "low_confidence_score": 0.45,
}


# =========================================================
# Explainability Settings
# =========================================================

EXPLAINABILITY_CONFIG: Dict[str, Any] = {
    "max_strengths": 5,
    "max_weaknesses": 5,
    "max_improvement_tips": 5,
    "max_missing_skills": 5,
    "enable_llm_explanations": True,
}


# =========================================================
# Bias Detection Thresholds
# =========================================================

BIAS_THRESHOLDS: Dict[str, float] = {
    "gender_imbalance": 0.70,
    "location_concentration": 0.50,
    "education_elitism": 0.40,
    "experience_dominance": 0.60,
}


# =========================================================
# Analytics Thresholds
# =========================================================

ANALYTICS_CONFIG: Dict[str, Any] = {
    "top_skills_limit": 20,
    "top_locations_limit": 10,
    "rare_skill_threshold": 5,
    "high_demand_skill_threshold": 0.30,
}


# =========================================================
# API Rate Limits
# =========================================================

RATE_LIMITS: Dict[str, int] = {
    "search": 100,
    "ranking": 50,
    "explain": 200,
    "analytics": 30,
    "copilot": 60,
}


# =========================================================
# Cache Keys
# =========================================================

CACHE_KEYS: Dict[str, str] = {
    "candidates_df": "candidates_dataframe",
    "candidate_embeddings": "candidate_embeddings",
    "faiss_index": "faiss_index",
    "bm25_index": "bm25_index",
    "jd_embeddings": "jd_embeddings",
    "search_results": "search_results",
    "ranked_results": "ranked_results",
}


# =========================================================
# Prompt Templates
# =========================================================

PROMPT_PATHS: Dict[str, str] = {
    "rerank": "backend/llm/prompts/rerank_prompt.txt",
    "explain": "backend/llm/prompts/explain_prompt.txt",
    "recruiter_chat": "backend/llm/prompts/recruiter_chat_prompt.txt",
    "bias_analysis": "backend/llm/prompts/bias_prompt.txt",
}


# =========================================================
# Monitoring / Health
# =========================================================

HEALTH_THRESHOLDS: Dict[str, float] = {
    "max_cpu_percent": 90.0,
    "max_memory_percent": 90.0,
    "max_disk_percent": 95.0,
}


# =========================================================
# Supported File Formats
# =========================================================

SUPPORTED_RESUME_FORMATS: List[str] = [
    ".pdf",
    ".docx",
    ".txt",
]

SUPPORTED_EXPORT_FORMATS: List[str] = [
    "json",
    "csv",
    "xlsx",
]


# =========================================================
# LLM Defaults
# =========================================================

LLM_CONFIG: Dict[str, Any] = {
    "default_provider": "openai",
    "default_temperature": 0.3,
    "default_max_tokens": 500,
    "rerank_temperature": 0.1,
    "explanation_temperature": 0.5,
}


# =========================================================
# Search Suggestions
# =========================================================

DEFAULT_SEARCH_SUGGESTIONS: List[str] = [
    "Python Developer",
    "Machine Learning Engineer",
    "Data Scientist",
    "Full Stack Developer",
    "DevOps Engineer",
    "Backend Engineer",
    "Frontend React Developer",
    "AI Engineer",
    "Cloud Architect",
    "Product Manager",
]