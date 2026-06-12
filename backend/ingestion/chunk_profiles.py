"""
Chunk candidate profiles for better embedding representation (ULTRA FAST VERSION)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


class ProfileChunker:
    """Fast vectorized-safe profile chunker (CPU optimized)"""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    # =========================================================
    # 🚀 MAIN ENTRY (FAST VERSION)
    # =========================================================
    async def chunk_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Chunking {len(df)} candidate profiles...")

        chunks_data = []

        # 🚀 FIX 1: use itertuples (FASTEST pandas iteration)
        for row in df.itertuples(index=False):
            candidate = row._asdict()

            candidate_chunks = self._create_chunks(candidate)

            chunks_data.append({
                "candidate_id": getattr(row, "candidate_id", None),
                "chunks": candidate_chunks,
                "num_chunks": len(candidate_chunks)
            })

        chunks_df = pd.DataFrame(chunks_data)

        total_chunks = int(chunks_df["num_chunks"].sum()) if not chunks_df.empty else 0
        logger.info(f"Created {total_chunks} total chunks")

        return chunks_df

    # =========================================================
    # CORE CHUNKING
    # =========================================================
    def _create_chunks(self, candidate: Dict) -> List[Dict[str, Any]]:
        chunks = []

        basic = self._create_basic_section(candidate)
        if basic:
            chunks.append({"section": "basic_info", "text": basic, "weight": 0.3})

        skills = self._create_skills_section(candidate)
        if skills:
            chunks.append({"section": "skills", "text": skills, "weight": 0.4})

        exp = self._create_experience_section(candidate)
        if exp:
            chunks.append({"section": "experience", "text": exp, "weight": 0.2})

        edu = self._create_education_section(candidate)
        if edu:
            chunks.append({"section": "education", "text": edu, "weight": 0.1})

        return chunks

    # =========================================================
    # SAFE HELPERS (FIXED FOR NUMPY + PANDAS)
    # =========================================================
    def _safe_list(self, value):
        if value is None:
            return []
        if isinstance(value, float) and np.isnan(value):
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return []

    def _safe_str(self, value):
        if value is None:
            return ""
        if isinstance(value, float) and np.isnan(value):
            return ""
        return str(value)

    # =========================================================
    # SECTION BUILDERS
    # =========================================================
    def _create_basic_section(self, candidate: Dict) -> str:
        parts = []

        role = candidate.get("current_role")
        if role:
            parts.append(f"Current Role: {self._safe_str(role)}")

        exp = candidate.get("years_experience")
        if exp is not None:
            parts.append(f"Experience: {exp} years")

        loc = candidate.get("location")
        if loc:
            parts.append(f"Location: {self._safe_str(loc)}")

        return " | ".join(parts)

    def _create_skills_section(self, candidate: Dict) -> str:
        skills = candidate.get("skills_normalized")

        if skills is None:
            skills = candidate.get("skills")

        skills = self._safe_list(skills)

        if not skills:
            return ""

        categorized = candidate.get("skills_categorized") or {}

        parts = ["Technical Skills:"]

        if isinstance(categorized, dict):
            for cat, vals in categorized.items():
                vals = self._safe_list(vals)

                if cat != "Other" and vals:
                    parts.append(f"{cat}: {', '.join(map(str, vals))}")

            other = self._safe_list(categorized.get("Other"))
            if other:
                parts.append(f"Other: {', '.join(map(str, other))}")

        else:
            parts.append(", ".join(map(str, skills[:30])))

        return " | ".join(parts)

    def _create_experience_section(self, candidate: Dict) -> str:
        parts = []

        companies = self._safe_list(candidate.get("previous_companies"))
        if companies:
            parts.append(f"Companies: {', '.join(map(str, companies[:5]))}")

        achievements = self._safe_list(candidate.get("achievements"))
        if achievements:
            parts.append(f"Achievements: {'; '.join(map(str, achievements[:3]))}")

        return " | ".join(parts)

    def _create_education_section(self, candidate: Dict) -> str:
        parts = []

        edu = candidate.get("education")
        if edu:
            parts.append(f"Education: {self._safe_str(edu)}")

        certs = self._safe_list(candidate.get("certifications"))
        if certs:
            parts.append(f"Certifications: {', '.join(map(str, certs[:3]))}")

        return " | ".join(parts)