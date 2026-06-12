"""
Complete preprocessing pipeline (SAFE + PRODUCTION FIX)
"""
import os
import pandas as pd
from pathlib import Path

from backend.ingestion.parse_candidates import CandidateParser
from backend.ingestion.clean_profiles import ProfileCleaner
from backend.ingestion.normalize_skills import SkillNormalizer
from backend.utils.logger import setup_logger
from backend.app.config import settings

logger = setup_logger(__name__)


class PreprocessingPipeline:
    """Orchestrate the entire preprocessing pipeline"""

    def __init__(self):
        self.parser = CandidateParser()
        self.cleaner = ProfileCleaner()
        self.skill_normalizer = SkillNormalizer()

    async def run(self, force_reprocess: bool = False):
        """Run complete preprocessing pipeline (safe mode)"""

        logger.info("Starting preprocessing pipeline...")

        output_path = os.path.join(
            settings.DATA_PROCESSED_PATH,
            "cleaned_candidates.parquet"
        )

        # =========================================================
        # LOAD CACHE IF EXISTS
        # =========================================================
        if os.path.exists(output_path) and not force_reprocess:
            logger.info(f"Loading cached dataset: {output_path}")
            df = pd.read_parquet(output_path)
            logger.info(f"Loaded {len(df)} candidates from cache")
            return df

        # =========================================================
        # STEP 1: PARSE
        # =========================================================
        logger.info("Step 1: Parsing raw candidate data...")
        df = await self.parser.parse_candidate_profiles()
        logger.info(f"Parsed {len(df)} candidates")

        # =========================================================
        # STEP 2: CLEAN
        # =========================================================
        logger.info("Step 2: Cleaning profiles...")
        df = await self.cleaner.clean_profiles(df)
        logger.info("Profile cleaning complete")

        # =========================================================
        # STEP 3: NORMALIZE SKILLS
        # =========================================================
        logger.info("Step 3: Normalizing skills...")
        df = await self.skill_normalizer.normalize(df)
        logger.info("Skill normalization complete")

        # =========================================================
        # STEP 4: FEATURE ENGINEERING (SAFE)
        # =========================================================
        logger.info("Step 4: Creating additional features...")
        df = await self._create_features(df)
        logger.info("Feature engineering complete")

        # =========================================================
        # STEP 5: SAVE OUTPUT
        # =========================================================
        logger.info("Step 5: Saving processed dataset...")

        os.makedirs(settings.DATA_PROCESSED_PATH, exist_ok=True)
        df.to_parquet(output_path, index=False)

        logger.info(f"Saved processed dataset -> {output_path}")

        return df

    # =========================================================
    # SAFE FEATURE ENGINEERING (FIXED)
    # =========================================================
    async def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Safe feature engineering (NO SCHEMA CRASH)"""

        # ---------------------------------------------------------
        # Ensure required columns ALWAYS exist
        # ---------------------------------------------------------
        defaults = {
            "name": "",
            "current_role": "",
            "years_experience": 0,
            "skills_normalized": [],
        }

        for col, default in defaults.items():
            if col not in df.columns:
                df[col] = default

        # Ensure numeric safety
        df["years_experience"] = pd.to_numeric(
            df["years_experience"],
            errors="coerce"
        ).fillna(0)

        # =========================================================
        # Feature 1: Profile completeness score
        # =========================================================
        mandatory_fields = ["name", "current_role", "years_experience"]

        df["profile_completeness"] = (
            df[mandatory_fields]
            .notna()
            .sum(axis=1) / len(mandatory_fields)
        )

        # =========================================================
        # Feature 2: Experience buckets
        # =========================================================
        df["seniority_level"] = pd.cut(
            df["years_experience"],
            bins=[0, 1, 3, 5, 8, 12, 100],
            labels=["Fresher", "Junior", "Mid", "Senior", "Lead", "Expert"]
        ).astype(str)

        # =========================================================
        # Feature 3: Safe profile text (for retrieval)
        # =========================================================
        df["profile_text"] = (
            df["current_role"].astype(str)
            + " "
            + df["skills_normalized"].astype(str)
            + " "
            + df.get("education", "").astype(str)
        )

        # =========================================================
        # Feature 4: Skill richness score
        # =========================================================
        df["skill_count"] = df["skills_normalized"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        return df


if __name__ == "__main__":
    import asyncio

    pipeline = PreprocessingPipeline()
    asyncio.run(pipeline.run(force_reprocess=True))