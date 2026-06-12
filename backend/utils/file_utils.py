"""
File handling utilities
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from backend.utils.logger import setup_logger

logger = setup_logger(__name__)


# =========================================================
# Directory Utilities
# =========================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists
    """

    directory = Path(path)

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    return directory


# =========================================================
# JSON Utilities
# =========================================================

def save_json(
    data: Dict[str, Any],
    filepath: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Save dictionary to JSON file
    """

    filepath = Path(filepath)

    ensure_directory(filepath.parent)

    try:

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=indent,
                ensure_ascii=False,
                default=str
            )

        logger.info(f"Saved JSON: {filepath}")

    except Exception as e:

        logger.exception(
            f"Failed saving JSON to {filepath}"
        )

        raise RuntimeError(str(e))


def load_json(
    filepath: Union[str, Path],
    default: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Load JSON file
    """

    filepath = Path(filepath)

    if not filepath.exists():

        if default is not None:
            return default

        raise FileNotFoundError(
            f"JSON file not found: {filepath}"
        )

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        logger.exception(
            f"Failed loading JSON from {filepath}"
        )

        raise RuntimeError(str(e))


# =========================================================
# JSONL Utilities
# =========================================================

def save_jsonl(
    records: List[Dict[str, Any]],
    filepath: Union[str, Path]
) -> None:
    """
    Save records as JSONL
    """

    filepath = Path(filepath)

    ensure_directory(filepath.parent)

    try:

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as f:

            for record in records:

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        default=str
                    )
                    + "\n"
                )

        logger.info(f"Saved JSONL: {filepath}")

    except Exception as e:

        logger.exception(
            f"Failed saving JSONL to {filepath}"
        )

        raise RuntimeError(str(e))


def load_jsonl(
    filepath: Union[str, Path]
) -> List[Dict[str, Any]]:
    """
    Load JSONL file
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {filepath}"
        )

    records = []

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if line:
                    records.append(json.loads(line))

        logger.info(f"Loaded JSONL: {filepath}")

        return records

    except Exception as e:

        logger.exception(
            f"Failed loading JSONL from {filepath}"
        )

        raise RuntimeError(str(e))


# =========================================================
# DataFrame Utilities
# =========================================================

def save_dataframe(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    format: str = "parquet"
) -> None:
    """
    Save dataframe in various formats
    """

    filepath = Path(filepath)

    ensure_directory(filepath.parent)

    format = format.lower()

    try:

        # ======================================
        # Parquet
        # ======================================

        if format == "parquet":

            df.to_parquet(
                filepath,
                index=False
            )

        # ======================================
        # CSV
        # ======================================

        elif format == "csv":

            df.to_csv(
                filepath,
                index=False
            )

        # ======================================
        # JSONL
        # ======================================

        elif format == "jsonl":

            df.to_json(
                filepath,
                orient="records",
                lines=True
            )

        # ======================================
        # Excel
        # ======================================

        elif format in ["xlsx", "excel"]:

            df.to_excel(
                filepath,
                index=False
            )

        else:

            raise ValueError(
                f"Unsupported format: {format}"
            )

        logger.info(
            f"Saved dataframe "
            f"(rows={len(df)}, cols={len(df.columns)}) "
            f"to {filepath}"
        )

    except Exception as e:

        logger.exception(
            f"Failed saving dataframe to {filepath}"
        )

        raise RuntimeError(str(e))


def load_dataframe(
    filepath: Union[str, Path]
) -> pd.DataFrame:
    """
    Load dataframe from supported formats
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataframe file not found: {filepath}"
        )

    extension = filepath.suffix.lower()

    try:

        # ======================================
        # Parquet
        # ======================================

        if extension == ".parquet":

            df = pd.read_parquet(filepath)

        # ======================================
        # CSV
        # ======================================

        elif extension == ".csv":

            df = pd.read_csv(filepath)

        # ======================================
        # JSONL
        # ======================================

        elif extension in [".jsonl", ".json"]:

            df = pd.read_json(
                filepath,
                lines=True
            )

        # ======================================
        # Excel
        # ======================================

        elif extension in [".xlsx", ".xls"]:

            df = pd.read_excel(filepath)

        else:

            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        logger.info(
            f"Loaded dataframe "
            f"(rows={len(df)}, cols={len(df.columns)}) "
            f"from {filepath}"
        )

        return df

    except Exception as e:

        logger.exception(
            f"Failed loading dataframe from {filepath}"
        )

        raise RuntimeError(str(e))


# =========================================================
# File Utilities
# =========================================================

def file_exists(
    filepath: Union[str, Path]
) -> bool:
    """
    Check if file exists
    """

    return Path(filepath).exists()


def delete_file(
    filepath: Union[str, Path]
) -> bool:
    """
    Delete file safely
    """

    filepath = Path(filepath)

    try:

        if filepath.exists():

            filepath.unlink()

            logger.info(f"Deleted file: {filepath}")

        return True

    except Exception as e:

        logger.error(
            f"Failed deleting file {filepath}: {str(e)}"
        )

        return False


def get_file_size_mb(
    filepath: Union[str, Path]
) -> float:
    """
    Get file size in MB
    """

    filepath = Path(filepath)

    if not filepath.exists():
        return 0.0

    size_bytes = filepath.stat().st_size

    return round(size_bytes / (1024 * 1024), 2)