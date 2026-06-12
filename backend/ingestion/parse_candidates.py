"""
Parse candidate data from various formats
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from backend.utils.logger import setup_logger
from backend.app.config import settings

logger = setup_logger(__name__)

class CandidateParser:
    """Parse candidate data from JSONL/CSV/Parquet formats"""
    
    def __init__(self, raw_data_path: str = None):
        self.raw_data_path = raw_data_path or settings.DATA_RAW_PATH
        self.processed_data_path = settings.DATA_PROCESSED_PATH
        
    async def parse_jsonl(self, file_path: str) -> pd.DataFrame:
        """Parse JSONL file"""
        logger.info(f"Parsing JSONL file: {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    data.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
                    continue
        
        df = pd.DataFrame(data)
        logger.info(f"Parsed {len(df)} records from JSONL")
        return df
    
    async def parse_candidate_profiles(self) -> pd.DataFrame:
        """Main parsing orchestrator"""
        # Look for candidate data file
        candidate_files = list(Path(self.raw_data_path).glob("candidates.*"))
        
        if not candidate_files:
            raise FileNotFoundError(f"No candidate data found in {self.raw_data_path}")
        
        file_path = candidate_files[0]
        ext = file_path.suffix.lower()
        
        if ext == '.jsonl':
            df = await self.parse_jsonl(str(file_path))
        elif ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return df