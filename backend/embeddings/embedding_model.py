"""
Embedding model wrapper (CPU + ONNX + fallback safe)
"""

import numpy as np
from typing import List, Union, Optional
import torch

from sentence_transformers import SentenceTransformer
from backend.utils.logger import setup_logger
from backend.app.config import settings

logger = setup_logger(__name__)


class EmbeddingModel:

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_onnx: bool = True,
        device: str = "cpu",
        batch_size: int = 128
    ):

        self.model_name = (
            model_name
            or getattr(settings, "EMBEDDING_MODEL_NAME", None)
            or "BAAI/bge-base-en-v1.5"
        )

        self.batch_size = batch_size
        self.device = "cpu"

        self.use_onnx = use_onnx
        self.embedding_dim = None  # 🔥 FIX: always define early

        logger.info(f"Loading embedding model: {self.model_name}")

        # ==========================
        # OPTION 1: ONNX (FASTEST)
        # ==========================
        if self.use_onnx:
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                self.model = ORTModelForFeatureExtraction.from_pretrained(
                    self.model_name,
                    file_name="model.onnx"
                )

                self.embedding_dim = self.model.config.hidden_size

                logger.info("ONNX model loaded successfully")

            except Exception as e:
                logger.warning(f"ONNX failed, falling back to SentenceTransformer: {e}")
                self.use_onnx = False

        # ==========================
        # OPTION 2: SentenceTransformer
        # ==========================
        if not self.use_onnx:

            self.model = SentenceTransformer(
                self.model_name,
                device="cpu"
            )

            self.model.max_seq_length = 256

            self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedding dimension: {self.embedding_dim}")

    # =========================================================
    # ENCODE
    # =========================================================
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:

        if not texts:
            return np.zeros((1, self.embedding_dim), dtype=np.float32)

        if isinstance(texts, str):
            texts = [texts]

        texts = [t or "" for t in texts]

        # ==========================
        # ONNX PATH
        # ==========================
        if self.use_onnx:

            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="np"
            )

            outputs = self.model(**inputs)

            # mean pooling
            emb = outputs.last_hidden_state.mean(axis=1)

            return emb.astype(np.float32)

        # ==========================
        # SentenceTransformer PATH
        # ==========================
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return np.asarray(emb, dtype=np.float32)