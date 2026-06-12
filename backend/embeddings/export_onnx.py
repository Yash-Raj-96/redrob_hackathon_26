from optimum.onnxruntime import ORTModelForFeatureExtraction
from sentence_transformers import SentenceTransformer

import os

MODEL_NAME = "BAAI/bge-base-en-v1.5"
EXPORT_PATH = "data/models/bge_onnx"

os.makedirs(EXPORT_PATH, exist_ok=True)

print("Loading model...")

# Load HF model + export to ONNX
ort_model = ORTModelForFeatureExtraction.from_pretrained(
    MODEL_NAME,
    export=True,
    provider="CPUExecutionProvider"
)

ort_model.save_pretrained(EXPORT_PATH)

print(f"ONNX model saved to: {EXPORT_PATH}")