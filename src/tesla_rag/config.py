"""Application configuration and repository paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "tsla-20231231-gen.pdf"
VECTOR_STORE_PATH = DATA_DIR / "vector_store.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
