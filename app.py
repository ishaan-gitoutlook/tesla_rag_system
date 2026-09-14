"""Backward-compatible entry point for the Tesla RAG web application."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tesla_rag.web import app  # noqa: E402


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
