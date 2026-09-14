"""
app.py
Flask Web Application for Interactive RAG over Tesla 10-K Filing.
Exposes REST endpoints and serves a modern, responsive UI.
"""

import os
import time

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


from flask import Flask, request, jsonify, render_template

from .rag.data_loader import load_and_chunk_pdf
from .rag.embeddings import get_embedding_model, generate_embeddings
from .rag.vector_store import build_vector_store, save_vector_store, load_vector_store, retrieve_top_k
from .rag.generator import generate_answer
from .config import EMBEDDING_MODEL_NAME, PDF_PATH, VECTOR_STORE_PATH

app = Flask(__name__, template_folder="templates", static_folder="static")


def create_app() -> Flask:
    """Return the configured Flask application."""
    return app

# Global singletons

STORE = None
EMBED_MODEL = None


def get_or_create_vector_store():
    """
    Loads pre-indexed vector store, or creates it on first launch.
    """
    global STORE, EMBED_MODEL
    if EMBED_MODEL is None:
        EMBED_MODEL = get_embedding_model(EMBEDDING_MODEL_NAME)

    if STORE is None:
        if VECTOR_STORE_PATH.exists():
            print(f"Loading vector store from {VECTOR_STORE_PATH}...")
            STORE = load_vector_store(str(VECTOR_STORE_PATH))
        else:
            print(f"Vector store not found. Indexing PDF: {PDF_PATH}...")
            chunks = load_and_chunk_pdf(str(PDF_PATH), chunk_size=750, chunk_overlap=150)
            texts = [c["text"] for c in chunks]
            embeddings = generate_embeddings(texts, EMBED_MODEL, batch_size=64, show_progress=True)
            STORE = build_vector_store(chunks, embeddings)
            save_vector_store(STORE, str(VECTOR_STORE_PATH))
            print(f"Vector store created with {STORE['total_chunks']} chunks.")

    return STORE, EMBED_MODEL


@app.route("/")
def index():
    """Serves the main application UI."""
    return render_template("index.html")


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Returns vector store and document statistics."""
    store, model = get_or_create_vector_store()
    return jsonify({
        "status": "ready",
        "document_name": "Tesla, Inc. Form 10-K (FY 2023)",
        "total_chunks": store["total_chunks"],
        "embedding_dim": store["dim"],
        "embedding_model": f"{EMBEDDING_MODEL_NAME} (384-d)",
        "generator_default": "Local Flan-T5 + Structured Extractor"
    })


@app.route("/api/query", methods=["POST"])
def api_query():
    """
    Main RAG query endpoint.
    Payload:
        query: str
        top_k: int (default 4)
        mode: 'local' | 'gemini' (default 'local')
        api_key: Optional[str]
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400
    raw_query = data.get("query", "")
    if not isinstance(raw_query, str):
        return jsonify({"error": "query must be a string."}), 400
    query_text = raw_query.strip()
    try:
        top_k = int(data.get("top_k", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be a positive integer."}), 400
    if top_k < 1 or top_k > 50:
        return jsonify({"error": "top_k must be between 1 and 50."}), 400
    mode = data.get("mode", "local")
    if not isinstance(mode, str) or mode not in {"local", "gemini"}:
        return jsonify({"error": "mode must be 'local' or 'gemini'."}), 400
    api_key = data.get("api_key")
    if api_key is not None and not isinstance(api_key, str):
        return jsonify({"error": "api_key must be a string."}), 400
    api_key = api_key.strip() if api_key else None

    if not query_text:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        t0 = time.time()
        store, model = get_or_create_vector_store()

        # 1. Retrieval
        t_retrieval_start = time.time()
        top_chunks = retrieve_top_k(query_text, model, store, k=top_k)
        retrieval_ms = (time.time() - t_retrieval_start) * 1000

        # 2. Generation
        t_gen_start = time.time()
        rag_output = generate_answer(query_text, top_chunks, mode=mode, api_key=api_key)
        generation_ms = (time.time() - t_gen_start) * 1000

        total_ms = (time.time() - t0) * 1000

        return jsonify({
            "query": query_text,
            "answer": rag_output["answer"],
            "sources": rag_output["sources"],
            "generator_mode": rag_output["generator_mode"],
            "timing": {
                "retrieval_ms": round(retrieval_ms, 1),
                "generation_ms": round(generation_ms, 1),
                "total_ms": round(total_ms, 1)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/reindex", methods=["POST"])
def api_reindex():
    """Forces re-indexing of the PDF document."""
    global STORE
    try:
        if VECTOR_STORE_PATH.exists():
            os.remove(VECTOR_STORE_PATH)
        STORE = None
        store, _ = get_or_create_vector_store()
        return jsonify({
            "message": "Re-indexing complete.",
            "total_chunks": store["total_chunks"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Pre-warm vector store on boot
    get_or_create_vector_store()
    print("Starting Flask RAG Server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
