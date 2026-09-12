"""
app.py
Flask Web Application for Interactive RAG over Tesla 10-K Filing.
Exposes REST endpoints and serves a modern, responsive UI.
"""

import os
import time
import warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import transformers
transformers.logging.set_verbosity_error()

from flask import Flask, request, jsonify, render_template

from data_loader import load_and_chunk_pdf
from embeddings import get_embedding_model, generate_embeddings
from vector_store import build_vector_store, save_vector_store, load_vector_store, retrieve_top_k
from generator import generate_answer

app = Flask(__name__)

# Global singletons
BASE_DIR = Path(__file__).parent
PDF_PATH = BASE_DIR / "data" / "tsla-20231231-gen.pdf"
VECTOR_STORE_PATH = BASE_DIR / "data" / "vector_store.pkl"

STORE = None
EMBED_MODEL = None


def get_or_create_vector_store():
    """
    Loads pre-indexed vector store, or creates it on first launch.
    """
    global STORE, EMBED_MODEL
    if EMBED_MODEL is None:
        EMBED_MODEL = get_embedding_model("all-MiniLM-L6-v2")
        
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
        "embedding_model": "all-MiniLM-L6-v2 (384-d)",
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
    data = request.get_json() or {}
    query_text = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))
    mode = data.get("mode", "local")
    api_key = data.get("api_key", "").strip() or None

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
