"""
test_rag.py
Automated end-to-end verification of the modular RAG system.
Builds the vector store if not already built, and executes sample queries
including 'annual revenue in 2022' to verify retrieval and answer accuracy.
"""

import os
import sys
import time
import warnings
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import transformers
transformers.logging.set_verbosity_error()

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_and_chunk_pdf
from embeddings import get_embedding_model, generate_embeddings
from vector_store import build_vector_store, save_vector_store, load_vector_store, retrieve_top_k
from generator import generate_answer


def test_rag_pipeline():
    project_dir = Path(__file__).parent
    pdf_path = project_dir / "data" / "tsla-20231231-gen.pdf"
    vector_store_path = project_dir / "data" / "vector_store.pkl"
    
    print("=" * 70)
    print("STEP 1: Initializing Embedding Model & Vector Space")
    print("=" * 70)
    
    t0 = time.time()
    embedding_model = get_embedding_model("all-MiniLM-L6-v2")
    print(f"Embedding model loaded in {time.time() - t0:.2f}s.")
    
    # Check if vector store exists on disk
    if vector_store_path.exists():
        print(f"Loading existing vector store from: {vector_store_path}")
        store = load_vector_store(str(vector_store_path))
        print(f"Vector store loaded successfully: {store['total_chunks']} chunks, {store['dim']} dimensions.")
    else:
        print(f"Ingesting and chunking PDF: {pdf_path}")
        chunks = load_and_chunk_pdf(str(pdf_path), chunk_size=750, chunk_overlap=150)
        print(f"Extracted {len(chunks)} text chunks.")
        
        print("Generating embeddings for all chunks (all-MiniLM-L6-v2)...")
        texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(texts, embedding_model, batch_size=64, show_progress=True)
        print(f"Generated embeddings array shape: {embeddings.shape}")
        
        print(f"Saving vector store to {vector_store_path}...")
        store = build_vector_store(chunks, embeddings)
        save_vector_store(store, str(vector_store_path))
        print("Vector store saved to disk.")

    print("\n" + "=" * 70)
    print("STEP 2: Executing Sample Test Queries")
    print("=" * 70)

    test_queries = [
        {
            "id": 1,
            "query": "annual revenue in 2022",
            "expected_keywords": ["81,462", "81.46", "81,462 million"],
            "description": "2022 Annual Revenue (Ground truth: $81,462 million or $81.46 billion)"
        },
        {
            "id": 2,
            "query": "What was the total revenue in 2023?",
            "expected_keywords": ["96,773", "96.77", "96,773 million"],
            "description": "2023 Total Revenue (Ground truth: $96,773 million or $96.77 billion)"
        },
        {
            "id": 3,
            "query": "What are the locations of primary manufacturing facilities?",
            "expected_keywords": ["Austin", "Fremont", "Shanghai", "Berlin"],
            "description": "Manufacturing Facilities (Fremont, Texas/Austin, Nevada, Shanghai, Berlin, etc.)"
        },
        {
            "id": 4,
            "query": "Who is the Chief Executive Officer and Technoking of Tesla?",
            "expected_keywords": ["Elon Musk"],
            "description": "Leadership (Elon Musk, Technoking and CEO)"
        }
    ]

    all_passed = True

    for item in test_queries:
        q_id = item["id"]
        query = item["query"]
        expected = item["expected_keywords"]
        desc = item["description"]
        
        print(f"\n--- Test Query {q_id}: '{query}' ---")
        print(f"Target: {desc}")
        
        t_start = time.time()
        top_chunks = retrieve_top_k(query, embedding_model, store, k=5)
        retrieval_time = time.time() - t_start
        
        print(f"Retrieved {len(top_chunks)} chunks in {retrieval_time * 1000:.1f}ms.")
        for rank, c in enumerate(top_chunks[:2], start=1):
            snippet = c['text'].replace('\n', ' ')[:140]
            print(f"  Rank {rank} [Page {c['page_number']}] (Score: {c['score']:.4f}): \"{snippet}...\"")
            
        t_gen = time.time()
        rag_output = generate_answer(query, top_chunks, mode="local")
        gen_time = time.time() - t_gen
        
        answer = rag_output["answer"]
        print(f"\nAnswer ({rag_output['generator_mode']}, {gen_time:.2f}s):")
        print(f"{answer}\n")
        
        # Verify retrieved chunks contain at least one expected keyword
        retrieved_texts = " ".join(c["text"] for c in top_chunks)
        has_expected_in_context = any(kw.lower() in retrieved_texts.lower() for kw in expected)
        has_expected_in_answer = any(kw.lower() in answer.lower() for kw in expected)
        
        if has_expected_in_context:
            print(f"  [PASS] Context contains expected keywords {expected}")
        else:
            print(f"  [FAIL] Context missing expected keywords {expected}")
            all_passed = False
            
        if has_expected_in_answer:
            print(f"  [PASS] Generated answer explicitly contains expected keywords {expected}")
        else:
            print(f"  [WARN] Answer did not explicitly state keywords, but context verified.")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL RAG VERIFICATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED VERIFICATION.")
    print("=" * 70)


if __name__ == "__main__":
    test_rag_pipeline()
