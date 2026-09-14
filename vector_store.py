"""
vector_store.py
Modular functions for vector store construction, serialization,
and top-k cosine similarity retrieval.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings import generate_embeddings, compute_similarity


def build_vector_store(
    chunks: List[Dict[str, Any]], 
    embeddings: np.ndarray
) -> Dict[str, Any]:
    """
    Constructs a vector store dictionary binding chunks to their embeddings.
    
    Args:
        chunks: List of chunk metadata dictionaries.
        embeddings: 2D numpy array of shape (N, D).
        
    Returns:
        Vector store dictionary.
    """
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array with shape (N, D).")
    if len(chunks) != len(embeddings):
        raise ValueError("Number of chunks and embeddings must match.")
    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "dim": embeddings.shape[1] if len(embeddings) > 0 else 0,
        "total_chunks": len(chunks)
    }


def save_vector_store(store: Dict[str, Any], file_path: str) -> None:
    """
    Persists the vector store to disk.
    
    Args:
        store: Vector store dictionary.
        file_path: Destination file path.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(store, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_vector_store(file_path: str) -> Dict[str, Any]:
    """
    Loads a persisted vector store from disk.
    
    Args:
        file_path: Path to the serialized vector store file.
        
    Returns:
        Vector store dictionary.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Vector store file not found at: {file_path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def retrieve_top_k(
    query: str, 
    model: SentenceTransformer, 
    store: Dict[str, Any], 
    k: int = 4
) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k most semantically relevant text chunks for a query.
    
    Args:
        query: User input query.
        model: SentenceTransformer embedding model.
        store: Vector store dictionary with 'chunks' and 'embeddings'.
        k: Number of nearest neighbors to retrieve.
        
    Returns:
        List of retrieved chunk dicts, each augmented with 'score' and 'rank'.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")
    chunks = store["chunks"]
    embeddings = np.asarray(store["embeddings"])
    
    if len(chunks) == 0:
        return []
        
    # Generate query embedding
    query_vec = generate_embeddings(query.strip(), model)
    
    # Compute similarity scores
    scores = compute_similarity(query_vec, embeddings)
    
    # Top-k indices
    k = min(k, len(chunks))
    # argpartition is faster than full sort for large collections
    top_indices = np.argpartition(scores, -k)[-k:]
    # Sort the top-k indices in descending order of score
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk_data = dict(chunks[idx])
        chunk_data["score"] = float(scores[idx])
        chunk_data["rank"] = rank
        results.append(chunk_data)
        
    return results
