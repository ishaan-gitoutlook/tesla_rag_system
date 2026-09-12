"""
embeddings.py
Modular functions for embedding model initialization, embedding generation,
and vector similarity computation.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

# Cached model instance
_CACHED_MODEL = None
_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = _DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Loads and caches the SentenceTransformer embedding model.
    'all-MiniLM-L6-v2' maps sentences & paragraphs to a 384 dimensional dense vector space.
    
    Args:
        model_name: HuggingFace model identifier.
        
    Returns:
        SentenceTransformer model instance.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None or getattr(_CACHED_MODEL, "_model_name", None) != model_name:
        _CACHED_MODEL = SentenceTransformer(model_name)
        _CACHED_MODEL._model_name = model_name
    return _CACHED_MODEL


def generate_embeddings(
    texts: Union[str, List[str]], 
    model: SentenceTransformer,
    batch_size: int = 64,
    show_progress: bool = False
) -> np.ndarray:
    """
    Generates normalized L2 embeddings for input text or list of texts.
    
    Args:
        texts: String or list of strings to embed.
        model: SentenceTransformer instance.
        batch_size: Batch size for encoding.
        show_progress: Whether to show tqdm progress bar.
        
    Returns:
        np.ndarray of shape (N, 384) with float32 dtype.
    """
    if isinstance(texts, str):
        texts = [texts]
        
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # Crucial: cosine similarity is then dot product!
        convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


def compute_similarity(
    query_vector: np.ndarray, 
    doc_vectors: np.ndarray
) -> np.ndarray:
    """
    Computes cosine similarity between a query vector and a matrix of document vectors.
    Since vectors are L2-normalized, cosine similarity equals the dot product.
    
    Args:
        query_vector: 1D array of shape (D,) or 2D array of shape (1, D).
        doc_vectors: 2D array of shape (N, D).
        
    Returns:
        1D np.ndarray of shape (N,) containing similarity scores in range [-1.0, 1.0].
    """
    if query_vector.ndim == 2:
        query_vector = query_vector.squeeze(0)
        
    # Dot product with all doc vectors
    scores = np.dot(doc_vectors, query_vector)
    return scores
