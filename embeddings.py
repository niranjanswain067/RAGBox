"""
embeddings.py - Local Text Vectorization Module

Generates dense semantic embeddings using sentence-transformers (all-MiniLM-L6-v2).
Runs 100% locally on CPU without relying on external API keys or cloud services.

Educational Note:
An embedding converts text into a numerical vector representing its semantic meaning.
In this 384-dimensional mathematical space, sentences with similar meanings
(e.g., "attendance rules" and "mandatory presence in class") map to vectors that point
in nearly the same direction, allowing similarity search via cosine distance.
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import config

# Global cache to ensure the model is loaded into memory only once
_EMBEDDING_MODEL_INSTANCE: Optional[SentenceTransformer] = None


def load_embedding_model(model_name: str = config.EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Loads and caches the SentenceTransformer embedding model.
    Ensures single initialization across the entire application lifecycle.

    Args:
        model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2).

    Returns:
        SentenceTransformer instance.
    """
    global _EMBEDDING_MODEL_INSTANCE
    if _EMBEDDING_MODEL_INSTANCE is None:
        print(f"[RAGBox] Loading embedding model: {model_name}...")
        _EMBEDDING_MODEL_INSTANCE = SentenceTransformer(model_name)
        print("[RAGBox] Embedding model loaded successfully.")
    return _EMBEDDING_MODEL_INSTANCE


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates numerical embedding vectors for a batch of text chunks.

    Args:
        texts: List of text strings to vectorize.

    Returns:
        List of 384-dimensional float vectors (one vector per input text).
    """
    if not texts:
        return []

    model = load_embedding_model()
    # convert_to_numpy=True and normalize_embeddings=True for standard cosine similarity
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    # Convert numpy array to standard Python list of lists for ChromaDB compatibility
    return embeddings.tolist()


def generate_query_embedding(query: str) -> List[float]:
    """
    Generates a single numerical embedding vector for the user's question.

    Args:
        query: The user's input question string.

    Returns:
        A single 384-dimensional float vector.
    """
    if not query or not query.strip():
        raise ValueError("Cannot generate embedding for an empty query.")

    model = load_embedding_model()
    query_vector = model.encode(
        query.strip(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return query_vector.tolist()
