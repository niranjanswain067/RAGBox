"""
retriever.py - High-Accuracy Document Retrieval Module

Executes the semantic search phase with relevance scoring and keyword-aware ranking.
Takes a user's natural language question, generates its dense vector embedding,
queries ChromaDB, and ranks the Top-K chunks for maximum context accuracy.

Educational Note:
Retrieval is the "R" in RAG. Higher accuracy is achieved by combining dense semantic
similarity (cosine distance in vector space) with term overlap relevance, ensuring that
exact terms, numbers, and conceptual themes are all captured.
"""

import re
from typing import Any, Dict, List
import config
from embeddings import generate_query_embedding
from vector_store import query_documents, get_collection_stats


def compute_keyword_overlap_score(query: str, chunk_text: str) -> float:
    """
    Computes lexical term overlap between query keywords and chunk text.
    Filters out common stopwords to focus on content words.
    """
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "with", "by", "what", "which", "who", "whom",
        "how", "why", "where", "when", "does", "do", "did", "can", "could",
        "would", "should", "and", "or", "not"
    }

    def tokenize(text: str) -> set:
        words = re.findall(r'\b[a-zA-Z0-9_%-]+\b', text.lower())
        return {w for w in words if w not in stopwords and len(w) > 1}

    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    chunk_tokens = tokenize(chunk_text)
    overlap = query_tokens.intersection(chunk_tokens)
    return len(overlap) / len(query_tokens)


def retrieve_relevant_chunks(
    question: str,
    top_k: int = config.DEFAULT_TOP_K,
    similarity_threshold: float = 0.20
) -> List[Dict[str, Any]]:
    """
    Retrieves and ranks the most relevant chunks for a question.
    Combines dense cosine similarity with keyword matching for superior precision.

    Args:
        question: User's natural language question.
        top_k: Number of chunks to retrieve.
        similarity_threshold: Minimum cosine similarity score (discards irrelevant noise).

    Returns:
        List of ranked chunk dictionaries.
    """
    if not question or not question.strip():
        return []

    stats = get_collection_stats()
    if stats["total_chunks"] == 0:
        return []

    # 1. Vectorize query
    query_vector = generate_query_embedding(question)

    # 2. Retrieve candidate chunks from ChromaDB (retrieve slightly more candidates for ranking)
    candidate_k = min(max(top_k * 2, top_k + 2), stats["total_chunks"])
    raw_chunks = query_documents(query_vector, top_k=candidate_k)

    if not raw_chunks:
        return []

    # 3. Score and rerank candidates
    scored_chunks = []
    for chunk in raw_chunks:
        cos_sim = chunk.get("similarity", 0.0)

        # Filter out chunks that fall below the relevance threshold
        if cos_sim < similarity_threshold:
            continue

        keyword_score = compute_keyword_overlap_score(question, chunk.get("text", ""))

        # Composite score: 70% dense semantic similarity + 30% exact keyword overlap
        composite_score = (0.70 * cos_sim) + (0.30 * keyword_score)

        chunk_copy = dict(chunk)
        chunk_copy["keyword_overlap"] = round(keyword_score, 4)
        chunk_copy["composite_score"] = round(composite_score, 4)
        scored_chunks.append(chunk_copy)

    # If all candidates fell below threshold, keep top candidates to avoid empty return if plausible
    if not scored_chunks and raw_chunks:
        scored_chunks = raw_chunks[:top_k]
    else:
        # Sort by composite score descending
        scored_chunks.sort(key=lambda c: c.get("composite_score", c.get("similarity", 0.0)), reverse=True)

    return scored_chunks[:top_k]
