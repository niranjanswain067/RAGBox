"""
vector_store.py - ChromaDB Persistent Vector Database Module

Manages persistent on-disk storage and retrieval of chunk embeddings
using ChromaDB. Handles document deduplication, collection statistics,
and safe database resets.

Educational Note:
A Vector Database (like ChromaDB) indexes multi-dimensional numerical vectors
using specialized algorithms like HNSW (Hierarchical Navigable Small World).
This enables finding the nearest neighbors (most semantically similar chunks)
in milliseconds, even among thousands of documents.
"""

from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
import config

_CHROMA_CLIENT: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Returns a persistent ChromaDB client instance.
    Data is stored on disk at config.PERSIST_DIRECTORY.
    """
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        _CHROMA_CLIENT = chromadb.PersistentClient(
            path=config.PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
    return _CHROMA_CLIENT


def get_or_create_collection(
    collection_name: str = config.COLLECTION_NAME
) -> chromadb.Collection:
    """
    Retrieves or creates the target ChromaDB collection with cosine distance metric.
    """
    client = get_chroma_client()
    # Uses cosine distance for sentence-transformers normalized vectors
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def is_document_indexed(doc_hash: str) -> bool:
    """
    Checks if a document with the given content hash has already been stored in ChromaDB.
    Prevents redundant re-embedding and duplicate chunks.
    """
    if not doc_hash:
        return False
    collection = get_or_create_collection()
    if collection.count() == 0:
        return False

    # Look for existing records with this doc_hash in metadata
    existing = collection.get(
        where={"doc_hash": doc_hash},
        limit=1
    )
    return bool(existing and existing.get("ids") and len(existing["ids"]) > 0)


def add_documents(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    doc_hash: str = ""
) -> int:
    """
    Inserts chunk texts, embeddings, and associated metadata into the ChromaDB collection.

    Args:
        chunks: List of chunk dictionaries from chunker.py.
        embeddings: List of embedding vectors from embeddings.py.
        doc_hash: Unique content hash of the source document.

    Returns:
        Number of chunks stored.
    """
    if not chunks or not embeddings:
        return 0

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})."
        )

    collection = get_or_create_collection()

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for chunk, emb in zip(chunks, embeddings):
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        metadatas.append({
            "source": str(chunk.get("source", "unknown")),
            "page": int(chunk.get("page", 1)),
            "chunk_id": str(chunk["chunk_id"]),
            "doc_hash": str(doc_hash),
            "char_count": int(chunk.get("char_count", len(chunk["text"])))
        })

    # Upsert ensures that existing chunks with matching IDs are safely updated
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(ids)


def query_documents(
    query_embedding: List[float],
    top_k: int = config.DEFAULT_TOP_K
) -> List[Dict[str, Any]]:
    """
    Queries ChromaDB for the Top-K most semantically similar chunks.

    Args:
        query_embedding: The 384-dimensional question embedding vector.
        top_k: Number of nearest chunks to retrieve.

    Returns:
        List of dictionaries with keys:
        - "chunk_id": str
        - "text": str
        - "source": str
        - "page": int
        - "distance": float (cosine distance: 0 is identical, higher is further)
        - "similarity": float (1 - cosine distance)
    """
    collection = get_or_create_collection()
    total_docs = collection.count()

    if total_docs == 0:
        return []

    # Safeguard top_k against total document count
    effective_k = min(top_k, total_docs)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=effective_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved: List[Dict[str, Any]] = []

    if not results or not results.get("ids") or len(results["ids"][0]) == 0:
        return retrieved

    ids_list = results["ids"][0]
    docs_list = results["documents"][0] if results.get("documents") else []
    metas_list = results["metadatas"][0] if results.get("metadatas") else []
    dists_list = results["distances"][0] if results.get("distances") else []

    for i in range(len(ids_list)):
        chunk_id = ids_list[i]
        doc_text = docs_list[i] if i < len(docs_list) else ""
        meta = metas_list[i] if i < len(metas_list) else {}
        distance = float(dists_list[i]) if i < len(dists_list) else 0.0

        # With cosine distance: similarity = 1 - distance
        # Clamp to [0.0, 1.0] for clean reporting
        similarity = max(0.0, min(1.0, 1.0 - distance))

        retrieved.append({
            "chunk_id": chunk_id,
            "text": doc_text,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", 1),
            "distance": round(distance, 4),
            "similarity": round(similarity, 4)
        })

    return retrieved


def get_collection_stats() -> Dict[str, Any]:
    """
    Returns aggregate statistics of the ChromaDB collection.
    Useful for Teaching Mode and UI diagnostics.
    """
    collection = get_or_create_collection()
    total_chunks = collection.count()

    if total_chunks == 0:
        return {
            "total_chunks": 0,
            "unique_sources": [],
            "source_count": 0,
            "total_pages_indexed": 0
        }

    all_data = collection.get(include=["metadatas"])
    metadatas = all_data.get("metadatas") or []

    sources = set()
    pages_by_source: Dict[str, set] = {}

    for meta in metadatas:
        if meta:
            src = meta.get("source", "unknown")
            pg = meta.get("page", 1)
            sources.add(src)
            if src not in pages_by_source:
                pages_by_source[src] = set()
            pages_by_source[src].add(pg)

    total_pages = sum(len(pages) for pages in pages_by_source.values())

    return {
        "total_chunks": total_chunks,
        "unique_sources": sorted(list(sources)),
        "source_count": len(sources),
        "total_pages_indexed": total_pages
    }


def clear_collection(collection_name: str = config.COLLECTION_NAME) -> bool:
    """
    Deletes and recreates the ChromaDB collection.
    Completely resets stored vectors and documents.
    """
    client = get_chroma_client()
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection might not exist yet
    # Recreate the fresh, empty collection
    get_or_create_collection(collection_name=collection_name)
    return True
