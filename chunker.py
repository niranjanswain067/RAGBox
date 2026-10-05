"""
chunker.py - Sentence-Aware Semantic Text Chunking Module

Segments document text into meaningful, overlapping windows that respect sentence
and paragraph boundaries instead of abruptly cutting words or sentences in half.

Educational Note:
Chunking divides large documents into smaller pieces so that relevant information
can be retrieved efficiently.
Why Sentence-Aware Chunking is Critical for Accuracy:
1. Blind character slicing often splits vital facts, numbers, or clauses in half
   (e.g., splitting "75" and "%" into different chunks).
2. Sentence-aware chunking ensures that every chunk contains complete, grammatically
   coherent statements.
3. Overlapping by complete sentences guarantees that semantic context spanning across
   chunk transitions is never lost.
"""

import re
from typing import Any, Dict, List


def split_into_sentences(text: str) -> List[str]:
    """
    Splits continuous text into sentence units using punctuation heuristics.
    Preserves delimiters and prevents splitting common abbreviations.
    """
    if not text:
        return []

    # Clean redundant whitespace
    normalized = " ".join(text.split())

    # Split on sentence terminals followed by space and capital letter or end of string
    # e.g., '. ', '? ', '! ', or newline breaks
    sentence_pattern = r'(?<=[.?!])\s+(?=[A-Z0-9])'
    sentences = re.split(sentence_pattern, normalized)

    # Filter out empty entries
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Splits text into overlapping chunks respecting sentence boundaries.
    Pure custom Python implementation without third-party frameworks like LangChain.

    Args:
        text: The raw input string to split.
        chunk_size: Target character length of each chunk.
        overlap: Target character overlap between consecutive chunks.

    Returns:
        List of non-empty, sentence-aligned string chunks.
    """
    if not text or not text.strip():
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return [text.strip()]

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_char_count = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If a single sentence exceeds chunk_size, we must include it
        if not current_sentences:
            current_sentences.append(sentence)
            current_char_count += sentence_len
            continue

        # If adding this sentence stays within chunk_size:
        if current_char_count + sentence_len + 1 <= chunk_size:
            current_sentences.append(sentence)
            current_char_count += sentence_len + 1
        else:
            # We reached the target chunk_size
            full_chunk = " ".join(current_sentences).strip()
            if full_chunk:
                chunks.append(full_chunk)

            # Slide window back to preserve overlap
            overlap_sentences: List[str] = []
            overlap_count = 0
            for prev_sentence in reversed(current_sentences):
                if overlap_count + len(prev_sentence) + 1 <= overlap:
                    overlap_sentences.insert(0, prev_sentence)
                    overlap_count += len(prev_sentence) + 1
                else:
                    break

            current_sentences = overlap_sentences + [sentence]
            current_char_count = sum(len(s) for s in current_sentences) + max(0, len(current_sentences) - 1)

    # Append remaining sentences
    if current_sentences:
        final_chunk = " ".join(current_sentences).strip()
        if final_chunk and (not chunks or final_chunk != chunks[-1]):
            chunks.append(final_chunk)

    return chunks


def chunk_document_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict[str, Any]]:
    """
    Splits page-by-page document text into sentence-aligned chunks
    while preserving page, document, and character metadata.

    Args:
        pages: List of dictionaries from pdf_processor (each with "text", "page", "source").
        chunk_size: Target character length per chunk.
        overlap: Overlap length between adjacent chunks.

    Returns:
        List of structured chunk dictionaries.
    """
    all_chunks: List[Dict[str, Any]] = []
    global_chunk_counter = 1

    for page_info in pages:
        page_text = page_info.get("text", "")
        page_num = page_info.get("page", 1)
        source_name = page_info.get("source", "document.pdf")

        clean_name = source_name.replace(" ", "_").replace(".pdf", "").replace(".", "_")

        raw_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)

        for page_chunk_idx, chunk_content in enumerate(raw_chunks, start=1):
            chunk_id = f"{clean_name}_p{page_num}_c{global_chunk_counter:04d}"

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_content,
                "source": source_name,
                "page": page_num,
                "char_count": len(chunk_content),
                "page_chunk_index": page_chunk_idx
            })
            global_chunk_counter += 1

    return all_chunks
