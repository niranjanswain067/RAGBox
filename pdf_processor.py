"""
pdf_processor.py - Document Text Extraction Module

Extracts text from PDF documents page-by-page using PyMuPDF (fitz).
Preserves page numbers, handles empty or scanned PDFs gracefully,
and computes document hashes for duplicate detection.

Educational Note:
In a RAG pipeline, high-quality text extraction is the foundational first step.
If document text is corrupted or page numbers are lost, the downstream retrieval
and source attribution cannot be accurate.
"""

import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import pymupdf as fitz
except ImportError:
    import fitz


def compute_file_hash(file_bytes: bytes) -> str:
    """
    Computes a SHA-256 hash of document bytes.
    Used to detect if a document has already been indexed in ChromaDB
    to avoid redundant processing.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def extract_text_from_pdf(
    file_source: Union[str, Path, bytes, io.BytesIO],
    filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extracts text page-by-page from a PDF document.

    Args:
        file_source: File path (str/Path), raw bytes, or BytesIO object.
        filename: Optional human-readable filename (used when file_source is bytes).

    Returns:
        Dict containing:
            - "success": bool
            - "source": str (filename)
            - "doc_hash": str (SHA-256)
            - "total_pages": int
            - "pages": List of dicts, each with {"text": str, "page": int, "source": str}
            - "error": Optional error message if extraction fails.

    Raises:
        ValueError if the PDF cannot be opened or contains no readable text.
    """
    doc = None
    file_bytes = b""
    source_name = filename or "document.pdf"

    try:
        # Determine source type and load PyMuPDF document
        if isinstance(file_source, (str, Path)):
            path_obj = Path(file_source)
            source_name = filename or path_obj.name
            with open(path_obj, "rb") as f:
                file_bytes = f.read()
            doc = fitz.open(stream=file_bytes, filetype="pdf")

        elif isinstance(file_source, bytes):
            file_bytes = file_source
            doc = fitz.open(stream=file_bytes, filetype="pdf")

        elif hasattr(file_source, "read"):
            # Handles Streamlit's UploadedFile or io.BytesIO
            file_bytes = file_source.read()
            if hasattr(file_source, "seek"):
                file_source.seek(0)  # Reset pointer for subsequent reads
            if hasattr(file_source, "name") and not filename:
                source_name = file_source.name
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        else:
            raise ValueError("Unsupported file source type provided to PDF processor.")

        if doc is None or doc.page_count == 0:
            raise ValueError(f"The PDF file '{source_name}' is empty or could not be read.")

        doc_hash = compute_file_hash(file_bytes)
        total_pages = doc.page_count
        extracted_pages: List[Dict[str, Any]] = []
        total_text_length = 0

        # Extract text page-by-page (1-indexed for students and user-facing clarity)
        for page_num in range(1, total_pages + 1):
            page = doc.load_page(page_num - 1)
            text = page.get_text("text").strip()

            # Clean up excessive whitespace/null bytes while preserving sentence flow
            cleaned_text = " ".join(text.split())

            if cleaned_text:
                total_text_length += len(cleaned_text)
                extracted_pages.append({
                    "text": cleaned_text,
                    "page": page_num,
                    "source": source_name,
                    "char_count": len(cleaned_text)
                })

        # Check for scanned or image-only PDFs that contain no extractable text
        if total_text_length == 0 or len(extracted_pages) == 0:
            raise ValueError(
                f"No readable text was found in '{source_name}'. "
                "This document may be scanned or image-based without an OCR text layer."
            )

        return {
            "success": True,
            "source": source_name,
            "doc_hash": doc_hash,
            "total_pages": total_pages,
            "readable_pages": len(extracted_pages),
            "pages": extracted_pages,
            "error": None
        }

    except Exception as exc:
        return {
            "success": False,
            "source": source_name,
            "doc_hash": compute_file_hash(file_bytes) if file_bytes else "",
            "total_pages": 0,
            "readable_pages": 0,
            "pages": [],
            "error": str(exc)
        }

    finally:
        if doc is not None:
            doc.close()
