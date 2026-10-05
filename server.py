"""
server.py - RAGBox FastAPI Backend Server

Serves the REST API for RAG operations and hosts the modern HTML/CSS/JS frontend.
Clean, lightweight, fast, and completely free of Streamlit dependencies.
"""

import os
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import config
from pdf_processor import extract_text_from_pdf
from chunker import chunk_document_pages
from embeddings import generate_embeddings, load_embedding_model
import vector_store
from rag_pipeline import answer_question, NOT_FOUND_MESSAGE
from llm import is_model_available

# Initialize FastAPI application
app = FastAPI(
    title="RAGBox API",
    description="Educational Local Document AI Assistant API",
    version="2.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = config.BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = config.DEFAULT_TOP_K
    temperature: Optional[float] = config.DEFAULT_TEMPERATURE
    max_tokens: Optional[int] = config.DEFAULT_MAX_TOKENS


# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.get("/api/health")
def health_check():
    """Returns system status and model readiness."""
    stats = vector_store.get_collection_stats()
    return {
        "status": "online",
        "model_available": is_model_available(),
        "model_path": config.MODEL_PATH,
        "indexed_chunks": stats["total_chunks"],
        "indexed_sources": stats["unique_sources"],
        "source_count": stats["source_count"]
    }


@app.get("/api/stats")
def get_stats():
    """Returns database statistics."""
    stats = vector_store.get_collection_stats()
    return stats


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(config.DEFAULT_CHUNK_SIZE),
    chunk_overlap: int = Form(config.DEFAULT_CHUNK_OVERLAP)
):
    """
    Extracts text from uploaded PDF, chunks it, generates embeddings,
    and stores vectors in ChromaDB.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Step 1: Text extraction
    extraction = extract_text_from_pdf(file_bytes, filename=file.filename)
    if not extraction["success"]:
        raise HTTPException(status_code=400, detail=extraction["error"])

    doc_hash = extraction["doc_hash"]

    # Step 2: Check duplicate
    if vector_store.is_document_indexed(doc_hash):
        stats = vector_store.get_collection_stats()
        return {
            "status": "already_indexed",
            "message": f"'{file.filename}' is already indexed in ChromaDB.",
            "source": file.filename,
            "total_pages": extraction["total_pages"],
            "stats": stats
        }

    # Step 3: Chunking
    chunks = chunk_document_pages(
        extraction["pages"],
        chunk_size=chunk_size,
        overlap=chunk_overlap
    )

    # Step 4: Vectorization
    chunk_texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)

    # Step 5: ChromaDB Storage
    added_count = vector_store.add_documents(chunks, embeddings, doc_hash=doc_hash)
    stats = vector_store.get_collection_stats()

    return {
        "status": "success",
        "message": f"Successfully indexed '{file.filename}'.",
        "source": file.filename,
        "total_pages": extraction["total_pages"],
        "chunks_created": len(chunks),
        "embeddings_generated": len(embeddings),
        "chunks_stored": added_count,
        "stats": stats
    }


@app.post("/api/ask")
def ask_question_endpoint(req: QuestionRequest):
    """
    Receives user question, runs semantic retrieval, constructs prompt,
    generates answer via Qwen 3.0 INT4, and returns full teaching telemetry.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    stats = vector_store.get_collection_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been indexed yet. Please upload a PDF first."
        )

    if not is_model_available():
        raise HTTPException(
            status_code=500,
            detail=f"Qwen model not found at '{config.MODEL_PATH}'."
        )

    result = answer_question(
        question=req.question,
        top_k=req.top_k or config.DEFAULT_TOP_K,
        temperature=req.temperature or config.DEFAULT_TEMPERATURE,
        max_tokens=req.max_tokens or config.DEFAULT_MAX_TOKENS,
        model_path=config.MODEL_PATH
    )

    return result


@app.post("/api/clear")
def clear_database_endpoint():
    """Resets the ChromaDB collection."""
    vector_store.clear_collection()
    return {"status": "success", "message": "ChromaDB collection cleared successfully."}


# -------------------------------------------------------------
# Static Web Frontend
# -------------------------------------------------------------

# Mount static folder
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serves the main HTML page."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse({"message": "Frontend not built yet."}, status_code=404)
    return FileResponse(str(index_file))


if __name__ == "__main__":
    # Pre-cache embedding model on startup
    try:
        load_embedding_model()
    except Exception as e:
        print(f"[RAGBox] Warning during embedding model load: {e}")

    print("\n" + "=" * 50)
    print(" [RAGBox] Local Server Starting...")
    print(" Access Web UI at: http://localhost:8000")
    print("=" * 50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
