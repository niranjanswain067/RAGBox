"""
test_suite.py - RAGBox Automated Verification Suite

Executes the 8 mandatory end-to-end tests:
1. Ingestion: PDF -> Text -> Chunks -> Embeddings -> ChromaDB
2. Known Answer QA: Question -> Retrieval -> Context -> Qwen -> Correct Answer
3. Unknown Answer / Hallucination: "I could not find the answer in the uploaded document."
4. Page numbers verification from metadata
5. ChromaDB persistence across restarts
6. Duplicate document processing prevention
7. Clear database operation
8. Missing Qwen model error handling
"""

import sys
import os
import config
from pdf_processor import extract_text_from_pdf
from chunker import chunk_document_pages
from embeddings import generate_embeddings, generate_query_embedding
import vector_store
from retriever import retrieve_relevant_chunks
import rag_pipeline
import llm

PASSED_TESTS = 0
FAILED_TESTS = 0


def log_test(name: str, passed: bool, details: str = ""):
    global PASSED_TESTS, FAILED_TESTS
    if passed:
        PASSED_TESTS += 1
        print(f" [PASS] {name} - {details}")
    else:
        FAILED_TESTS += 1
        print(f" [FAIL] {name} - {details}")


print("=" * 60)
print(" RAGBOX END-TO-END VERIFICATION TEST SUITE ")
print("=" * 60)

# Ensure sample PDF exists
sample_pdf_path = "data/sample_handbook.pdf"
if not os.path.exists(sample_pdf_path):
    print("Error: sample_handbook.pdf not found in data/.")
    sys.exit(1)

# -------------------------------------------------------------
# TEST 7 (Pre-run reset): Verify Clear Database
# -------------------------------------------------------------
print("\n--- Test 7: Clear Database ---")
try:
    vector_store.clear_collection()
    stats_after_clear = vector_store.get_collection_stats()
    is_cleared = (stats_after_clear["total_chunks"] == 0 and stats_after_clear["source_count"] == 0)
    log_test("Test 7: Clear Database", is_cleared, f"Total chunks after reset: {stats_after_clear['total_chunks']}")
except Exception as e:
    log_test("Test 7: Clear Database", False, f"Exception: {e}")


# -------------------------------------------------------------
# TEST 1: Valid PDF Ingestion (PDF -> Text -> Chunks -> Embeddings -> ChromaDB)
# -------------------------------------------------------------
print("\n--- Test 1: Document Ingestion Pipeline ---")
try:
    # 1. Extraction
    extraction = extract_text_from_pdf(sample_pdf_path)
    assert extraction["success"] is True, "PDF extraction failed"
    assert extraction["total_pages"] == 2, f"Expected 2 pages, got {extraction['total_pages']}"
    
    # 2. Chunking
    chunks = chunk_document_pages(extraction["pages"], chunk_size=300, overlap=50)
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    
    # 3. Embedding
    texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(texts)
    assert len(embeddings) == len(chunks), "Embedding count mismatch"
    assert len(embeddings[0]) == 384, f"Expected 384 vector dimensions, got {len(embeddings[0])}"
    
    # 4. Storage in ChromaDB
    added = vector_store.add_documents(chunks, embeddings, doc_hash=extraction["doc_hash"])
    assert added == len(chunks), f"Expected {len(chunks)} chunks stored, got {added}"
    
    stats = vector_store.get_collection_stats()
    assert stats["total_chunks"] == len(chunks), "Collection stats chunk count mismatch"
    
    log_test(
        "Test 1: Valid PDF Ingestion",
        True,
        f"Pages: {extraction['total_pages']}, Chunks: {len(chunks)}, Embeddings: {len(embeddings)}, Stored: {added}"
    )
except Exception as e:
    log_test("Test 1: Valid PDF Ingestion", False, f"Exception: {e}")


# -------------------------------------------------------------
# TEST 6: Duplicate Document Processing Prevention
# -------------------------------------------------------------
print("\n--- Test 6: Duplicate Document Detection ---")
try:
    doc_hash = extraction["doc_hash"]
    is_indexed = vector_store.is_document_indexed(doc_hash)
    log_test(
        "Test 6: Duplicate Document Processing",
        is_indexed is True,
        f"Document hash {doc_hash[:12]}... detected as already indexed: {is_indexed}"
    )
except Exception as e:
    log_test("Test 6: Duplicate Document Processing", False, f"Exception: {e}")


# -------------------------------------------------------------
# TEST 2: Known Answer Question Answering
# -------------------------------------------------------------
print("\n--- Test 2: Known Answer Question ---")
try:
    question = "What is the minimum attendance requirement?"
    qa_result = rag_pipeline.answer_question(question, top_k=2)
    
    answer_text = qa_result["answer"]
    has_attendance_fact = ("75%" in answer_text or "75" in answer_text)
    
    log_test(
        "Test 2: Known Answer QA",
        has_attendance_fact and qa_result["has_answer"],
        f"Question: '{question}' -> Answer: '{answer_text}'"
    )
except Exception as e:
    log_test("Test 2: Known Answer QA", False, f"Exception: {e}")


# -------------------------------------------------------------
# TEST 4: Page Numbers Verification from Metadata
# -------------------------------------------------------------
print("\n--- Test 4: Source Page Number Verification ---")
try:
    sources = qa_result.get("sources", [])
    has_page_one = any(s["page"] == 1 and "sample_handbook.pdf" in s["source"] for s in sources)
    log_test(
        "Test 4: Page Numbers Verification",
        has_page_one,
        f"Sources cited: {sources} (Page 1 confirmed)"
    )
except Exception as e:
    log_test("Test 4: Page Numbers Verification", False, f"Exception: {e}")


# -------------------------------------------------------------
# TEST 3: Unknown Answer / Hallucination Test
# -------------------------------------------------------------
print("\n--- Test 3: Hallucination / Unknown Question Test ---")
try:
    out_of_doc_question = "Who is the Prime Minister of Japan?"
    unknown_result = rag_pipeline.answer_question(out_of_doc_question, top_k=2)
    
    ans = unknown_result["answer"]
    is_grounded_fallback = (
        "could not find the answer" in ans.lower()
        or ans == rag_pipeline.NOT_FOUND_MESSAGE
    )
    no_misleading_sources = len(unknown_result["sources"]) == 0
    
    log_test(
        "Test 3: Hallucination Prevention",
        is_grounded_fallback and no_misleading_sources,
        f"Answer: '{ans}', Sources: {unknown_result['sources']}"
    )
except Exception as e:
    log_test("Test 3: Hallucination Prevention", False, f"Exception: {e}")


# -------------------------------------------------------------
# TEST 5: ChromaDB Persistence Across Restarts
# -------------------------------------------------------------
print("\n--- Test 5: ChromaDB Persistence Across Restarts ---")
try:
    # Simulate application restart by resetting the global in-memory client
    vector_store._CHROMA_CLIENT = None
    fresh_client = vector_store.get_chroma_client()
    fresh_stats = vector_store.get_collection_stats()
    
    is_persisted = (fresh_stats["total_chunks"] == stats["total_chunks"] and stats["total_chunks"] > 0)
    log_test(
        "Test 5: ChromaDB Persistence",
        is_persisted,
        f"Chunks found after client re-instantiation: {fresh_stats['total_chunks']}"
    )
except Exception as e:
    log_test("Test 5: ChromaDB Persistence", False, f"Exception: {e}")


# -------------------------------------------------------------
# TEST 8: Missing Qwen Model Error Handling
# -------------------------------------------------------------
print("\n--- Test 8: Missing Qwen Model Error Handling ---")
try:
    non_existent_path = "models/non_existent_model_file.gguf"
    is_available = llm.is_model_available(non_existent_path)
    assert is_available is False, "Missing model check returned True unexpectedly"
    
    error_caught = False
    try:
        llm.load_llm(model_path=non_existent_path)
    except FileNotFoundError as fnf_err:
        error_caught = True
        err_msg = str(fnf_err)
    
    log_test(
        "Test 8: Missing Model Handling",
        error_caught and ("Qwen model not found" in err_msg),
        "FileNotFoundError with guidance successfully raised."
    )
except Exception as e:
    log_test("Test 8: Missing Model Handling", False, f"Exception: {e}")


# -------------------------------------------------------------
# Summary Report
# -------------------------------------------------------------
print("\n" + "=" * 60)
print(f" TEST SUITE SUMMARY: {PASSED_TESTS} / {PASSED_TESTS + FAILED_TESTS} PASSED")
print("=" * 60)

if FAILED_TESTS == 0:
    print(">>> ALL 8 TESTS PASSED FULLY! RAGBox is 100% verified. <<<")
    sys.exit(0)
else:
    print(f">>> {FAILED_TESTS} TESTS FAILED. <<<")
    sys.exit(1)
