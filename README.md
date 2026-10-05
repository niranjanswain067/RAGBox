# 📦 RAGBox — Local Document AI Assistant

An educational, beginner-friendly **Retrieval-Augmented Generation (RAG)** application built from scratch to teach students and developers how RAG works internally.

> **Zero Cloud APIs. Zero Framework Black Boxes.**  
> Built strictly **without LangChain or LlamaIndex**, ensuring every single step—from PDF parsing to vector search and LLM inference—is transparent, modular, and easy to understand.

---

## 🎯 What is RAGBox?

**RAGBox** allows you to upload any PDF document, process its text, store its semantic representations into a local vector database, and ask questions about its contents.

Answering is performed entirely on your computer using:
- **PyMuPDF (`fitz`)** for clean page-by-page text extraction.
- **`sentence-transformers/all-MiniLM-L6-v2`** for generating local semantic embeddings.
- **ChromaDB** for persistent local vector similarity indexing.
- **Qwen 3.0 INT4 GGUF** executed locally on CPU via **`llama-cpp-python`**.
- **Streamlit** for an interactive, modern user interface featuring a deep **5-Step Teaching Mode**.

---

## 🧠 What is RAG?

Standard Large Language Models (LLMs) rely purely on their static pre-training data. When asked about private documents or recent information, they either guess or hallucinate.

**Retrieval-Augmented Generation (RAG)** solves this by retrieving relevant excerpts from your document first, and then injecting those excerpts into the LLM prompt as context.

```text
Traditional LLM:
Question ────► [ LLM Pre-trained Memory ] ────► Answer (May Hallucinate)

RAG Pipeline:
Question ────► [ ChromaDB Vector Search ] ────► Relevant Excerpts
                                                       │
                                                       ▼
Question + Relevant Excerpts ────► [ Local Qwen LLM ] ────► Grounded Fact-Based Answer
```

---

## 🏛️ RAGBox Architecture

Here is the exact step-by-step pipeline implemented across RAGBox:

```text
                    ┌─────────────────────────┐
                    │      PDF Document       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     pdf_processor.py    │ (PyMuPDF extracts page text & metadata)
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │        chunker.py       │ (Sliding-window text segmentation)
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      embeddings.py      │ (all-MiniLM-L6-v2 vectorization)
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     vector_store.py     │ (Persistent ChromaDB HNSW Index)
                    └────────────┬────────────┘
                                 │
                         User Question
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       retriever.py      │ (Question vectorization & Top-K search)
                    └────────────┬────────────┘
                                 │
                          Relevant Chunks
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     rag_pipeline.py     │ (Prompt construction & grounding rules)
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │          llm.py         │ (Local Qwen 3.0 INT4 inference)
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    app.py (Streamlit)   │ (Answer, Source citations, Teaching Mode)
                    └─────────────────────────┘
```

---

## 📂 Project Structure

```text
RAGBox/
│
├── app.py              # Streamlit Web UI & 5-Step Teaching Mode Inspector
├── config.py           # Central configuration (paths, chunk sizes, Top-K, model settings)
├── pdf_processor.py    # PyMuPDF text extractor with page tracking & empty PDF guards
├── chunker.py          # Custom sliding-window chunker preserving chunk IDs and metadata
├── embeddings.py       # Local Sentence-Transformers embedding generator with singleton caching
├── vector_store.py     # Persistent ChromaDB client (CRUD, deduplication, and statistics)
├── retriever.py        # Semantic search querying ChromaDB with question embeddings
├── llm.py              # llama-cpp-python wrapper with real CPU execution telemetry
├── rag_pipeline.py     # Grounded prompt builder and end-to-end RAG orchestrator
│
├── requirements.txt    # Minimal pinned dependencies (no LangChain / LlamaIndex)
├── README.md           # This comprehensive guide
├── .gitignore          # Git exclusion rules
│
├── data/
│   └── sample_handbook.pdf  # Sample college handbook for testing
│
├── models/
│   ├── README.md            # Model placement instructions
│   └── qwen3-int4.gguf      # Local Qwen 3.0 INT4 GGUF model
│
└── chroma_db/               # Persistent on-disk vector database directory
```

---

## 🧩 Module Breakdown & Responsibilities

| File | Primary Responsibility |
|---|---|
| **`config.py`** | Central settings file storing default parameters (`CHUNK_SIZE = 500`, `CHUNK_OVERLAP = 100`, `TOP_K = 3`, `TEMPERATURE = 0.2`, `MODEL_PATH = "models/qwen3-int4.gguf"`). |
| **`pdf_processor.py`** | Extracts text page-by-page using PyMuPDF (`fitz`), calculates document SHA-256 hashes to detect duplicates, and gracefully handles scanned/image-only PDFs. |
| **`chunker.py`** | Splits long text into overlapping windows without third-party frameworks. Retains metadata (`source`, `page`, `chunk_id`) on every chunk. |
| **`embeddings.py`** | Uses `sentence-transformers/all-MiniLM-L6-v2` locally to convert text strings into 384-dimensional dense semantic vectors. |
| **`vector_store.py`** | Connects to a persistent ChromaDB instance at `./chroma_db`, indexing chunk vectors using cosine distance and providing collection statistics. |
| **`retriever.py`** | Embeds the user's question, queries ChromaDB, and returns the Top-K closest chunks with genuine similarity and distance metrics. |
| **`llm.py`** | Loads `models/qwen3-int4.gguf` using `llama-cpp-python` on CPU and records real metrics (prompt tokens, completion tokens, time in seconds, and tokens/sec). |
| **`rag_pipeline.py`** | Formats retrieved chunks into the strict grounding prompt, executes Qwen, and enforces the fallback rule (*"I could not find the answer in the uploaded document."*). |
| **`app.py`** | Streamlit web interface containing settings sliders, live document statistics, answer cards, verified source attribution, and interactive Teaching Mode. |

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13.
- Normal laptop with 8 GB to 16 GB RAM (runs smoothly on CPU).

### 2. Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Place Your Local Qwen Model
Ensure your local `Qwen 3.0 INT4` GGUF model file is placed inside the `models/` directory:
```text
RAGBox/
└── models/
    └── qwen3-int4.gguf
```
*(If your model file has a different name, simply update `MODEL_PATH` in `config.py`)*.

---

## 💻 Running the Application

Launch the RAGBox local application:
```bash
python server.py
```

Then open your browser at:
```text
http://localhost:8000
```

The application serves a fast, clean, and modern **HTML5, CSS, and Vanilla JavaScript** interface with live 5-Step Teaching Mode telemetry.

---

## 🎓 How RAGBox Works (A Concrete Example)

Suppose you upload `data/sample_handbook.pdf` which contains:
> *"Students must maintain a minimum attendance of 75% in each subject to be eligible for end-semester examinations."*

When you ask:
> *"What is the minimum attendance requirement?"*

Here is the exact pipeline execution:
1. **Embedding**: The sentence-transformers model converts your question into a 384-dimensional vector.
2. **Retrieval**: ChromaDB compares the question vector against all document chunk vectors and finds that Chunk 1 (Page 1) has the highest cosine similarity (~0.64).
3. **Prompt Construction**: RAGBox dynamically builds this grounded prompt:
   ```text
   You are RAGBox, a document question-answering assistant.

   Answer the user's question using ONLY the provided context.

   Rules:
   1. Do not use outside knowledge.
   2. Do not invent facts.
   3. If the answer is not present in the context, say:
      "I could not find the answer in the uploaded document."
   4. Keep the answer clear and concise.

   Context:
   [Excerpt 1 - Source: sample_handbook.pdf, Page: 1]
   Apex University College Handbook Section 1: Academic Policies Students must maintain a minimum attendance of 75% in each subject to be eligible for end-semester examinations.

   Question:
   What is the minimum attendance requirement?

   Answer:
   ```
4. **Generation**: Qwen 3.0 INT4 reads the prompt and produces:
   > **`75%`**
5. **Attribution**: RAGBox displays the answer along with verified citations:
   > 📄 `sample_handbook.pdf` (Page 1)

---

## 🛡️ Grounding & Hallucination Resistance

If you ask an out-of-document question such as:
> *"Who is the Prime Minister of Japan?"*

Because the document does not contain Japanese political information, RAGBox strictly outputs:
> **`I could not find the answer in the uploaded document.`**

It will **not** attempt to use general world knowledge to guess. This demonstrates **RAG grounding and hallucination prevention** to students.

---

## 🧪 Testing Checklist

RAGBox includes built-in verification for:
- [x] Multi-page PDF text extraction with 1-indexed page tracking.
- [x] Sliding-window text chunking with deterministic chunk IDs.
- [x] Local vectorization via `sentence-transformers`.
- [x] ChromaDB persistent storage and duplicate document detection.
- [x] High-precision semantic retrieval with genuine cosine scores.
- [x] CPU inference on local Qwen 3.0 INT4 GGUF.
- [x] Zero-hallucination fallback for absent information.
- [x] Interactive 5-Step Teaching Mode diagnostics.
