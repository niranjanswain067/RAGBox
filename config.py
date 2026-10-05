"""
RAGBox - Configuration Module

Central configuration file for RAGBox settings.
All model paths, chunking parameters, retrieval parameters, and generation hyperparameters
are defined here to allow easy adjustments without modifying core application logic.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration (Local Qwen 3.0 INT4 GGUF)
# Path to your local GGUF model file
MODEL_PATH = str(MODELS_DIR / "qwen3-int4.gguf")

# LLM Inference Parameters
DEFAULT_TEMPERATURE = 0.2     # Low temperature for factual, grounded answers
DEFAULT_MAX_TOKENS = 512       # Maximum tokens generated in LLM answer
N_CTX = 4096                  # Context window size (input prompt + generated answer)

# CPU Thread configuration for llama.cpp inference
# Default to all physical/logical cores minus 1 (or at least 2) for smooth multitasking
CPU_CORES = os.cpu_count() or 4
DEFAULT_THREADS = max(1, CPU_CORES - 1)

# Embedding Configuration (Local Sentence-Transformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384     # Dimensionality of all-MiniLM-L6-v2 vectors

# Vector Database (ChromaDB) Configuration
COLLECTION_NAME = "ragbox_documents"
PERSIST_DIRECTORY = str(CHROMA_DB_DIR)

# Text Chunking Parameters
DEFAULT_CHUNK_SIZE = 500       # Approximate number of characters per chunk
DEFAULT_CHUNK_OVERLAP = 100    # Character overlap between adjacent chunks

# Retrieval Parameters
DEFAULT_TOP_K = 3              # Number of most similar chunks to retrieve
