"""
llm.py - Local LLM Inference Module

Wraps llama-cpp-python to perform local CPU inference using the user's
quantized Qwen 3.0 INT4 GGUF model. Tracks real-time generation metrics
(tokens generated, duration, and tokens/sec) without third-party frameworks.

Educational Note:
The LLM is the "G" (Generation) in RAG. Unlike typical LLM usage where the model
relies on general pretraining memory (which can hallucinate), RAG conditions the
LLM's probability distribution on the factual text retrieved from the document.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from llama_cpp import Llama
import config

_LLM_INSTANCE: Optional[Llama] = None
_LOADED_MODEL_PATH: Optional[str] = None


def is_model_available(model_path: str = config.MODEL_PATH) -> bool:
    """
    Verifies that the local GGUF model file exists and is accessible.
    """
    path_obj = Path(model_path)
    return path_obj.is_file() and path_obj.stat().st_size > 0


def load_llm(
    model_path: str = config.MODEL_PATH,
    n_ctx: int = config.N_CTX,
    n_threads: int = config.DEFAULT_THREADS
) -> Llama:
    """
    Loads and caches the local GGUF model using llama-cpp-python.
    Reuses the cached instance if the model path has not changed.

    Args:
        model_path: Path to the .gguf model file.
        n_ctx: Context window length in tokens.
        n_threads: Number of CPU threads to allocate for inference.

    Returns:
        Llama model instance.

    Raises:
        FileNotFoundError: If the model file does not exist at model_path.
    """
    global _LLM_INSTANCE, _LOADED_MODEL_PATH

    if not is_model_available(model_path):
        raise FileNotFoundError(
            f"Qwen model not found.\n\n"
            f"Expected location: '{model_path}'\n"
            f"Please ensure your Qwen 3.0 INT4 GGUF model is placed inside the 'models/' folder."
        )

    # Return cached instance if already loaded for the same path
    if _LLM_INSTANCE is not None and _LOADED_MODEL_PATH == model_path:
        return _LLM_INSTANCE

    print(f"[RAGBox] Loading local LLM from: {model_path} (n_ctx={n_ctx}, threads={n_threads})...")
    _LLM_INSTANCE = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False
    )
    _LOADED_MODEL_PATH = model_path
    print("[RAGBox] Local LLM loaded successfully.")
    return _LLM_INSTANCE


def generate_response(
    prompt: str,
    temperature: float = config.DEFAULT_TEMPERATURE,
    max_tokens: int = config.DEFAULT_MAX_TOKENS,
    model_path: str = config.MODEL_PATH
) -> Dict[str, Any]:
    """
    Generates a response from the local Qwen model given an assembled prompt.
    Measures genuine execution time and token statistics from llama.cpp.

    Args:
        prompt: Complete assembled RAG prompt string.
        temperature: Sampling temperature (lower = more deterministic/factual).
        max_tokens: Maximum tokens allowed in the completion.
        model_path: Path to the GGUF model file.

    Returns:
        Dict containing:
            - "text": str (the generated answer text)
            - "prompt_tokens": int
            - "completion_tokens": int
            - "total_tokens": int
            - "generation_time_sec": float
            - "tokens_per_second": float
    """
    llm = load_llm(model_path=model_path)

    start_time = time.perf_counter()

    # ChatML turn boundary stop sequences for Qwen
    stop_tokens = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_start|>"
    ]

    # Call llama-cpp-python completion
    output = llm(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_tokens,
        echo=False
    )

    elapsed_time = max(0.001, time.perf_counter() - start_time)

    # Extract generated text
    choice = output["choices"][0]
    generated_text = choice.get("text", "").strip()

    # Extract real token usage metrics from llama.cpp
    usage = output.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    tokens_per_sec = round(completion_tokens / elapsed_time, 2) if completion_tokens > 0 else 0.0

    return {
        "text": generated_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "generation_time_sec": round(elapsed_time, 2),
        "tokens_per_second": tokens_per_sec
    }
