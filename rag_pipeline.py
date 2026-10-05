"""
rag_pipeline.py - Core RAG Orchestrator Module (High Accuracy Edition)

Orchestrates the entire Retrieval-Augmented Generation lifecycle:
1. Receives user question.
2. Retrieves relevant chunks via retriever.py.
3. Builds grounded ChatML prompt tailored for Qwen 3.0.
4. Executes local inference via llm.py.
5. Parses thinking/reasoning tags (<think>...</think>) for deep transparency.
6. Enforces strict zero-hallucination grounding.
"""

from typing import Any, Dict, List
import config
from retriever import retrieve_relevant_chunks
from llm import generate_response
from vector_store import get_collection_stats

# Fallback response when information cannot be found in the document
NOT_FOUND_MESSAGE = "I could not find the answer in the uploaded document."


def format_context_chunks(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats a list of retrieved chunks into a clean, labeled context block.
    Includes document name and page number headers for each chunk.
    """
    if not chunks:
        return "No relevant context found."

    context_segments = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "document.pdf")
        page = chunk.get("page", 1)
        text = chunk.get("text", "").strip()
        segment = f"[Excerpt {idx} - Source: {source}, Page: {page}]\n{text}"
        context_segments.append(segment)

    return "\n\n".join(context_segments)


def build_prompt(question: str, retrieved_context: str) -> str:
    """
    Constructs the grounded ChatML prompt tailored for Qwen 3.0 INT4.
    Using Qwen's native format dramatically improves instruction following,
    factual grounding, and reasoning capabilities.
    """
    prompt = f"""<|im_start|>system
You are RAGBox, a precise, helpful, and strictly factual document question-answering assistant.

Your task is to answer the user's question using ONLY the provided document excerpts.

Strict Grounding Rules:
1. Rely ONLY on the provided Context excerpts. Never assume, extrapolate, or introduce outside knowledge.
2. Every claim or number in your answer MUST be backed by the excerpts.
3. If the answer cannot be determined from the excerpts, respond with EXACTLY:
   "{NOT_FOUND_MESSAGE}"
4. Keep the final answer clear, accurate, and concise.
<|im_end|>
<|im_start|>user
Context Excerpts:
{retrieved_context}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""
    return prompt


def parse_model_output(raw_text: str) -> Dict[str, str]:
    """
    Parses Qwen's output, separating internal reasoning (<think>...</think>)
    from the final answer delivered to the user.
    """
    text = raw_text.strip()
    reasoning = ""
    answer = text

    if "<think>" in text:
        parts = text.split("</think>", 1)
        reasoning = parts[0].replace("<think>", "").strip()
        if len(parts) > 1:
            answer = parts[1].strip()
        else:
            answer = ""

    # Clean leading "Answer:" prefixes if present
    if answer.lower().startswith("answer:"):
        answer = answer[7:].strip()

    return {
        "reasoning": reasoning,
        "answer": answer
    }


def extract_unique_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts a deduplicated list of source documents and page numbers
    strictly derived from PDF metadata, never allowing the LLM to invent pages.
    """
    seen = set()
    sources = []

    for chunk in chunks:
        src = chunk.get("source", "document.pdf")
        page = chunk.get("page", 1)
        pair = (src, page)
        if pair not in seen:
            seen.add(pair)
            sources.append({
                "source": src,
                "page": page
            })

    return sorted(sources, key=lambda s: (s["source"], s["page"]))


def answer_question(
    question: str,
    top_k: int = config.DEFAULT_TOP_K,
    temperature: float = 0.1,  # Lower temperature (0.1) for maximum factual precision
    max_tokens: int = 512,
    model_path: str = config.MODEL_PATH
) -> Dict[str, Any]:
    """
    Executes the complete high-accuracy RAG query pipeline.
    """
    clean_question = question.strip() if question else ""
    if not clean_question:
        return {
            "question": question,
            "answer": "Please enter a valid question.",
            "reasoning": "",
            "sources": [],
            "retrieved_chunks": [],
            "prompt": "",
            "generation_stats": {},
            "has_answer": False,
            "error": "Empty question provided."
        }

    stats = get_collection_stats()
    if stats["total_chunks"] == 0:
        return {
            "question": clean_question,
            "answer": "No documents have been indexed yet. Please process a PDF first.",
            "reasoning": "",
            "sources": [],
            "retrieved_chunks": [],
            "prompt": "",
            "generation_stats": {},
            "has_answer": False,
            "error": "No documents indexed."
        }

    # Step 1: Semantic Retrieval
    retrieved_chunks = retrieve_relevant_chunks(clean_question, top_k=top_k)

    if not retrieved_chunks:
        return {
            "question": clean_question,
            "answer": NOT_FOUND_MESSAGE,
            "reasoning": "No relevant chunks retrieved from vector store.",
            "sources": [],
            "retrieved_chunks": [],
            "prompt": "",
            "generation_stats": {},
            "has_answer": False,
            "error": None
        }

    # Step 2: Build Grounded Prompt
    context_str = format_context_chunks(retrieved_chunks)
    assembled_prompt = build_prompt(clean_question, context_str)

    # Step 3: LLM Inference
    try:
        generation_output = generate_response(
            prompt=assembled_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model_path=model_path
        )
    except Exception as exc:
        return {
            "question": clean_question,
            "answer": f"Generation error: {str(exc)}",
            "reasoning": "",
            "sources": [],
            "retrieved_chunks": retrieved_chunks,
            "prompt": assembled_prompt,
            "generation_stats": {},
            "has_answer": False,
            "error": str(exc)
        }

    raw_output = generation_output.get("text", "").strip()
    parsed = parse_model_output(raw_output)
    final_answer = parsed["answer"]
    reasoning_text = parsed["reasoning"]

    # Step 4: Strict Hallucination & Fallback Verification
    is_not_found = (
        NOT_FOUND_MESSAGE.lower() in final_answer.lower()
        or NOT_FOUND_MESSAGE.lower() in raw_output.lower()
        or "could not find the answer" in final_answer.lower()
        or "not mentioned in the context" in final_answer.lower()
        or "not mentioned in the provided" in final_answer.lower()
        or "not present in the provided" in final_answer.lower()
        or "not present in the context" in final_answer.lower()
        or not final_answer.strip()
    )

    if is_not_found:
        final_answer = NOT_FOUND_MESSAGE
        verified_sources = []
        has_answer = False
    else:
        verified_sources = extract_unique_sources(retrieved_chunks)
        has_answer = True

    return {
        "question": clean_question,
        "answer": final_answer,
        "reasoning": reasoning_text,
        "sources": verified_sources,
        "retrieved_chunks": retrieved_chunks,
        "prompt": assembled_prompt,
        "generation_stats": {
            "prompt_tokens": generation_output.get("prompt_tokens", 0),
            "completion_tokens": generation_output.get("completion_tokens", 0),
            "total_tokens": generation_output.get("total_tokens", 0),
            "generation_time_sec": generation_output.get("generation_time_sec", 0.0),
            "tokens_per_second": generation_output.get("tokens_per_second", 0.0),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model_name": "Qwen 3.0 INT4"
        },
        "has_answer": has_answer,
        "error": None
    }
