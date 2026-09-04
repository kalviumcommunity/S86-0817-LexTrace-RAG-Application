import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from prompts.answer import CITATION_SYSTEM_PROMPT, render_citation_prompt
from src.config import config
from src.retrival import get_collection, retrieve

logger = logging.getLogger("rag_service")
FALLBACK_MESSAGE = "I could not find this information in the provided documents."


def get_llm_client() -> OpenAI:
    """Create OpenAI client configured via environment settings."""
    config.validate()
    api_key = config.OPENAI_API_KEY or config.GEMINI_API_KEY
    return OpenAI(
        base_url=config.GEMINI_BASE_URL,
        api_key=api_key
    )


def _compute_similarity_score(distance: float) -> float:
    """Convert vector distance into a normalized similarity score (0.0 to 1.0)."""
    # Chroma returns squared L2 or cosine distance
    similarity = max(0.0, min(1.0, 1.0 - distance))
    return round(similarity, 3)


def guarded_answer(question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    """
    Execute full guarded RAG answering pipeline:
    1. Validates input
    2. Retrieves top-k chunks with metadata, distances, and chunk IDs
    3. Synthesizes grounded answer with citations
    4. Applies guardrails and fallback refusal detection
    5. Returns structured dictionary conforming to QueryResponse shape
    """
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question cannot be empty or only whitespace.")

    k = top_k or config.DEFAULT_TOP_K

    # 1. Retrieve candidate chunks
    retrieval_results = retrieve(
        query=cleaned_question,
        top_k=k
    )

    docs: List[str] = retrieval_results.get("documents", [[]])[0]
    metas: List[Dict[str, Any]] = retrieval_results.get("metadatas", [[]])[0]
    ids: List[str] = retrieval_results.get("ids", [[]])[0]
    distances: List[float] = retrieval_results.get("distances", [[]])[0] if "distances" in retrieval_results else []

    # Map retrieved chunks by source for fast lookup
    retrieved_chunk_records = []
    available_sources: Set[str] = set()

    for idx, (doc, meta) in enumerate(zip(docs, metas)):
        src_name = meta.get("source", "unknown") if meta else "unknown"
        chunk_id = ids[idx] if idx < len(ids) else f"chunk_{idx}"
        dist = distances[idx] if idx < len(distances) else 0.0
        score = _compute_similarity_score(dist)

        available_sources.add(src_name)
        retrieved_chunk_records.append({
            "source": src_name,
            "chunk_id": chunk_id,
            "score": score,
            "text": doc
        })

    # If no documents retrieved at all
    if not docs:
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": [],
            "status": "refused"
        }

    # 2. Format context for prompt
    context_blocks = []
    for item in retrieved_chunk_records:
        context_blocks.append(f"--- Document Source: {item['source']} (Chunk: {item['chunk_id']}) ---\n{item['text']}")
    context_text = "\n\n".join(context_blocks)

    user_prompt = render_citation_prompt(
        context=context_text,
        question=cleaned_question
    )

    client = get_llm_client()

    # 3. Call LLM
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": CITATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )

    raw_content = response.choices[0].message.content or ""
    clean_content = raw_content.strip()

    # Strip markdown code blocks if present
    if clean_content.startswith("```"):
        clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content)
        clean_content = re.sub(r"\s*```$", "", clean_content)
        clean_content = clean_content.strip()

    # 4. Parse response
    answer_text = ""
    cited_source_names: Set[str] = set()

    try:
        parsed = json.loads(clean_content)
        answer_text = parsed.get("answer", "").strip()
        raw_cites = parsed.get("citations", [])
        if isinstance(raw_cites, list):
            cited_source_names = {str(c).strip() for c in raw_cites if str(c).strip()}
        elif isinstance(raw_cites, str) and raw_cites.strip():
            cited_source_names = {raw_cites.strip()}
    except json.JSONDecodeError:
        answer_text = clean_content
        for src in available_sources:
            if src.lower() in answer_text.lower():
                cited_source_names.add(src)
        if not cited_source_names and "could not find" not in answer_text.lower():
            cited_source_names = set(available_sources)

    # 5. Guardrail / Refusal detection
    is_refusal = (
        FALLBACK_MESSAGE.lower() in answer_text.lower() or
        "could not find this information" in answer_text.lower() or
        "not mentioned in the provided" in answer_text.lower()
    )

    if is_refusal:
        return {
            "answer": FALLBACK_MESSAGE,
            "sources": [],
            "status": "refused"
        }

    # Match cited sources with chunk evidence
    structured_sources: List[Dict[str, Any]] = []
    seen_sources = set()

    for item in retrieved_chunk_records:
        if item["source"] in cited_source_names or not cited_source_names:
            if item["source"] not in seen_sources:
                structured_sources.append({
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "score": item["score"]
                })
                seen_sources.add(item["source"])

    # If no structured source matched but citations exist, attach citation names
    if not structured_sources and cited_source_names:
        for c in cited_source_names:
            structured_sources.append({
                "source": c,
                "chunk_id": None,
                "score": None
            })

    return {
        "answer": answer_text,
        "sources": structured_sources,
        "status": "answered"
    }


def get_health_status() -> Dict[str, Any]:
    """Inspect system health, vector database collection, and model configs."""
    collection = get_collection()
    count = collection.count()

    return {
        "status": "healthy",
        "collection_name": config.COLLECTION_NAME,
        "collection_count": count,
        "chat_model": config.CHAT_MODEL,
        "embedding_model": config.EMBEDDING_MODEL
    }
