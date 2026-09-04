import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from openai import OpenAI

from prompts.answer import CITATION_SYSTEM_PROMPT, render_citation_prompt
from src.retrival import retrieve

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "I could not find this information in the provided documents."


def get_llm_client() -> OpenAI:
    """Initialize and return OpenAI client pointing to Gemini endpoint."""
    api_key = os.getenv("GEMINI_API_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env")
    if not base_url:
        raise ValueError("GEMINI_BASE_URL is missing from .env")

    return OpenAI(base_url=base_url, api_key=api_key)


def format_context(documents: List[str], metadatas: List[Dict[str, Any]]) -> str:
    """Format retrieved document chunks with clear source headers."""
    formatted_chunks = []
    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "unknown")
        formatted_chunks.append(f"--- Document Source: {source} ---\n{doc}")
    return "\n\n".join(formatted_chunks)


def _parse_llm_response(raw_content: str, available_sources: Set[str]) -> Dict[str, Any]:
    """Parse JSON response from LLM, handling markdown fences or fallback text."""
    clean_content = raw_content.strip()

    # Strip markdown code blocks if present
    if clean_content.startswith("```"):
        clean_content = re.sub(r"^```(?:json)?\s*", "", clean_content)
        clean_content = re.sub(r"\s*```$", "", clean_content)
        clean_content = clean_content.strip()

    try:
        parsed = json.loads(clean_content)
        answer = parsed.get("answer", "").strip()
        raw_citations = parsed.get("citations", [])

        # Normalize citations to set of strings
        if isinstance(raw_citations, list):
            citations = {str(c).strip() for c in raw_citations if str(c).strip()}
        elif isinstance(raw_citations, str):
            citations = {raw_citations.strip()} if raw_citations.strip() else set()
        else:
            citations = set()

        # If answer is a refusal / fallback, clear citations
        if FALLBACK_MESSAGE.lower() in answer.lower() or "could not find" in answer.lower():
            citations = set()

        return {"answer": answer, "citations": citations}

    except json.JSONDecodeError:
        # Fallback parsing if LLM returned plain text
        logger.warning("LLM response was not valid JSON. Using text fallback.")
        answer = clean_content

        # Extract any mentioned sources from text
        citations = set()
        if "could not find" not in answer.lower():
            for src in available_sources:
                if src.lower() in answer.lower():
                    citations.add(src)
            # If no explicit source mentions but valid answer, attribute to top available source
            if not citations and available_sources:
                citations = set(available_sources)

        return {"answer": answer, "citations": citations}


def answer_with_citations(
    question: str,
    top_k: int = 3,
    metadata_filter: Optional[Dict[str, Any]] = None,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute full RAG pipeline:
    1. Retrieve relevant chunks from ChromaDB
    2. Construct citation-aware prompt
    3. Generate grounded answer via LLM
    4. Return structured response with answer and citations
    """
    if client is None:
        client = get_llm_client()

    if model is None:
        model = os.getenv("CHAT_MODEL", "gemini-3.1-flash-lite")

    # 1. Retrieve relevant documents
    retrieval_results = retrieve(
        query=question,
        top_k=top_k,
        metadata_filter=metadata_filter
    )

    docs = retrieval_results.get("documents", [[]])[0]
    metas = retrieval_results.get("metadatas", [[]])[0]
    distances = retrieval_results.get("distances", [[]])[0] if "distances" in retrieval_results else []

    retrieved_sources: Set[str] = {
        meta.get("source", "unknown")
        for meta in metas
        if meta and "source" in meta
    }

    if not docs:
        return {
            "question": question,
            "answer": FALLBACK_MESSAGE,
            "citations": set(),
            "retrieved_sources": set(),
            "retrieved_contexts": [],
            "distances": []
        }

    # 2. Format context with source headers
    context_text = format_context(docs, metas)

    # 3. Render prompt and call LLM
    user_prompt = render_citation_prompt(
        context=context_text,
        question=question
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CITATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        raw_response = response.choices[0].message.content or ""
        parsed = _parse_llm_response(raw_response, retrieved_sources)

        return {
            "question": question,
            "answer": parsed["answer"],
            "citations": parsed["citations"],
            "retrieved_sources": retrieved_sources,
            "retrieved_contexts": docs,
            "distances": distances,
            "raw_response": raw_response
        }

    except Exception as e:
        logger.error(f"Error during LLM answer generation: {e}")
        return {
            "question": question,
            "answer": f"Error generating answer: {e}",
            "citations": set(),
            "retrieved_sources": retrieved_sources,
            "retrieved_contexts": docs,
            "distances": distances,
            "error": str(e)
        }


if __name__ == "__main__":
    sample_q = "When can the agreement be terminated?"
    print(f"\nTesting RAG pipeline with question: {sample_q}")
    result = answer_with_citations(sample_q)
    print("\nAnswer:", result["answer"])
    print("Citations:", result["citations"])
    print("Retrieved Sources:", result["retrieved_sources"])
