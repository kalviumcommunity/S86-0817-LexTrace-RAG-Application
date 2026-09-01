"""
Grounded Generation: Generating Answers from Retrieved Context Only

This module handles the final RAG step: generating answers that are grounded
in retrieved context rather than model hallucination.

Key principles:
  1. Generate answers using ONLY injected context
  2. Verify answers reflect retrieved chunks accurately
  3. Return fallback when supporting context is missing
  4. Enable comparison between grounded and ungrounded answers
  5. Reduce hallucination by explicit grounding directives

A grounded answer is traceable to the retrieved chunks. If a sentence cannot
be supported by the context, it should not appear in the answer.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from src.prompt_augmentation import build_augmented_prompt, format_chunk
from src.retrival import retrieve

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")

# Guardrail thresholds for retrieval quality.
# Start conservative and tune against real evaluation data.
MIN_TOP_SCORE = 0.72
MIN_SUPPORTING_CHUNKS = 1

# Validate required environment variables
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")
if not GEMINI_BASE_URL:
    raise ValueError("GEMINI_BASE_URL is missing from .env")

# Initialize LLM client
llm_client = OpenAI(
    base_url=GEMINI_BASE_URL,
    api_key=GEMINI_API_KEY,
)

# Grounding-focused system prompt
GROUNDING_SYSTEM_PROMPT = """You are a grounded assistant for legal document analysis.

CRITICAL RULES:
1. Answer ONLY using the provided context.
2. Do NOT use external knowledge, general facts, or information not in the context.
3. If the question cannot be answered from the context, respond with:
   "I don't have enough information in the provided context to answer this question."
4. When possible, cite your sources using markers like [1], [2], etc.
5. Be accurate, concise, and directly address the question.
6. If the context is incomplete or contradictory, acknowledge it.

Remember: Grounded answers are better than confident hallucinations."""


def retrieval_is_strong(chunks: List[Dict[str, Any]]) -> bool:
    """Return True when retrieved context is strong enough for generation."""
    if not chunks:
        return False

    strong_chunks = [
        chunk for chunk in chunks if chunk.get("score", 0) >= MIN_TOP_SCORE
    ]
    return len(strong_chunks) >= MIN_SUPPORTING_CHUNKS


def call_llm(prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
    """Small wrapper for prompt-based LLM calls used in rewriting."""
    messages = [{"role": "user", "content": prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = llm_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def rewrite_followup(history: List[Dict[str, str]], question: str) -> str:
    """Rewrite a follow-up question into a standalone query using recent history."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    history_text = "\n".join(
        f"{turn.get('role', 'user')}: {turn.get('content', '')}"
        for turn in (history or [])
    )

    prompt = f"""
Rewrite the user's latest question as a standalone search query.
Use the conversation history only to resolve references.
Do not answer the question.

History:
{history_text}

Latest question:
{question}
"""
    return call_llm(prompt, system_prompt="You rewrite follow-up questions into standalone retrieval queries.").strip()


def guarded_answer(
    question: str,
    k: int = 4,
    metadata_filter: Optional[Dict] = None,
    temperature: float = 0.3,
    fallback_message: Optional[str] = None,
) -> Dict[str, Any]:
    """Guardrails: refuse weak or missing context before generation."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    chunks = retrieve(question, top_k=k, metadata_filter=metadata_filter)

    if not retrieval_is_strong(chunks):
        refusal = fallback_message or "I don't have enough reliable context to answer that."
        return {
            "answer": refusal,
            "question": question,
            "sources": [],
            "context": "",
            "sources_count": 0,
            "status": "refused_weak_context"
        }

    result = generate_grounded_answer(question, chunks, temperature=temperature)
    result["status"] = "answered"
    return result


def conversational_answer(
    history: List[Dict[str, str]],
    user_question: str,
    k: int = 4,
    metadata_filter: Optional[Dict] = None,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Rewrite a follow-up question to a standalone query and answer with retrieval guardrails."""
    rewritten_query = rewrite_followup(history, user_question)
    chunks = retrieve(rewritten_query, top_k=k, metadata_filter=metadata_filter)

    if not retrieval_is_strong(chunks):
        answer = "I don't have enough reliable context to answer that."
    else:
        answer = generate_grounded_answer(user_question, chunks, temperature=temperature)["answer"]

    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": answer})

    return {
        "rewritten_query": rewritten_query,
        "answer": answer,
        "sources": [chunk.get("metadata", {}) for chunk in chunks],
        "status": "answered" if retrieval_is_strong(chunks) else "refused_weak_context",
    }


def generate_grounded_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Generate an answer grounded ONLY in retrieved context.
    
    This is the core grounded generation function. It:
    1. Takes a question and retrieved chunks
    2. Builds an augmented prompt with grounding directives
    3. Calls the LLM with explicit instructions to use only context
    4. Returns the answer with source metadata
    
    Args:
        question: The user's question
        retrieved_chunks: Chunks from retrieval stage
        system_prompt: Custom system prompt (uses grounding prompt if None)
        temperature: LLM temperature (default 0.3 for consistency)
        max_tokens: Maximum tokens in response
        
    Returns:
        Dictionary with keys:
        - answer: Generated answer grounded in context
        - question: Original question
        - sources: List of source metadata used
        - context: The augmented prompt context used
        - sources_count: Number of sources included
        - status: "success" or "no_context"
        
    Raises:
        ValueError: If question is empty
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    logger.info(f"Generating grounded answer for: {question[:60]}...")
    
    # Handle missing context
    if not retrieved_chunks:
        logger.warning("No retrieved chunks provided")
        return {
            "answer": "I don't have enough information in the provided context to answer this question.",
            "question": question,
            "sources": [],
            "context": "",
            "sources_count": 0,
            "status": "no_context"
        }
    
    # Use provided or default grounding prompt
    if system_prompt is None:
        system_prompt = GROUNDING_SYSTEM_PROMPT
    
    # Build augmented prompt with retrieved context
    try:
        prompt_data = build_augmented_prompt(
            question,
            retrieved_chunks,
            system_instructions=system_prompt,
            include_token_info=False
        )
    except Exception as e:
        logger.error(f"Failed to build augmented prompt: {e}")
        raise
    
    # Call LLM with grounded prompt
    try:
        logger.info(f"Calling LLM with {prompt_data['included_chunks']} sources...")
        
        response = llm_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt_data["prompt"]
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        answer = response.choices[0].message.content
        logger.info(f"Grounded answer generated ({len(answer)} characters)")
        
        return {
            "answer": answer,
            "question": question,
            "sources": prompt_data["sources_used"],
            "context": prompt_data["context"],
            "sources_count": prompt_data["included_chunks"],
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


def generate_ungrounded_answer(
    question: str,
    temperature: float = 0.7,
    max_tokens: int = 1024
) -> Dict[str, Any]:
    """
    Generate an answer WITHOUT context (for comparison).
    
    Used to demonstrate the difference between hallucination and grounding.
    The model answers from its training data without any external context.
    
    Args:
        question: The user's question
        temperature: LLM temperature (default 0.7 for more creativity)
        max_tokens: Maximum tokens in response
        
    Returns:
        Dictionary with keys:
        - answer: Generated answer without context
        - question: Original question
        - sources: Empty list (no grounding)
        - status: "ungrounded"
        
    Raises:
        ValueError: If question is empty
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    logger.info(f"Generating ungrounded answer (for comparison)...")
    
    prompt = f"""Answer the following question based on your general knowledge.

Question: {question}"""
    
    try:
        response = llm_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        answer = response.choices[0].message.content
        logger.info(f"Ungrounded answer generated ({len(answer)} characters)")
        
        return {
            "answer": answer,
            "question": question,
            "sources": [],
            "status": "ungrounded"
        }
        
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


def answer_query(
    question: str,
    k: int = 4,
    metadata_filter: Optional[Dict] = None,
    temperature: float = 0.3,
    fallback_message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Answer a question with automatic grounding fallback.
    
    Complete workflow:
    1. Retrieve relevant chunks
    2. Check if retrieval succeeded
    3. Generate grounded answer (or return fallback)
    4. Include source attribution
    
    Args:
        question: The user's question
        k: Number of chunks to retrieve
        metadata_filter: Optional ChromaDB where filter
        temperature: LLM temperature
        fallback_message: Custom fallback if no context found
        
    Returns:
        Dictionary with:
        - answer: The response (grounded or fallback)
        - sources: Source metadata
        - sources_count: Number of sources
        - status: "success" or "no_context"
        
    Raises:
        ValueError: If question is empty
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    logger.info("=" * 70)
    logger.info(f"Answer Query: {question}")
    logger.info("=" * 70)
    
    # Retrieve chunks
    try:
        logger.info(f"Retrieving top-{k} chunks...")
        chunks = retrieve(question, top_k=k, metadata_filter=metadata_filter)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {
            "answer": fallback_message or "Unable to retrieve context. Please try again.",
            "sources": [],
            "sources_count": 0,
            "status": "retrieval_failed",
            "error": str(e)
        }
    
    # Check if retrieval found anything
    if not chunks:
        logger.warning("No relevant chunks found")
        return {
            "answer": fallback_message or "I don't have enough information in the provided context to answer this question.",
            "sources": [],
            "sources_count": 0,
            "status": "no_context"
        }

    if not retrieval_is_strong(chunks):
        logger.warning("Weak retrieval quality; refusing to answer without enough support")
        return {
            "answer": fallback_message or "I don't have enough reliable context to answer that.",
            "question": question,
            "sources": [],
            "context": "",
            "sources_count": 0,
            "status": "refused_weak_context"
        }
    
    # Generate grounded answer
    try:
        result = generate_grounded_answer(question, chunks, temperature=temperature)
        logger.info("=" * 70)
        return result
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {
            "answer": fallback_message or "Unable to generate answer. Please try again.",
            "sources": [],
            "sources_count": 0,
            "status": "generation_failed",
            "error": str(e)
        }


def compare_grounded_vs_ungrounded(
    question: str,
    k: int = 4,
    show_context: bool = False
) -> Dict[str, Any]:
    """
    Compare grounded (RAG) and ungrounded (hallucination) answers.
    
    Demonstrates the impact of grounding by showing:
    1. The grounded answer based on retrieved context
    2. The ungrounded answer from model knowledge
    3. The difference in accuracy and source attribution
    
    Args:
        question: The user's question
        k: Number of chunks to retrieve for grounding
        show_context: Include retrieved context in comparison
        
    Returns:
        Dictionary comparing both approaches:
        - question: Original question
        - grounded: Grounded answer result
        - ungrounded: Ungrounded answer result
        - comparison: Analysis of differences
    """
    logger.info("=" * 70)
    logger.info("GROUNDED vs UNGROUNDED COMPARISON")
    logger.info("=" * 70)
    
    # Generate grounded answer
    logger.info("\n1. Generating GROUNDED answer (with retrieval)...")
    try:
        grounded = answer_query(question, k=k)
    except Exception as e:
        logger.error(f"Grounded generation failed: {e}")
        grounded = {
            "answer": f"Error: {e}",
            "sources": [],
            "status": "error"
        }
    
    # Generate ungrounded answer
    logger.info("\n2. Generating UNGROUNDED answer (without retrieval)...")
    try:
        ungrounded = generate_ungrounded_answer(question)
    except Exception as e:
        logger.error(f"Ungrounded generation failed: {e}")
        ungrounded = {
            "answer": f"Error: {e}",
            "status": "error"
        }
    
    # Compile comparison
    comparison = {
        "question": question,
        "grounded": grounded,
        "ungrounded": ungrounded,
        "analysis": {
            "grounded_has_sources": len(grounded.get("sources", [])) > 0,
            "grounded_status": grounded.get("status"),
            "ungrounded_status": ungrounded.get("status"),
            "grounded_answer_length": len(grounded.get("answer", "")),
            "ungrounded_answer_length": len(ungrounded.get("answer", ""))
        }
    }
    
    if show_context and grounded.get("context"):
        comparison["context_used"] = grounded["context"]
    
    logger.info("=" * 70)
    return comparison


def verify_grounding(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that an answer is properly grounded in retrieved chunks.
    
    Checks:
    1. Answer contains source citations (if sources present)
    2. Sources are listed in result
    3. No obvious hallucinations
    
    Args:
        result: Result from answer_query() or generate_grounded_answer()
        
    Returns:
        Dictionary with verification details:
        - is_grounded: Boolean indicating if answer is grounded
        - has_sources: Whether sources are included
        - source_count: Number of sources
        - citations_found: Number of citation markers in answer
        - issues: List of potential grounding issues
    """
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    status = result.get("status")
    
    logger.info("Verifying grounding...")
    
    issues = []
    
    # Check status
    if status == "no_context":
        issues.append("No context found - fallback answer used")
    elif status != "success":
        issues.append(f"Non-success status: {status}")
    
    # Check for sources
    has_sources = len(sources) > 0
    if not has_sources:
        issues.append("No sources included in result")
    
    # Count citation markers in answer
    citation_count = answer.count("[")  # Simple heuristic for [1], [2], etc.
    
    # Check if answer is the fallback
    if "don't have enough information" in answer.lower():
        issues.append("Fallback message used - no answer provided")
    
    # Compile verification
    verification = {
        "is_grounded": status == "success" and has_sources,
        "has_sources": has_sources,
        "source_count": len(sources),
        "citations_found": citation_count,
        "issues": issues,
        "status": "verified" if not issues else "warnings"
    }
    
    logger.info(f"Grounding verification: {verification['status']}")
    if issues:
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    return verification


def print_grounding_check(result: Dict[str, Any], include_context: bool = False):
    """
    Pretty-print a grounding verification check.
    
    Args:
        result: Result from answer_query()
        include_context: Include retrieved context in output
    """
    print("\n" + "=" * 80)
    print("GROUNDING CHECK")
    print("=" * 80)
    
    print(f"\nQuestion: {result.get('question', 'N/A')}")
    print(f"\nAnswer:\n{result.get('answer', 'N/A')}")
    
    sources = result.get("sources", [])
    if sources:
        print(f"\nSources ({len(sources)}):")
        for source in sources:
            rank = source.get("index", "?")
            file = source.get("source", "Unknown")
            relevance = source.get("relevance", 0)
            print(f"  [{rank}] {file} (relevance: {relevance:.2f})")
    else:
        print("\nSources: None")
    
    if include_context and result.get("context"):
        print(f"\nContext Used:\n{result['context']}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    """
    Example usage of grounded generation functions.
    
    Run with:
        python -m src.grounded_generation
    """
    
    # Example question
    question = "What evidence is required for project submission?"
    
    print("\n" + "=" * 80)
    print("GROUNDED GENERATION EXAMPLES")
    print("=" * 80)
    
    # Example 1: Single grounded answer
    print("\n1. GROUNDED ANSWER (with retrieval)")
    print("-" * 80)
    try:
        result = answer_query(question, k=4)
        print_grounding_check(result)
        
        # Verify grounding
        verification = verify_grounding(result)
        print("Verification Result:")
        for key, value in verification.items():
            if key != "issues":
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {', '.join(value) if value else 'None'}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Grounded vs Ungrounded comparison
    print("\n\n2. GROUNDED vs UNGROUNDED COMPARISON")
    print("-" * 80)
    try:
        comparison = compare_grounded_vs_ungrounded(question, k=4)
        
        print("\nGROUNDED ANSWER:")
        print(comparison["grounded"]["answer"])
        print(f"Status: {comparison['grounded']['status']}")
        print(f"Sources: {len(comparison['grounded'].get('sources', []))}")
        
        print("\n" + "-" * 80)
        print("\nUNGROUNDED ANSWER:")
        print(comparison["ungrounded"]["answer"])
        print(f"Status: {comparison['ungrounded']['status']}")
        
        print("\n" + "-" * 80)
        print("\nANALYSIS:")
        analysis = comparison["analysis"]
        print(f"  Grounded answer has sources: {analysis['grounded_has_sources']}")
        print(f"  Grounded answer length: {analysis['grounded_answer_length']} chars")
        print(f"  Ungrounded answer length: {analysis['ungrounded_answer_length']} chars")
    
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80 + "\n")
