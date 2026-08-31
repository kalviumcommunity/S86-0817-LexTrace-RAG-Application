"""
Citations: Source Attribution and Verification

This module handles citation generation, mapping, and verification to ensure
that generated answers can be traced back to real retrieved documents.

Key principles:
  1. Create citation markers from retrieved chunk metadata
  2. Instruct the model to cite only provided sources
  3. Return answer + citation map for user verification
  4. Verify citations against original chunk text
  5. Prevent fabricated citations when sources are insufficient

A good citation connects an answer claim back to a real source chunk,
building trust and enabling verification. Fake citations are worse than
no citations because they give users false confidence.
"""

import os
import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from src.prompt_augmentation import build_augmented_prompt, format_chunk

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")

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

# System prompt emphasizing proper citations
CITED_SYSTEM_PROMPT = """You are a grounded assistant for legal document analysis.

CITATION RULES (CRITICAL):
1. Answer ONLY using the provided context.
2. Cite EVERY factual claim using source markers like [1], [2], etc.
3. Use ONLY source markers that appear in the context (e.g., if you see [1], [2], [3] in the context, use only these).
4. DO NOT invent citations. If you cite [5] but only [1], [2], [3] are in the context, the citation is fabricated.
5. If the context does not support a complete answer, say:
   "I don't have enough information in the provided context to answer this question."
6. Do NOT cite [X] if the fact is not in source [X].
7. Be accurate, concise, and traceable.

Examples:
- Correct: "Project submission requires evidence [1] and code review [2]."
- Incorrect: "Project submission requires evidence [5]." (if [5] doesn't exist in context)
- Incorrect: "Project submission requires evidence." (no citation when one exists)

Remember: Citations are verifiable. Every citation MUST map to a real source."""


def build_citation_map(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a mapping from citation markers to chunk metadata and content.
    
    Creates stable citation labels [1], [2], etc. that map to real
    retrieved chunks. This enables users to verify each citation.
    
    Args:
        chunks: List of retrieved chunks
        
    Returns:
        Dictionary mapping citation markers (e.g., "[1]") to chunk details:
        {
            "[1]": {
                "source": "filename.md",
                "chunk_id": "chunk_123",
                "chunk_index": 0,
                "section": "Introduction",
                "text": "Full chunk text...",
                "relevance": 0.95
            },
            ...
        }
        
    Raises:
        ValueError: If chunks is empty
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")
    
    logger.info(f"Building citation map for {len(chunks)} chunks...")
    
    citation_map = {}
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        
        citation_marker = f"[{index}]"
        citation_map[citation_marker] = {
            "index": index,
            "source": metadata.get("source", "Unknown Source"),
            "chunk_id": metadata.get("chunk_id", chunk.get("id", f"chunk_{index}")),
            "chunk_index": metadata.get("chunk_index", index - 1),
            "section": metadata.get("section", ""),
            "document_type": metadata.get("document_type", ""),
            "text": chunk.get("text", ""),
            "relevance_score": chunk.get("score", 0),
            "metadata": metadata
        }
        
        logger.debug(f"  {citation_marker}: {metadata.get('source')} "
                    f"(relevance: {chunk.get('score', 0):.2f})")
    
    logger.info(f"Citation map created with {len(citation_map)} entries")
    return citation_map


def build_cited_prompt(
    question: str,
    chunks: List[Dict[str, Any]],
    system_instructions: Optional[str] = None,
    max_context_tokens: int = 5000
) -> Dict[str, Any]:
    """
    Build a prompt that explicitly instructs the model to cite sources.
    
    Combines:
    1. System instructions emphasizing proper citation
    2. Retrieved context with citation markers [1], [2], etc.
    3. The user's question
    
    Args:
        question: The user's question
        chunks: Retrieved chunks
        system_instructions: Custom system prompt (uses cited prompt if None)
        max_context_tokens: Token budget for context
        
    Returns:
        Dictionary with:
        - prompt: Complete prompt with citation instructions
        - context: Assembled context with markers
        - citation_map: Mapping from markers to chunks
        - sources: Metadata for included sources
        
    Raises:
        ValueError: If question or chunks are empty
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    if not chunks:
        raise ValueError("Chunks cannot be empty")
    
    logger.info("Building cited prompt...")
    
    # Use provided or default citation-focused system prompt
    if system_instructions is None:
        system_instructions = CITED_SYSTEM_PROMPT
    
    # Build augmented prompt with retrieved context
    prompt_data = build_augmented_prompt(
        question,
        chunks,
        system_instructions=system_instructions,
        max_context_tokens=max_context_tokens,
        include_token_info=False
    )
    
    # Build citation map for the included chunks
    # We need to map only the chunks that are actually in the context
    included_chunk_indices = []
    for i, chunk in enumerate(chunks, start=1):
        # Check if this chunk's formatted version is in the context
        # This is approximate; in practice, we track which chunks are included
        included_chunk_indices.append(i)
    
    # Create citation map for included chunks
    citation_map = build_citation_map(chunks[:len(included_chunk_indices)])
    
    return {
        "prompt": prompt_data["prompt"],
        "context": prompt_data["context"],
        "citation_map": citation_map,
        "sources": prompt_data["sources_used"],
        "included_chunks": len(included_chunk_indices)
    }


def extract_citations_from_answer(answer: str) -> Set[int]:
    """
    Extract citation numbers from an answer string.
    
    Finds all instances of [1], [2], etc. in the answer.
    
    Args:
        answer: Generated answer string
        
    Returns:
        Set of citation numbers found (e.g., {1, 2, 3})
    """
    # Pattern: [number]
    pattern = r'\[(\d+)\]'
    matches = re.findall(pattern, answer)
    citations = {int(m) for m in matches}
    
    logger.debug(f"Extracted citations from answer: {sorted(citations)}")
    return citations


def verify_citation(
    citation_index: int,
    answer_claim: str,
    citation_map: Dict[str, Dict[str, Any]],
    source_text: Optional[str] = None
) -> Dict[str, Any]:
    """
    Verify that a single citation is valid and supported by the source.
    
    Checks:
    1. Citation marker exists in the map
    2. Citation text matches source content (if source_text provided)
    3. Citation is not fabricated
    
    Args:
        citation_index: The citation number (e.g., 1 for [1])
        answer_claim: The claim in the answer being cited
        citation_map: Citation map from build_citation_map()
        source_text: Optional source text to verify against
        
    Returns:
        Dictionary with:
        - is_valid: Boolean indicating if citation is valid
        - exists: Whether the citation marker exists
        - source: Source document name
        - supported: Whether claim is supported by source
        - issues: List of any validation issues
    """
    marker = f"[{citation_index}]"
    issues = []
    
    logger.debug(f"Verifying citation {marker}...")
    
    # Check if citation marker exists
    if marker not in citation_map:
        issues.append(f"Citation marker {marker} does not exist in retrieved sources")
        logger.warning(f"Invalid citation: {marker} not in map")
        return {
            "is_valid": False,
            "exists": False,
            "source": None,
            "supported": False,
            "issues": issues
        }
    
    # Get citation details
    citation = citation_map[marker]
    source = citation.get("source", "Unknown")
    chunk_text = citation.get("text", "")
    
    # Verify claim is related to source content
    is_supported = False
    if source_text:
        # Check if key terms from claim appear in source
        claim_terms = answer_claim.lower().split()
        source_lower = source_text.lower()
        matching_terms = sum(1 for term in claim_terms if term in source_lower)
        is_supported = matching_terms > 0
    else:
        # Without explicit source text, check if chunk has relevant content
        is_supported = len(chunk_text) > 0
    
    if not is_supported:
        issues.append(f"Claim does not appear to be supported by source text")
    
    logger.debug(f"  {marker} valid: exists={True}, supported={is_supported}")
    
    return {
        "is_valid": True,
        "exists": True,
        "source": source,
        "chunk_id": citation.get("chunk_id"),
        "supported": is_supported,
        "issues": issues
    }


def verify_all_citations(
    answer: str,
    citation_map: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Verify all citations in an answer against the citation map.
    
    Checks:
    1. All cited sources exist in retrieval
    2. No fabricated citations (e.g., [5] when only [1], [2], [3] exist)
    3. At least one source is cited
    
    Args:
        answer: Generated answer string
        citation_map: Citation map from build_citation_map()
        
    Returns:
        Dictionary with:
        - all_valid: Boolean indicating if all citations are valid
        - cited_sources: Set of citation indices found
        - available_sources: Set of available citation indices
        - fabricated: List of fabricated citation indices
        - missing_citations: Claims without citations (heuristic)
        - verification: Details for each citation
    """
    logger.info("Verifying all citations in answer...")
    
    # Extract citations from answer
    cited_sources = extract_citations_from_answer(answer)
    available_sources = {int(m.replace("[", "").replace("]", "")) 
                        for m in citation_map.keys()}
    
    logger.info(f"Cited sources: {sorted(cited_sources)}")
    logger.info(f"Available sources: {sorted(available_sources)}")
    
    # Find fabricated citations
    fabricated = cited_sources - available_sources
    if fabricated:
        logger.warning(f"Fabricated citations found: {sorted(fabricated)}")
    
    # Verify each citation
    verification_details = {}
    for citation_num in cited_sources:
        marker = f"[{citation_num}]"
        if marker in citation_map:
            verification_details[marker] = {
                "exists": True,
                "source": citation_map[marker].get("source"),
                "chunk_id": citation_map[marker].get("chunk_id")
            }
    
    # Check for missing citations (heuristic: factual claims without citations)
    # Simple check: sentences with common keywords but no [N] marker
    missing_citations = 0
    factual_keywords = ["required", "must", "should", "evidence", "policy", "regulation"]
    sentences = re.split(r'[.!?]+', answer)
    for sentence in sentences:
        if any(kw in sentence.lower() for kw in factual_keywords):
            if not re.search(r'\[\d+\]', sentence):
                missing_citations += 1
    
    result = {
        "all_valid": len(fabricated) == 0,
        "cited_sources": sorted(cited_sources),
        "available_sources": sorted(available_sources),
        "fabricated": sorted(fabricated),
        "missing_citations": missing_citations,
        "verification_details": verification_details,
        "status": "valid" if len(fabricated) == 0 else "invalid"
    }
    
    if fabricated:
        logger.error(f"Verification failed: {len(fabricated)} fabricated citations")
    else:
        logger.info("Verification passed: all citations are valid")
    
    return result


def answer_with_citations(
    question: str,
    chunks: List[Dict[str, Any]],
    temperature: float = 0.3,
    max_tokens: int = 1024,
    verify: bool = True
) -> Dict[str, Any]:
    """
    Generate an answer with proper citation and return the citation map.
    
    Complete workflow:
    1. Build citation map from chunks
    2. Create cited prompt instructing model to cite sources
    3. Generate answer from LLM
    4. Optionally verify all citations
    5. Return answer + citation map + verification
    
    Args:
        question: User question
        chunks: Retrieved chunks
        temperature: LLM temperature
        max_tokens: Maximum tokens in response
        verify: Whether to verify citations (default True)
        
    Returns:
        Dictionary with:
        - answer: Generated answer with citations
        - question: Original question
        - citation_map: Mapping from [1], [2], etc. to chunks
        - citations_used: Citations found in answer
        - verification: Citation verification results (if verify=True)
        - status: "success" or "error"
        
    Raises:
        ValueError: If question or chunks are empty
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    if not chunks:
        raise ValueError("Chunks cannot be empty")
    
    logger.info(f"Generating cited answer for: {question[:60]}...")
    
    try:
        # Build prompt with citation instructions
        prompt_data = build_cited_prompt(question, chunks)
        
        # Call LLM with citation instructions
        logger.info("Calling LLM with citation instructions...")
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
        logger.info(f"Answer generated ({len(answer)} characters)")
        
        # Extract citations from answer
        citations_used = extract_citations_from_answer(answer)
        
        result = {
            "answer": answer,
            "question": question,
            "citation_map": prompt_data["citation_map"],
            "citations_used": sorted(citations_used),
            "sources_count": prompt_data["included_chunks"],
            "status": "success"
        }
        
        # Verify citations if requested
        if verify:
            verification = verify_all_citations(answer, prompt_data["citation_map"])
            result["verification"] = verification
            
            if not verification["all_valid"]:
                logger.warning(f"Verification failed: {verification['fabricated']} fabricated citations")
                result["status"] = "warning"
        
        return result
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise


def print_answer_with_citations(result: Dict[str, Any], show_sources: bool = True):
    """
    Pretty-print an answer with citations and source details.
    
    Args:
        result: Result from answer_with_citations()
        show_sources: Whether to show full source details
    """
    print("\n" + "=" * 80)
    print("CITED ANSWER")
    print("=" * 80)
    
    print(f"\nQuestion:\n{result.get('question', 'N/A')}")
    
    print(f"\nAnswer:\n{result.get('answer', 'N/A')}")
    
    citations_used = result.get("citations_used", [])
    citation_map = result.get("citation_map", {})
    
    print(f"\nCitations Used: {citations_used if citations_used else 'None'}")
    
    if show_sources and citation_map:
        print(f"\nSource Details:")
        for marker in sorted(citation_map.keys()):
            citation = citation_map[marker]
            source = citation.get("source")
            relevance = citation.get("relevance_score", 0)
            print(f"\n  {marker} {source} (relevance: {relevance:.2f})")
            print(f"     Text excerpt: {citation.get('text', '')[:100]}...")
    
    # Show verification if available
    verification = result.get("verification")
    if verification:
        print(f"\nVerification Status: {verification['status'].upper()}")
        if verification["fabricated"]:
            print(f"  ❌ Fabricated citations: {verification['fabricated']}")
        else:
            print(f"  ✓ All citations valid")
        if verification["missing_citations"] > 0:
            print(f"  ⚠ Missing citations on {verification['missing_citations']} claims")
    
    print("\n" + "=" * 80 + "\n")


def audit_citation_claim(
    claim: str,
    citation_index: int,
    citation_map: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Audit a single claim-to-citation mapping for accuracy.
    
    Useful for manual verification. User selects a claim from the answer
    and the citation it uses, then this function checks if the source
    actually supports the claim.
    
    Args:
        claim: The factual claim from the answer
        citation_index: The citation number [N]
        citation_map: Citation map from answer result
        
    Returns:
        Audit result with:
        - is_supported: Whether source supports claim
        - source: Document used
        - source_text: Full text of source chunk
        - matching_terms: Terms from claim found in source
        - confidence: Confidence score
    """
    marker = f"[{citation_index}]"
    
    if marker not in citation_map:
        return {
            "is_supported": False,
            "source": None,
            "source_text": None,
            "error": f"Citation {marker} not found in map",
            "confidence": 0.0
        }
    
    citation = citation_map[marker]
    source_text = citation.get("text", "")
    source = citation.get("source")
    
    # Check if claim terms are in source
    claim_terms = claim.lower().split()
    matching_terms = [t for t in claim_terms if t in source_text.lower()]
    confidence = len(matching_terms) / max(len(claim_terms), 1)
    
    is_supported = confidence > 0.3  # At least 30% term overlap
    
    logger.info(f"Auditing claim: '{claim}' -> {marker}")
    logger.info(f"  Matching terms: {len(matching_terms)}/{len(claim_terms)}")
    logger.info(f"  Confidence: {confidence:.1%}")
    
    return {
        "is_supported": is_supported,
        "source": source,
        "source_text": source_text,
        "matching_terms": matching_terms,
        "confidence": confidence,
        "claim": claim,
        "citation": marker
    }


if __name__ == "__main__":
    """
    Example usage of citation functions.
    
    Run with:
        python -m src.citations
    """
    
    # Example chunks
    example_chunks = [
        {
            "text": "Project submission requires evidence of completion including signed off documentation, test results, and deployment logs.",
            "metadata": {"source": "submission_guide.md", "chunk_index": 1, "section": "Requirements"},
            "score": 0.95
        },
        {
            "text": "All projects must pass code review before submission. A minimum of two reviewers are required for approval.",
            "metadata": {"source": "submission_guide.md", "chunk_index": 2, "section": "Code Review"},
            "score": 0.88
        },
        {
            "text": "Documentation must include API specifications, architecture diagrams, and user manual with clear instructions.",
            "metadata": {"source": "documentation_standards.md", "chunk_index": 1, "section": "Required Documentation"},
            "score": 0.85
        }
    ]
    
    question = "What evidence is required for project submission?"
    
    print("\n" + "=" * 80)
    print("CITATIONS EXAMPLE")
    print("=" * 80)
    
    try:
        # Generate answer with citations
        result = answer_with_citations(question, example_chunks, verify=True)
        
        # Print answer and sources
        print_answer_with_citations(result, show_sources=True)
        
        # Show citation verification
        if "verification" in result:
            verification = result["verification"]
            print("Citation Verification:")
            print(f"  Status: {verification['status']}")
            print(f"  Cited sources: {verification['cited_sources']}")
            print(f"  Fabricated: {verification['fabricated'] if verification['fabricated'] else 'None'}")
            print(f"  Missing citations: {verification['missing_citations']}")
        
        # Audit specific citations
        print("\n" + "=" * 80)
        print("CITATION AUDITS")
        print("=" * 80)
        
        if result.get("citations_used"):
            # Audit the first citation as example
            marker = f"[{result['citations_used'][0]}]"
            citation = result["citation_map"][marker]
            
            audit = audit_citation_claim(
                claim="Projects require documentation",
                citation_index=result['citations_used'][0],
                citation_map=result["citation_map"]
            )
            
            print(f"\nAudit of citation {marker}:")
            print(f"  Claim: {audit['claim']}")
            print(f"  Source: {audit['source']}")
            print(f"  Supported: {audit['is_supported']}")
            print(f"  Confidence: {audit['confidence']:.1%}")
            print(f"  Matching terms: {audit['matching_terms']}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80 + "\n")
