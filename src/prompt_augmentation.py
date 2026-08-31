"""
Prompt Augmentation: Context Injection and Token Budget Management

This module handles the assembly of retrieved chunks into a grounded prompt that:
  1. Formats chunks with source markers for citations
  2. Tracks token usage and respects budget limits
  3. Prioritizes best-ranked chunks
  4. Builds a prompt that instructs the model to answer only from context
  5. Prevents hallucination by enforcing grounding in retrieved evidence

The prompt augmentation stage is the control point where you choose what evidence
the model sees, how sources are named, and what to do when there is not enough context.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional

from dotenv import load_dotenv
import tiktoken

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Token budget configuration
# Adjust these based on your model's context window and requirements
DEFAULT_MAX_CONTEXT_TOKENS = 5000      # Reserve for retrieved chunks
DEFAULT_RESERVED_FOR_ANSWER = 1500     # Reserve for model's response
DEFAULT_RESERVED_FOR_INSTRUCTIONS = 800  # Reserve for system prompt + question
DEFAULT_MODEL = "gpt-3.5-turbo"        # Default model for token counting

# Get model from environment or use default
CHAT_MODEL = os.getenv("CHAT_MODEL", DEFAULT_MODEL)


def get_token_counter(model: str = CHAT_MODEL):
    """
    Get a token counter for the specified model.
    
    Uses tiktoken for OpenAI models. Falls back to approximation for other models.
    
    Args:
        model: Model name (e.g., "gpt-3.5-turbo", "gpt-4", "gemini-2.0-flash")
        
    Returns:
        A callable that takes text and returns token count
    """
    try:
        # Try to get the encoding for the model
        encoding = tiktoken.encoding_for_model(model)
        return lambda text: len(encoding.encode(text))
    except KeyError:
        # Model not found in tiktoken; try common alternatives
        try:
            encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5/4 encoding
            return lambda text: len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counter unavailable for {model}: {e}")
            # Fallback: rough approximation (1 token ≈ 4 characters)
            return lambda text: max(1, len(text) // 4)


# Initialize token counter
count_tokens = get_token_counter()


def format_chunk(
    index: int,
    chunk: Dict[str, Any],
    include_score: bool = True
) -> str:
    """
    Format a single chunk with source marker for citation.
    
    Format: [1] source.txt#chunk_5 (score: 0.92)
    Followed by the chunk text.
    
    Args:
        index: Sequential chunk number (1-based) for citation reference
        chunk: Retrieved chunk dict with keys: text, metadata, score
        include_score: Include relevance score in marker (default True)
        
    Returns:
        Formatted chunk string with marker and text
    """
    # Extract source information
    metadata = chunk.get("metadata", {})
    source = metadata.get("source", "Unknown Source")
    chunk_index = metadata.get("chunk_index", "")
    score = chunk.get("score", 0)
    
    # Build source marker
    source_marker = f"[{index}] {source}"
    if chunk_index:
        source_marker += f"#{chunk_index}"
    
    if include_score:
        source_marker += f" (relevance: {score:.2f})"
    
    # Combine marker and text
    text = chunk.get("text", "")
    formatted = f"{source_marker}\n{text}"
    
    return formatted


def assemble_context(
    chunks: List[Dict[str, Any]],
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    separator: str = "\n\n---\n\n"
) -> Tuple[str, int, List[int]]:
    """
    Assemble retrieved chunks into context string within token budget.
    
    Prioritizes highest-ranked chunks first. Stops adding chunks when the
    token limit is reached. Returns the assembled context, token count,
    and the indices of included chunks.
    
    Args:
        chunks: List of retrieved chunks (assumed sorted by relevance)
        max_context_tokens: Maximum tokens allowed for context (default 5000)
        separator: String to separate chunks in assembled context
        
    Returns:
        Tuple of:
        - assembled_context: Formatted context string
        - total_tokens: Number of tokens used
        - included_indices: List of chunk indices that were included
        
    Raises:
        ValueError: If chunks list is empty
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")
    
    logger.info(f"Assembling context from {len(chunks)} chunks "
                f"(max {max_context_tokens} tokens)...")
    
    selected_chunks = []
    used_tokens = 0
    included_indices = []
    
    for index, chunk in enumerate(chunks, start=1):
        # Format the chunk with source marker
        formatted = format_chunk(index, chunk)
        
        # Count tokens in this chunk
        chunk_tokens = count_tokens(formatted)
        
        # Check if adding this chunk exceeds the budget
        if used_tokens + chunk_tokens > max_context_tokens:
            logger.info(f"Token budget reached. Included {len(selected_chunks)}/{len(chunks)} chunks "
                       f"({used_tokens}/{max_context_tokens} tokens)")
            break
        
        # Add chunk to selection
        selected_chunks.append(formatted)
        used_tokens += chunk_tokens
        included_indices.append(index)
        
        logger.debug(f"  [{index}] Added {chunk_tokens} tokens "
                    f"(total: {used_tokens}/{max_context_tokens})")
    
    # Join chunks with separator
    context = separator.join(selected_chunks)
    
    logger.info(f"Context assembled: {len(selected_chunks)} chunks, "
               f"{used_tokens} tokens, {len(context)} characters")
    
    return context, used_tokens, included_indices


def build_augmented_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    system_instructions: Optional[str] = None,
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    include_token_info: bool = False
) -> Dict[str, Any]:
    """
    Build a complete augmented prompt with retrieved context.
    
    Assembles a prompt that clearly separates:
    1. System instructions (grounding directive)
    2. Retrieved context with source markers
    3. The user's question
    
    Args:
        question: The user's question
        retrieved_chunks: List of chunks from retrieval stage
        system_instructions: Custom system instructions (uses default if None)
        max_context_tokens: Token budget for context assembly
        include_token_info: Include token counts in response (default False)
        
    Returns:
        Dictionary with keys:
        - prompt: The complete augmented prompt string
        - context: The assembled context string
        - context_tokens: Number of tokens used for context
        - included_chunks: Count of chunks included
        - question: The original question
        - sources_used: Metadata for all included chunks
        - total_tokens: Total tokens in the complete prompt (if include_token_info=True)
        
    Raises:
        ValueError: If question or chunks are empty
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    if not retrieved_chunks:
        logger.warning("No chunks provided for prompt augmentation")
        return {
            "prompt": f"Question: {question}",
            "context": "",
            "context_tokens": 0,
            "included_chunks": 0,
            "question": question,
            "sources_used": [],
            "status": "no_context"
        }
    
    logger.info("Building augmented prompt...")
    
    # Default system instructions if not provided
    if system_instructions is None:
        system_instructions = """You are a grounded assistant for legal document analysis.

Instructions:
1. Answer the question ONLY using the provided context.
2. Do not use external knowledge or general facts.
3. If the answer is not in the context, clearly state: "I don't have enough information in the provided context to answer this question."
4. When possible, cite your sources using the markers like [1], [2], etc.
5. Be concise and accurate in your answer."""
    
    # Assemble context within token budget
    context, context_tokens, included_indices = assemble_context(
        retrieved_chunks,
        max_context_tokens=max_context_tokens
    )
    
    # Build the complete prompt
    prompt = f"""{system_instructions}

Context:
{context}

Question:
{question}"""
    
    # Extract metadata for included chunks
    sources_used = [
        {
            "index": i,
            "source": retrieved_chunks[i-1].get("metadata", {}).get("source", "Unknown"),
            "relevance": retrieved_chunks[i-1].get("score", 0),
            "metadata": retrieved_chunks[i-1].get("metadata", {})
        }
        for i in included_indices
    ]
    
    result = {
        "prompt": prompt,
        "context": context,
        "context_tokens": context_tokens,
        "included_chunks": len(included_indices),
        "question": question,
        "sources_used": sources_used
    }
    
    # Optionally include total token count
    if include_token_info:
        total_tokens = count_tokens(prompt)
        result["total_tokens"] = total_tokens
        logger.info(f"Total prompt tokens: {total_tokens} "
                   f"(context: {context_tokens}, instructions+question: "
                   f"{total_tokens - context_tokens})")
    
    logger.info(f"Augmented prompt built: {len(included_indices)} sources, "
               f"{context_tokens} context tokens")
    
    return result


def optimize_context_for_token_budget(
    chunks: List[Dict[str, Any]],
    question: str,
    total_budget: int = 8000,
    answer_reserve: int = DEFAULT_RESERVED_FOR_ANSWER,
    instructions_reserve: int = DEFAULT_RESERVED_FOR_INSTRUCTIONS
) -> Dict[str, Any]:
    """
    Optimize context assembly by calculating dynamic token budgets.
    
    Useful for models with different context windows. Allocates the available
    tokens intelligently between instructions, context, and answer.
    
    Args:
        chunks: Retrieved chunks
        question: User question
        total_budget: Total model context window tokens
        answer_reserve: Tokens to reserve for model's answer
        instructions_reserve: Tokens to reserve for instructions/question
        
    Returns:
        Dictionary with optimized budget breakdown and assembled prompt
    """
    # Calculate available tokens for context
    available_for_context = (
        total_budget - answer_reserve - instructions_reserve
    )
    
    if available_for_context <= 0:
        logger.warning(f"Insufficient token budget. Available: {available_for_context}")
        available_for_context = 1000  # Minimum fallback
    
    logger.info(f"Token Budget Breakdown:")
    logger.info(f"  Total: {total_budget}")
    logger.info(f"  Instructions + Question: {instructions_reserve}")
    logger.info(f"  Answer reserve: {answer_reserve}")
    logger.info(f"  Available for context: {available_for_context}")
    
    # Build prompt with calculated budget
    augmented = build_augmented_prompt(
        question,
        chunks,
        max_context_tokens=available_for_context,
        include_token_info=True
    )
    
    augmented["budget_breakdown"] = {
        "total": total_budget,
        "instructions_reserve": instructions_reserve,
        "answer_reserve": answer_reserve,
        "context_used": augmented["context_tokens"],
        "available_for_context": available_for_context,
        "remaining_unused": available_for_context - augmented["context_tokens"]
    }
    
    return augmented


def trim_long_chunk(text: str, max_chars: int = 1000) -> str:
    """
    Trim a chunk to maximum character length while preserving meaning.
    
    Useful when a single chunk is very long and would consume too many tokens.
    Attempts to cut at sentence boundaries.
    
    Args:
        text: The chunk text
        max_chars: Maximum character length
        
    Returns:
        Trimmed text with ellipsis if cut
    """
    if len(text) <= max_chars:
        return text
    
    # Try to cut at the last sentence boundary
    trimmed = text[:max_chars]
    last_period = trimmed.rfind(". ")
    
    if last_period > max_chars * 0.8:  # Only use if we're not cutting too much
        return trimmed[:last_period + 1]
    
    # Fall back to character limit with ellipsis
    return trimmed.rstrip() + "..."


def rank_and_reorder_chunks(
    chunks: List[Dict[str, Any]],
    strategy: str = "score"
) -> List[Dict[str, Any]]:
    """
    Re-rank chunks using different strategies before context assembly.
    
    Strategies:
    - "score": Sort by relevance score (highest first)
    - "diversity": Spread chunks by source to avoid redundancy
    - "recency": Favor chunks with newer metadata (if available)
    
    Args:
        chunks: Retrieved chunks
        strategy: Ranking strategy ("score", "diversity", "recency")
        
    Returns:
        Re-ranked chunks list
    """
    if strategy == "score":
        # Simple: sort by relevance score
        return sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
    
    elif strategy == "diversity":
        # Group by source, then interleave
        by_source = {}
        for chunk in chunks:
            source = chunk.get("metadata", {}).get("source", "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk)
        
        # Interleave chunks from different sources
        reordered = []
        max_per_source = max(len(chunks_list) for chunks_list in by_source.values())
        for i in range(max_per_source):
            for source in by_source:
                if i < len(by_source[source]):
                    reordered.append(by_source[source][i])
        
        return reordered[:len(chunks)]
    
    elif strategy == "recency":
        # Sort by metadata timestamp if available
        def get_timestamp(chunk):
            metadata = chunk.get("metadata", {})
            timestamp = metadata.get("timestamp", 0)
            # If no timestamp, use score as fallback
            return (timestamp, chunk.get("score", 0))
        
        return sorted(chunks, key=get_timestamp, reverse=True)
    
    else:
        logger.warning(f"Unknown ranking strategy: {strategy}. Using score.")
        return rank_and_reorder_chunks(chunks, strategy="score")


if __name__ == "__main__":
    """
    Example usage of prompt augmentation functions.
    
    Run with:
        python -m src.prompt_augmentation
    """
    
    # Example retrieved chunks
    example_chunks = [
        {
            "text": "Project submission requires evidence of completion, including signed off documentation, test results, and deployment logs.",
            "metadata": {"source": "submission_guide.md", "chunk_index": 1},
            "score": 0.95
        },
        {
            "text": "All projects must pass code review before submission. A minimum of two reviewers are required.",
            "metadata": {"source": "submission_guide.md", "chunk_index": 2},
            "score": 0.88
        },
        {
            "text": "Documentation must include API specifications, architecture diagrams, and user manual.",
            "metadata": {"source": "submission_guide.md", "chunk_index": 3},
            "score": 0.85
        },
        {
            "text": "Projects are evaluated on functionality, code quality, documentation completeness, and adherence to coding standards.",
            "metadata": {"source": "evaluation_criteria.txt", "chunk_index": 1},
            "score": 0.82
        },
    ]
    
    example_question = "What evidence is required for project submission?"
    
    print("\n" + "=" * 80)
    print("PROMPT AUGMENTATION EXAMPLE")
    print("=" * 80)
    
    # Example 1: Build augmented prompt
    print("\n1. Building Augmented Prompt")
    print("-" * 80)
    result = build_augmented_prompt(
        example_question,
        example_chunks,
        include_token_info=True
    )
    
    print(f"\nContext Tokens Used: {result['context_tokens']}")
    print(f"Chunks Included: {result['included_chunks']}")
    print(f"Total Prompt Tokens: {result['total_tokens']}")
    print(f"\nSources Used:")
    for source in result['sources_used']:
        print(f"  [{source['index']}] {source['source']} (relevance: {source['relevance']:.2f})")
    
    print(f"\nAssembled Prompt:")
    print("-" * 80)
    print(result['prompt'])
    
    # Example 2: Optimize for different token budgets
    print("\n\n2. Token Budget Optimization")
    print("-" * 80)
    for budget in [4000, 8000, 16000]:
        print(f"\nBudget: {budget} tokens")
        optimized = optimize_context_for_token_budget(
            example_chunks,
            example_question,
            total_budget=budget
        )
        breakdown = optimized['budget_breakdown']
        print(f"  Context used: {breakdown['context_used']}/{breakdown['available_for_context']} tokens")
        print(f"  Chunks: {optimized['included_chunks']}")
    
    # Example 3: Ranking strategies
    print("\n\n3. Chunk Re-ranking Strategies")
    print("-" * 80)
    for strategy in ["score", "diversity"]:
        reranked = rank_and_reorder_chunks(example_chunks, strategy=strategy)
        print(f"\n{strategy.upper()} Strategy:")
        for i, chunk in enumerate(reranked, 1):
            source = chunk['metadata']['source']
            score = chunk['score']
            print(f"  [{i}] {source} (score: {score:.2f})")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80 + "\n")
