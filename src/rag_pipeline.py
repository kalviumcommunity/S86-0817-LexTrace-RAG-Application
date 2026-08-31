"""
RAG Pipeline: End-to-end Retrieval-Augmented Generation

This module orchestrates the complete RAG flow:
  1. Embed query → Convert user question to vector
  2. Retrieve context → Find top-k relevant chunks
  3. Assemble context → Format with citations
  4. Generate answer → Use LLM to ground response
  5. Return answer + sources
"""

import os
import logging
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from src.retrival import retrieve
from prompts.answer import render_prompt

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.0-flash")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

# Initialize embedding model
embed_model = GoogleGenAIEmbedding(
    model_name=EMBED_MODEL,
    api_key=GEMINI_API_KEY,
)


def embed_query(query: str) -> List[float]:
    """
    Convert a user query into an embedding vector.
    
    Args:
        query: The user's question string
        
    Returns:
        A vector embedding of the query
        
    Raises:
        ValueError: If query is empty
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    logger.info(f"Embedding query: {query[:50]}...")
    query_embedding = embed_model.get_text_embedding(query)
    logger.info(f"Query embedded successfully (dimension: {len(query_embedding)})")
    
    return query_embedding


def retrieve_context(
    query: str, 
    k: int = 4, 
    metadata_filter: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most relevant chunks for a query.
    
    Args:
        query: The user's question
        k: Number of top chunks to retrieve (default 4)
        metadata_filter: Optional ChromaDB where filter for metadata
        
    Returns:
        List of retrieved chunks with scores, text, and metadata
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    logger.info(f"Retrieving top-{k} chunks for query...")
    chunks = retrieve(query, top_k=k, metadata_filter=metadata_filter)
    logger.info(f"Retrieved {len(chunks)} chunks")
    
    return chunks


def assemble_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a single context string with citations.
    
    Each chunk is numbered and includes its source in the assembled context.
    This allows the LLM to generate answers with proper attribution.
    
    Args:
        chunks: List of retrieved chunks from retrieve_context()
        
    Returns:
        Formatted context string ready for the LLM prompt
    """
    if not chunks:
        logger.warning("No chunks provided to assemble_context")
        return ""
    
    logger.info(f"Assembling context from {len(chunks)} chunks...")
    parts = []
    
    for index, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("source", "Unknown Source")
        text = chunk["text"]
        score = chunk.get("score", 0)
        
        # Format: [1] Source: filename.txt (confidence: 0.95)
        # Followed by the chunk text
        citation = f"[{index}] Source: {source} (relevance score: {score:.2f})"
        parts.append(f"{citation}\n{text}")
    
    context = "\n\n".join(parts)
    logger.info(f"Context assembled ({len(context)} characters)")
    
    return context


def generate_answer(query: str, context: str, temperature: float = 0.3) -> str:
    """
    Generate a grounded answer using an LLM and retrieved context.
    
    The LLM is instructed to:
    - Only use the provided context
    - Not invent information
    - Acknowledge if context is insufficient
    
    Args:
        query: The original user question
        context: Assembled context from retrieve + assemble steps
        temperature: LLM sampling temperature (0.0-1.0, default 0.3 for consistency)
        
    Returns:
        Generated answer string from the LLM
        
    Raises:
        ValueError: If query or context is empty
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if not context or not context.strip():
        logger.warning("Context is empty - LLM may return a fallback response")
    
    # Use the prompt template from prompts/answer.py
    prompt = render_prompt(context, query)
    
    logger.info("Calling LLM to generate answer...")
    
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
            max_tokens=1024,
        )
        
        answer = response.choices[0].message.content
        logger.info("Answer generated successfully")
        return answer
        
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise


def answer_query(
    query: str,
    k: int = 4,
    metadata_filter: Optional[Dict] = None,
    temperature: float = 0.3,
    return_chunks: bool = False
) -> Dict[str, Any]:
    """
    End-to-end RAG orchestrator: query → embedding → retrieval → generation.
    
    This function connects all stages of the RAG pipeline:
    1. Embeds the query to a vector
    2. Retrieves top-k relevant chunks
    3. Assembles chunks into formatted context
    4. Generates answer using LLM
    5. Returns answer with sources and optional chunk details
    
    Args:
        query: The user's question
        k: Number of retrieved chunks (default 4)
        metadata_filter: Optional ChromaDB where filter
        temperature: LLM temperature parameter (default 0.3)
        return_chunks: Include full chunk details in response (default False)
        
    Returns:
        Dictionary with keys:
        - answer: Generated answer string
        - sources: List of source metadata from retrieved chunks
        - num_chunks: Number of chunks retrieved
        - chunks: (optional) Full chunk details if return_chunks=True
        
    Raises:
        ValueError: If query is empty
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    logger.info("=" * 60)
    logger.info(f"RAG Pipeline Started")
    logger.info(f"Query: {query}")
    logger.info("=" * 60)
    
    try:
        # Stage 1: Embed the query
        # (Note: retrieve() handles embedding internally, but we could use
        # embed_query() if we need the vector for other purposes)
        
        # Stage 2: Retrieve relevant chunks
        chunks = retrieve_context(query, k=k, metadata_filter=metadata_filter)
        
        # Handle case where retrieval returns no results
        if not chunks:
            logger.warning("No relevant chunks found for query")
            return {
                "answer": "I could not find relevant context for that question. Please try a more specific query.",
                "sources": [],
                "num_chunks": 0,
                "chunks": [] if return_chunks else None,
                "status": "no_results"
            }
        
        # Stage 3: Assemble context from chunks
        context = assemble_context(chunks)
        
        # Stage 4: Generate answer
        answer = generate_answer(query, context, temperature=temperature)
        
        # Extract source metadata (for citations)
        sources = [
            {
                "rank": chunk["rank"],
                "source": chunk["metadata"].get("source", "Unknown"),
                "relevance_score": chunk.get("score", 0),
                "metadata": chunk["metadata"]
            }
            for chunk in chunks
        ]
        
        result = {
            "answer": answer,
            "sources": sources,
            "num_chunks": len(chunks),
            "status": "success"
        }
        
        # Optionally include full chunk details
        if return_chunks:
            result["chunks"] = chunks
        
        logger.info("=" * 60)
        logger.info(f"RAG Pipeline Completed Successfully")
        logger.info(f"Answer length: {len(answer)} characters")
        logger.info(f"Sources used: {len(sources)}")
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"RAG Pipeline failed: {e}", exc_info=True)
        raise


def batch_answer_queries(
    queries: List[str],
    k: int = 4,
    temperature: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Process multiple queries through the RAG pipeline.
    
    Useful for batch evaluation, testing, or processing multiple questions.
    Logs progress and any failures per query.
    
    Args:
        queries: List of query strings
        k: Number of chunks per query
        temperature: LLM temperature
        
    Returns:
        List of results (one dict per query)
    """
    logger.info(f"Processing {len(queries)} queries in batch mode...")
    results = []
    
    for i, query in enumerate(queries, start=1):
        try:
            logger.info(f"[{i}/{len(queries)}] Processing: {query[:50]}...")
            result = answer_query(query, k=k, temperature=temperature)
            results.append(result)
        except Exception as e:
            logger.error(f"[{i}/{len(queries)}] Failed: {e}")
            results.append({
                "answer": None,
                "sources": [],
                "status": "failed",
                "error": str(e)
            })
    
    logger.info(f"Batch processing complete: {len(results)} queries processed")
    return results


if __name__ == "__main__":
    """
    Example usage of the RAG pipeline.
    
    Run with:
        python -m src.rag_pipeline
    """
    
    # Example queries
    example_queries = [
        "What evidence is required for project submission?",
        "What are the employment policy regulations?",
        "How should contracts be structured?"
    ]
    
    print("\n" + "=" * 80)
    print("RAG PIPELINE EXAMPLE")
    print("=" * 80 + "\n")
    
    for i, query in enumerate(example_queries, start=1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print("=" * 80)
        
        try:
            result = answer_query(query, k=4, temperature=0.3)
            
            print(f"\n📝 ANSWER:")
            print(result["answer"])
            
            print(f"\n📚 SOURCES ({result['num_chunks']} chunks):")
            for source in result["sources"]:
                print(f"  [{source['rank']}] {source['source']} "
                      f"(score: {source['relevance_score']:.2f})")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80 + "\n")
