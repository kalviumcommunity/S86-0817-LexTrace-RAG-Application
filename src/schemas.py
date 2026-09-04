from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming user query payload."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The natural language question to ask the RAG system.",
        examples=["What notice period is required for standard contract termination?"]
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional number of context chunks to retrieve (defaults to server config)."
    )


class Source(BaseModel):
    """Source reference and evidence metadata."""
    source: str = Field(
        ...,
        description="Source document filename or identifier.",
        examples=["contract.txt"]
    )
    chunk_id: Optional[str] = Field(
        default=None,
        description="Unique identifier of the retrieved chunk.",
        examples=["chunk_0"]
    )
    score: Optional[float] = Field(
        default=None,
        description="Similarity / confidence score (0.0 to 1.0).",
        examples=[0.84]
    )


class QueryResponse(BaseModel):
    """Structured response returned to frontend / API consumers."""
    answer: str = Field(
        ...,
        description="Grounded natural language answer synthesized from retrieved documents."
    )
    sources: List[Source] = Field(
        default_factory=list,
        description="List of supporting document sources and evidence."
    )
    status: str = Field(
        ...,
        description="Status of response: 'answered', 'refused', or 'error'.",
        examples=["answered", "refused"]
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., examples=["healthy"])
    collection_name: str
    collection_count: int
    chat_model: str
    embedding_model: str
