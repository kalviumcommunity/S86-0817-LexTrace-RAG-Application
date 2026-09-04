import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import config
from src.document_indexer import process_uploaded_document, store_upload
from src.rag_service import get_health_status, guarded_answer
from src.schemas import (
    DocumentSummary,
    DocumentUploadResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    Source,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("lextrace-api")

app = FastAPI(
    title="LexTrace RAG API",
    version="1.0.0",
    description="Production-ready Backend API for Legal Document Retrieval-Augmented Generation.",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integrations (React, Next.js, Vue, Streamlit, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    """API welcome endpoint."""
    return {
        "service": "LexTrace RAG Backend API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "query": "/query"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint to inspect ChromaDB vector collection and models."""
    try:
        health_info = get_health_status()
        return health_info
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector DB or service unhealthy: {e}"
        )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
def query_rag(request: QueryRequest):
    """
    Query the RAG pipeline with a user question.

    - **question**: User question (3 to 1000 characters)
    - **top_k**: Optional number of chunks to retrieve
    - **Returns**: Grounded natural language answer, supporting sources/citations, and status.
    """
    logger.info(f"Received query: '{request.question}' (top_k={request.top_k})")

    try:
        result = guarded_answer(
            question=request.question,
            top_k=request.top_k
        )

        sources_payload = [
            Source(
                source=src.get("source", "unknown"),
                chunk_id=src.get("chunk_id"),
                score=src.get("score")
            )
            for src in result.get("sources", [])
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources_payload,
            status=result.get("status", "answered")
        )

    except ValueError as error:
        logger.warning(f"Validation error processing query: {error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error)
        )

    except Exception as error:
        logger.error(f"Unexpected error in RAG service: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG service failed"
        )


@app.post(
    "/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"]
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document, validate it, chunk it, embed it, and index it into the vector database at runtime.

    - **file**: Multipart document file (.txt, .md, .pdf, .docx, .html)
    - **Returns**: Indexing summary with chunk counts and runtime indexed status.
    """
    logger.info(f"Received document upload: '{file.filename}' (content_type={file.content_type})")

    try:
        path = await store_upload(file)
        summary = process_uploaded_document(path)
        return DocumentUploadResponse(
            status="indexed",
            filename=file.filename,
            summary=summary
        )

    except HTTPException:
        raise

    except ValueError as error:
        logger.warning(f"Validation error during document indexing: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error)
        )

    except Exception as error:
        logger.error(f"Document indexing failed for {file.filename}: {error}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document indexing failed: {error}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    )
