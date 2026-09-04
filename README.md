# LexTrace – RAG Application

LexTrace is a Retrieval-Augmented Generation (RAG) application that processes documents, creates embeddings, stores them in a vector database, and retrieves relevant chunks for user queries.

The project is being developed step by step to understand and build a complete RAG pipeline.

---

## Project Overview

The current document-processing pipeline is:

```text
Documents
   ↓
Document Loading
   ↓
Text Cleaning
   ↓
Chunking
   ↓
Embedding Generation
   ↓
Vector Storage (ChromaDB)
   ↓
Semantic Retrieval
```

---

## Project Structure

```text
LexTrace-RAG-Application/
├── data/
│   ├── chroma/
│   │   └── ChromaDB persistent storage
│   └── sample/
│       ├── NON.docx
│       ├── contract.txt
│       ├── employement_policy.md
│       └── regulation.html
├── prompts/
│   └── answer.py
├── src/
│   ├── api.py
│   ├── chunking.py
│   ├── config.py
│   ├── embeddings.py
│   ├── evaluate_rag.py
│   ├── evaluator.py
│   ├── ingestion.py
│   ├── llm_test.py
│   ├── prompt_template_test.py
│   ├── prompt_test.py
│   ├── rag_pipeline.py
│   ├── rag_service.py
│   ├── retrival.py
│   ├── schemas.py
│   ├── structured_output_test.py
│   ├── temperature_test.py
│   ├── test_sets.py
│   ├── text_cleaner.py
│   └── token_test.py
├── tests/
│   ├── test_api.py
│   └── test_evaluation.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Tech Stack

- Python
- FastAPI & Uvicorn (Backend REST API)
- Pydantic v2 (Request/Response Validation)
- Google Gemini Embeddings & Chat Models (`gemini-embedding-001`, `gemini-3.1-flash-lite`)
- ChromaDB (Vector Store)
- LlamaIndex
- python-dotenv & OpenAI SDK
- pytest & httpx

---

## Backend REST API

LexTrace exposes a production-ready FastAPI backend serving as the stable contract for frontends, chatbot UIs, or microservices.

### 1. Starting the API Server

```bash
# Start server with auto-reload on http://localhost:8000
python src/api.py

# Or via uvicorn directly
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger UI is available at:
👉 **`http://localhost:8000/docs`**

---

### 2. API Endpoints

#### **`POST /query`** — Submit User Question

Validates request body, retrieves relevant context, executes guarded generation, and returns structured JSON with answers and citation sources.

- **Request Schema (`QueryRequest`)**:
```json
{
  "question": "What notice period is required for standard contract termination?",
  "top_k": 3
}
```

- **Successful Response (`QueryResponse` — 200 OK)**:
```json
{
  "answer": "Either party may terminate the agreement by providing 30 days written notice.",
  "sources": [
    {
      "source": "contract.txt",
      "chunk_id": "chunk_0",
      "score": 0.754
    }
  ],
  "status": "answered"
}
```

- **Guardrail / Out-of-Context Response (200 OK)**:
```json
{
  "answer": "I could not find this information in the provided documents.",
  "sources": [],
  "status": "refused"
}
```

- **Validation Error (422 Unprocessable Entity)**:
Triggered if `question` is shorter than 3 characters or longer than 1000 characters.

#### **`GET /health`** — System & Vector DB Health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "collection_name": "lextrace_documents",
  "collection_count": 4,
  "chat_model": "gemini-3.1-flash-lite",
  "embedding_model": "gemini-embedding-001"
}
```

---

### 3. Sample cURL Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"When can the agreement be terminated?"}'
```

---

### 4. Frontend Integration Guide

A frontend (React, Vue, Next.js, Streamlit) can POST user queries to `/query`:
- Render `answer` directly in the chat interface.
- Display `sources` as clickable citation pills or evidence cards showing `source`, `chunk_id`, and relevance `score`.
- Handle `status === "refused"` by displaying a friendly fallback badge.

---

## End-to-End RAG Evaluation System

Evaluating retrieval alone is insufficient. LexTrace includes an end-to-end RAG answer evaluation framework that scores answer quality across three critical dimensions:

```text
User Question
     ↓
RAG Pipeline (Retrieve + Grounded LLM Answering + Citation Attribution)
     ↓
Answer & Citations
     ↓
Multi-Dimensional Scoring:
  ├── 1. Correctness: Expected answer points covered (0.0 – 1.0)
  ├── 2. Grounding: Faithfulness to retrieved context / no hallucinations (0.0 – 1.0)
  └── 3. Citation Accuracy: Precision/Recall/F1 match with expected sources (0.0 – 1.0)
     ↓
Summary & Failure Diagnostics:
  ├── Aggregate Quality Metrics (avg_correctness, avg_grounding, avg_citation_accuracy)
  ├── Failure Root Cause Pinpointing (Weak Retrieval, Hallucination, Citation Misattribution, Fallback Failure)
  └── Actionable Remediation Plan
```

### Evaluation Dimensions

| Dimension | Metric / Question | How It Is Scored |
| :--- | :--- | :--- |
| **Correctness** | Does the answer match expected points? | Normalized semantic & keyword coverage ratio (`0.0` to `1.0`), including refusal verification. |
| **Grounding** | Are claims supported strictly by context? | Context containment ratio (`0.0` to `1.0`). Penalizes hallucinations and rewards faithful refusal. |
| **Citation Accuracy** | Do citations match expected sources? | F1 score between returned citations and expected source files (`0.0` to `1.0`). |

---

## Running Evaluations & Tests

Run the full automated test suite:
```bash
pytest tests/ -v
```

Run evaluation on LexTrace documents:
```bash
python src/evaluate_rag.py lextrace
```

Run evaluation on canonical concept rubric:
```bash
python src/evaluate_rag.py concept
```

---

## Further Reading & References

- [FastAPI - Request Body & Validation](https://fastapi.tiangolo.com/tutorial/body/)
- [FastAPI - Handling Errors](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [LangSmith - Evaluate a RAG application](https://docs.smith.langchain.com/evaluation/tutorials/rag)
- [Pinecone - Evaluating retrieval in RAG](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)
- [LlamaIndex - Evaluating RAG](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/)