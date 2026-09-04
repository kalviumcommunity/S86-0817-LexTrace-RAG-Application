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
│   ├── chunking.py
│   ├── embeddings.py
│   ├── evaluate_rag.py
│   ├── evaluator.py
│   ├── ingestion.py
│   ├── llm_test.py
│   ├── prompt_template_test.py
│   ├── prompt_test.py
│   ├── rag_pipeline.py
│   ├── retrival.py
│   ├── structured_output_test.py
│   ├── temperature_test.py
│   ├── test_sets.py
│   ├── text_cleaner.py
│   └── token_test.py
├── tests/
│   └── test_evaluation.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Tech Stack

- Python
- Google Gemini Embeddings & Chat Models (`gemini-embedding-001`, `gemini-3.1-flash-lite`)
- ChromaDB (Vector Store)
- LlamaIndex
- python-dotenv & OpenAI SDK
- pytest

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

### 1. The Three Answer Evaluation Dimensions

| Dimension | Metric / Question | How It Is Scored |
| :--- | :--- | :--- |
| **Correctness** | Does the answer match the expected answer points? | Normalized semantic & keyword coverage ratio (`0.0` to `1.0`), including refusal verification for unanswerable questions. |
| **Grounding** | Are answer claims supported strictly by retrieved context? | Context containment ratio (`0.0` to `1.0`). Penalizes hallucinations and rewards faithful refusal when context is missing. |
| **Citation Accuracy** | Do citations point to the sources that actually support the claims? | F1 score between returned citations and expected source files (`0.0` to `1.0`), penalizing both under-citation and over-citation. |

### 2. Test Set Structure

Test cases define questions, expected answer points, and expected source references:

```python
test_set = [
    {
        "question": "What notice period is required for standard contract termination?",
        "expected_points": ["30 days", "written notice"],
        "expected_sources": {"contract.txt"}
    },
    {
        "question": "What is the company policy on international travel per diem?",
        "expected_points": ["could not find", "refuse", "no information"],
        "expected_sources": set()  # Missing context guardrail test
    }
]
```

### 3. Running Evaluations

Run the evaluation on the LexTrace corpus:

```bash
python src/evaluate_rag.py lextrace
```

Run the canonical concept test set:

```bash
python src/evaluate_rag.py concept
```

Run unit tests with pytest:

```bash
pytest tests/test_evaluation.py -v
```

### 4. Failure Diagnostics & Remediation

When evaluation scores drop below `1.0`, the system automatically diagnoses the failure cause and prescribes remediation steps:

- **If Correctness is low**: Inspect top-k retrieval count, chunk size/overlap boundaries, or query formulation in the prompt template.
- **If Grounding is low**: Strengthen system constraints (`temperature=0.0`, strict context-only instructions, fallback phrases).
- **If Citation Accuracy is low**: Fix document ingestion metadata (`source` field), enforce structured JSON citation output, and prune citations not present in the retrieved candidate set.

---

## Further Reading & References

- [LangSmith - Evaluate a RAG application](https://docs.smith.langchain.com/evaluation/tutorials/rag)
- [Pinecone - Evaluating retrieval in RAG](https://www.pinecone.io/learn/series/vector-databases-in-production-for-busy-engineers/rag-evaluation/)
- [LlamaIndex - Evaluating RAG](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/)