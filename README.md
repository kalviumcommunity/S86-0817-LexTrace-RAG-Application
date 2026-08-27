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
│       ├── contract.txt
│       ├── employement_policy.md
│       └── regulation.html
├── prompts/
│   └── answer.py
├── src/
│   ├── chunking.py
│   ├── embeddings.py
│   ├── ingestion.py
│   ├── llm_test.py
│   ├── prompt_template_test.py
│   ├── prompt_test.py
│   ├── retrival.py
│   ├── structured_output_test.py
│   ├── temperature_test.py
│   ├── text_cleaner.py
│   └── token_test.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Tech Stack

- Python
- Google Gemini Embeddings
- ChromaDB
- python-dotenv
- LlamaIndex (or related retrieval tooling, depending on usage)
- Document processing utilities for text and HTML content

---

## Notes

This project demonstrates a practical RAG workflow using document ingestion, cleaning, chunking, embeddings, vector storage, and retrieval for answering user questions from local documents.