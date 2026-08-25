from src.text_cleaner import load_clean_documents
from src.chunking import chunk_documents


def run_ingestion():
    print("\n--- LexTrace Ingestion ---")

    # Load and clean documents
    documents = load_clean_documents()

    # Chunk the cleaned documents
    chunks = chunk_documents()

    print(f"Documents processed: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    # Show sample chunks
    print("\n--- Sample Chunks ---")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\nChunk {i + 1}")
        print(f"Source: {chunk.metadata.get('source')}")
        print(f"Characters: {len(chunk.text)}")
        print(f"Text: {chunk.text[:200]}")

    return documents, chunks


if __name__ == "__main__":
    run_ingestion()