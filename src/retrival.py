import chromadb
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from dotenv import load_dotenv

from src.embeddings import create_embeddings


load_dotenv()

# ChromaDB storage location and collection name
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "lextrace_documents"

# Gemini model used to create embeddings
EMBED_MODEL = "gemini-embedding-001"


def get_collection():
    """Create or open the ChromaDB collection."""

    # Create a persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    # Open the collection if it exists, otherwise create it
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def store_embeddings():
    """Generate embeddings and store them in ChromaDB."""

    # Get our existing chunks with Gemini embeddings
    nodes = create_embeddings()

    # Get the ChromaDB collection
    collection = get_collection()

    # Store every chunk, embedding and metadata
    for i, node in enumerate(nodes):

        collection.upsert(
            ids=[f"chunk_{i}"],
            documents=[node.text],
            embeddings=[node.embedding],
            metadatas=[node.metadata]
        )

    print(f"Vectors stored: {collection.count()}")

    return collection


def retrieve(query, top_k=3, metadata_filter=None):
    """Find the most relevant chunks for a user query."""

    # Open the existing ChromaDB collection
    collection = get_collection()

    # Create the Gemini embedding model
    embed_model = GoogleGenAIEmbedding(
        model_name=EMBED_MODEL
    )

    # Convert the user's question into an embedding vector
    query_embedding = embed_model.get_text_embedding(query)

    # Search ChromaDB for the most similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=metadata_filter
    )

    return results


if __name__ == "__main__":

    # First store our document embeddings
    store_embeddings()

    # Test query
    query = "When can the agreement be terminated?"

    # Retrieve the most relevant chunks
    results = retrieve(query)

    print("\n--- Retrieval Results ---")

    # Display the retrieved chunks and their sources
    for i, text in enumerate(results["documents"][0]):

        print(f"\nResult {i + 1}")
        print("Source:", results["metadatas"][0][i])
        print("Distance:", results["distances"][0][i])
        print("Text:", text)

'''
Here, we are doing this bcz docs[0] says that first query in collections and same for metadates[0][i]
will give us the result belonging to its i.
i = 0
documents[0][0] → termination clause
metadatas[0][0] → contract.txt

i = 1
documents[0][1] → payment clause
metadatas[0][1] → contract.txt
'''