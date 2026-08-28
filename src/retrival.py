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
    """Return the top-k chunks most similar to a user query."""

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
        where=metadata_filter,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "rank": i + 1,
            "score": 1 - distances[i],
            "text": documents[i],
            "metadata": metadatas[i],
            "distance": distances[i],
        }
        for i in range(len(documents))
    ]


def keyword_score(text, keywords):
    """Count the supplied keywords that occur in a chunk."""
    lowered_text = text.lower()
    return sum(keyword.lower() in lowered_text for keyword in keywords)


def hybrid_rank(vector_results, keywords, vector_weight=0.8, keyword_weight=0.2):
    """Rerank vector results using semantic and keyword scores."""
    ranked_results = []

    for result in vector_results:
        lexical_score = keyword_score(result["text"], keywords)
        hybrid_score = (
            vector_weight * result["score"]
            + keyword_weight * lexical_score
        )
        ranked_results.append({
            **result,
            "keyword_score": lexical_score,
            "hybrid_score": hybrid_score,
        })

    return sorted(
        ranked_results,
        key=lambda result: result["hybrid_score"],
        reverse=True
    )


def show_results(label, results):
    """Print retrieval results with their source and relevance details."""
    print(f"\n--- {label} ---")

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(f"\nRank: {rank}")
        print("Score:", round(result["score"], 4))
        print("Keyword score:", result.get("keyword_score", 0))
        print("Source:", metadata.get("source"))
        print("Text:", result["text"][:120])


if __name__ == "__main__":

    # First store our document embeddings
    store_embeddings()

    query = "When can the agreement be terminated?"

    unfiltered = retrieve(query, top_k=3)
    filtered = retrieve(
        query,
        top_k=3,
        metadata_filter={"source": "contract.txt"}
    )
    hybrid = hybrid_rank(
        filtered,
        keywords=["agreement", "terminated"]
    )

    show_results("Unfiltered retrieval", unfiltered)
    show_results("Filtered retrieval", filtered)
    show_results("Hybrid filtered retrieval", hybrid)

