import os

from dotenv import load_dotenv
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from src.ingestion import run_ingestion


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")


def create_embeddings():
    # Get our already-cleaned and chunked documents
    _, nodes = run_ingestion() #Here, _, it is there at first bcz in run ingestion func we are only accessing the chunks not the docs as it is returning both.

    # Gemini embedding model
    embed_model = GoogleGenAIEmbedding(
        model_name="gemini-embedding-001",
        api_key=api_key,
    )

    # Generate an embedding for every chunk, here we are storing the numerical vector attached to the node itself like node.source and node.embeddings
    for node in nodes:
        node.embedding = embed_model.get_text_embedding(node.text)

    return nodes# We return the chunks with their embeddings attached.


if __name__ == "__main__":

    nodes = create_embeddings()

    print(f"\nTotal chunks embedded: {len(nodes)}")

    if nodes:
        print("Embedding dimension:", len(nodes[0].embedding)) # Each chunk is represented by these many nums.
        print("First 8 values:", nodes[0].embedding[:8]) # The list of numbers here is the embeddings vector for every chunk.
        #Here we are only printing the embeddings of the first node for verification.