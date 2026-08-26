import os
import time

from dotenv import load_dotenv
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from src.ingestion import run_ingestion


load_dotenv()

'''With this method, we dont make embeddings for indivudial chunks rather we store a group of them in 
batches and then we embedd once for batches so fewew API calls, more scalable when docs are higher, and 
if one batch temporarily fails, it tries automatically for every few seconds.'''

BATCH_SIZE = 16
MAX_ATTEMPTS = 5
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

    # # Generate an embedding for every chunk, here we are storing the numerical vector attached to the node itself like node.source and node.embeddings
    # for node in nodes:
    #     node.embedding = embed_model.get_text_embedding(node.text)

    # Process chunks in batches
    for start in range(0, len(nodes), BATCH_SIZE):

        batch = nodes[start:start + BATCH_SIZE]#The grp of chunks are stored in batches 
        texts = [node.text for node in batch]# chunk1 text, chunk2 text, it is stored like that.

        # Retry the batch if a temporary API error occurs
        for attempt in range(MAX_ATTEMPTS):

            try:
                embeddings = embed_model.get_text_embedding_batch(texts)
                break

            except Exception as error:

                if attempt == MAX_ATTEMPTS - 1:
                    raise

                wait_seconds = 2 ** attempt

                print(
                    f"Embedding failed. "
                    f"Retrying in {wait_seconds}s..."
                )

                time.sleep(wait_seconds)

        # Attach each generated vector to its chunk like node will have the embeddings, metadata and all attached.
        for node, embedding in zip(batch, embeddings):
            node.embedding = embedding

        print(
            f"Embedded {min(start + BATCH_SIZE, len(nodes))}"
            f"/{len(nodes)} chunks"
        )

    return nodes# We return the chunks with their embeddings attached.


if __name__ == "__main__":

    nodes = create_embeddings()

    print(f"\nTotal chunks embedded: {len(nodes)}")

    if nodes:
        print("Embedding dimension:", len(nodes[0].embedding)) # Each chunk is represented by these many nums.
        print("First 8 values:", nodes[0].embedding[:8]) # The list of numbers here is the embeddings vector for every chunk.
        #Here we are only printing the embeddings of the first node for verification.