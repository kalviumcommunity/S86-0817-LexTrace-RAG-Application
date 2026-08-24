from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from src.text_cleaner import load_clean_documents


CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def chunk_documents():

    cleaned_documents = load_clean_documents()

    documents = [
        Document(
            text=document["text"],
            metadata={"source": document["source"]}
        )
        for document in cleaned_documents
    ]

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.get_nodes_from_documents(documents)


if __name__ == "__main__":

    nodes = chunk_documents()

    print(f"\nTotal chunks: {len(nodes)}")

    for i, node in enumerate(nodes[:5]):
        print(f"\n--- Chunk {i + 1} ---")
        print("Source:", node.metadata.get("source"))
        print("Characters:", len(node.text))
        print(node.text[:300])