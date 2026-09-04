import re

from src.retrival import retrieve, store_embeddings


CANDIDATE_K = 10
FINAL_K = 3


def _terms(text):
    """Return normalized terms while ignoring punctuation and short words."""
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) > 2
    }


def rerank_score(query, chunk):
    """Score candidate relevance using query-term coverage and vector score."""
    query_terms = _terms(query)
    chunk_terms = _terms(chunk["text"])
    lexical_score = (
        len(query_terms & chunk_terms) / len(query_terms)
        if query_terms
        else 0.0
    )

    return 0.7 * lexical_score + 0.3 * chunk["score"]


def rerank(query, candidates, final_k=FINAL_K):
    """Rerank candidates and return the highest-scoring final chunks."""
    reranked = [
        {
            **candidate,
            "rerank_score": rerank_score(query, candidate),
        }
        for candidate in candidates
    ]
    reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

    return reranked[:final_k]


def show_results(label, results):
    """Print retrieval order and scores for inspection."""
    print(f"\n--- {label} ---")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print("Vector score:", round(result["score"], 4))
        if "rerank_score" in result:
            print("Rerank score:", round(result["rerank_score"], 4))
        print("Source:", result["metadata"].get("source"))
        print("Text:", result["text"][:120])


if __name__ == "__main__":
    # Build the local Chroma collection before running the demonstration.
    store_embeddings()

    query = "When can the agreement be terminated?"
    candidates = retrieve(query, top_k=CANDIDATE_K)
    final_context = rerank(query, candidates, final_k=FINAL_K)

    show_results("Initial vector retrieval", candidates[:FINAL_K])
    show_results("After re-ranking", final_context)
    print(
        f"\nCompared {len(candidates)} candidates and kept "
        f"{len(final_context)} final chunks."
    )
