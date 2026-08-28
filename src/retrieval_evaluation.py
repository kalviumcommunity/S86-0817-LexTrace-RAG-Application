from src.retrival import retrieve


TEST_QUERIES = [
    {
        "query": "When can the service agreement be terminated?",
        "expected_source": "contract.txt",
    },
    {
        "query": "How many days of paid annual leave are employees entitled to?",
        "expected_source": "employement_policy.md",
    },
    {
        "query": "Can an individual request access to personal data?",
        "expected_source": "regulation.html",
    },
]


SETTINGS = [
    {
        "name": "baseline_k1",
        "top_k": 1,
        "metadata_filter": None,
        "min_score": 0.0,
    },
    {
        "name": "baseline_k3",
        "top_k": 3,
        "metadata_filter": None,
        "min_score": 0.0,
    },
    {
        "name": "contract_filter_k3",
        "top_k": 3,
        "metadata_filter": {"source": "contract.txt"},
        "min_score": 0.0,
    },
]


def evaluate(setting, test_queries=TEST_QUERIES):
    """Evaluate one retrieval configuration using source hit rate."""
    rows = []

    for item in test_queries:
        results = retrieve(
            item["query"],
            top_k=setting["top_k"],
            metadata_filter=setting["metadata_filter"],
        )
        kept_results = [
            result
            for result in results
            if result["score"] >= setting["min_score"]
        ]
        returned_sources = [
            result["metadata"].get("source")
            for result in kept_results
        ]

        rows.append({
            "query": item["query"],
            "expected_source": item["expected_source"],
            "returned_sources": returned_sources,
            "hit": item["expected_source"] in returned_sources,
        })

    return rows


def evaluate_settings(settings=SETTINGS, test_queries=TEST_QUERIES):
    """Return hit-rate summaries for all supplied retrieval settings."""
    summaries = []

    for setting in settings:
        rows = evaluate(setting, test_queries)
        hits = sum(row["hit"] for row in rows)
        summaries.append({
            "setting": setting["name"],
            "hit_rate": hits / len(rows) if rows else 0.0,
            "details": rows,
        })

    return summaries


def print_summary(summaries):
    """Print hit rates and per-query source matches."""
    for summary in summaries:
        print(
            f"{summary['setting']} hit_rate: "
            f"{summary['hit_rate']:.2%}"
        )
        for row in summary["details"]:
            status = "HIT" if row["hit"] else "MISS"
            print(
                f"  {status}: expected={row['expected_source']}, "
                f"returned={row['returned_sources']}"
            )


if __name__ == "__main__":
    summaries = evaluate_settings()
    print("\n--- Retrieval Evaluation ---")
    print_summary(summaries)

    best = max(summaries, key=lambda summary: summary["hit_rate"])
    print(f"\nBest setting by hit rate: {best['setting']}")
