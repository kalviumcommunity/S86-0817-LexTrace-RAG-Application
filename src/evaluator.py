import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import FALLBACK_MESSAGE, answer_with_citations

REFUSAL_INDICATORS = [
    "could not find",
    "not found",
    "not provided",
    "not mentioned",
    "cannot find",
    "no information",
    "not enough information",
    "insufficient information",
    "refuse",
]

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "must", "may", "might"
}


def _normalize_text(text: str) -> str:
    """Normalize text by lowercasing and standardizing whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())


def _normalize_source_set(sources: Union[Set[str], List[str]]) -> Set[str]:
    """Normalize source filenames (strip whitespace, lowercase, basename)."""
    normalized = set()
    for s in sources:
        cleaned = Path(str(s).strip().lower()).name
        if cleaned:
            normalized.add(cleaned)
    return normalized


def judge_expected_points(answer: str, expected_points: List[str]) -> float:
    """
    Score correctness: Does the answer contain the expected answer points?
    Returns a score between 0.0 and 1.0 (proportion of expected points covered).
    """
    if not expected_points:
        return 1.0

    normalized_answer = _normalize_text(answer)
    if not normalized_answer:
        return 0.0

    is_answer_refusal = any(
        ind in normalized_answer for ind in ["could not find", "not found", "no information", "cannot find"]
    )

    matched = 0
    for point in expected_points:
        point_norm = _normalize_text(point)

        # Check if expected point represents a refusal/guardrail test
        is_refusal_point = any(ref_word in point_norm for ref_word in REFUSAL_INDICATORS)

        if is_refusal_point:
            if is_answer_refusal:
                matched += 1
            continue

        # Check if the answer is a refusal when a factual answer was expected
        if is_answer_refusal:
            continue

        # Check exact or token-based inclusion
        if point_norm in normalized_answer:
            matched += 1
        else:
            # Check if all significant tokens of the expected point appear in answer
            point_tokens = [w for w in re.findall(r"\b\w+\b", point_norm) if w not in STOPWORDS]
            if point_tokens and all(tok in normalized_answer for tok in point_tokens):
                matched += 1

    return round(matched / len(expected_points), 2)


def judge_grounding(
    answer: str,
    retrieved_contexts: Optional[List[str]] = None,
    retrieved_sources: Optional[Union[Set[str], List[str]]] = None
) -> float:
    """
    Score grounding: Are answer claims supported strictly by the retrieved context?
    Returns 1.0 if claims are fully grounded or if it appropriately refuses ungrounded context.
    """
    normalized_answer = _normalize_text(answer)
    if not normalized_answer:
        return 0.0

    is_refusal = any(
        ind in normalized_answer for ind in ["could not find", "not found", "no information", "cannot find"]
    )

    if is_refusal:
        # Faithful adherence to grounding when context is insufficient
        return 1.0

    if not retrieved_contexts:
        # Substantive answer without any context -> 0.0 (pure hallucination)
        return 0.0

    combined_context = _normalize_text(" ".join(retrieved_contexts))
    if not combined_context:
        return 0.0

    # Extract informative tokens (length >= 3, non-stopwords)
    answer_tokens = [
        w for w in re.findall(r"\b[a-zA-Z0-9_-]+\b", normalized_answer)
        if len(w) >= 3 and w not in STOPWORDS
    ]

    if not answer_tokens:
        return 1.0

    grounded_tokens = sum(1 for tok in answer_tokens if tok in combined_context)
    grounding_ratio = grounded_tokens / len(answer_tokens)

    # If vast majority (>80%) of informative tokens appear in context, consider well-grounded
    if grounding_ratio >= 0.80:
        return 1.0
    elif grounding_ratio >= 0.50:
        return round(grounding_ratio, 2)
    else:
        return round(max(0.0, grounding_ratio * 0.5), 2)


def check_citations(
    citations: Union[Set[str], List[str]],
    expected_sources: Union[Set[str], List[str]]
) -> float:
    """
    Score citation accuracy: Do citations point to the sources that actually support the claims?
    Returns F1 score between generated citations and expected sources (0.0 to 1.0).
    """
    gen_set = _normalize_source_set(citations)
    exp_set = _normalize_source_set(expected_sources)

    # Both empty -> perfect score
    if not gen_set and not exp_set:
        return 1.0

    # One empty, other not -> mismatch
    if not gen_set or not exp_set:
        return 0.0

    intersection = gen_set & exp_set
    if not intersection:
        return 0.0

    precision = len(intersection) / len(gen_set)
    recall = len(intersection) / len(exp_set)

    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 2)


def diagnose_failure(row: Dict[str, Any]) -> List[str]:
    """Identify specific root causes for a failure."""
    diagnoses = []
    correctness = row.get("correctness", 1.0)
    grounding = row.get("grounding", 1.0)
    citation_accuracy = row.get("citation_accuracy", 1.0)
    retrieved_sources = _normalize_source_set(row.get("retrieved_sources", set()))
    expected_sources = _normalize_source_set(row.get("expected_sources", set()))
    answer = row.get("answer", "").lower()

    # 1. Check Retrieval Failure
    if expected_sources and not (expected_sources & retrieved_sources):
        diagnoses.append("Weak Retrieval: Required document was not retrieved in top-k")

    # 2. Check Grounding / Hallucination Failure
    if grounding < 1.0:
        diagnoses.append("Ungrounded Content: Answer contains claims unsupported by retrieved context")

    # 3. Check Citation Failure
    if citation_accuracy < 1.0:
        if not row.get("citations") and expected_sources:
            diagnoses.append("Missing Citation: No citation returned when sources were expected")
        elif row.get("citations") and not expected_sources:
            diagnoses.append("Over-citation: Cited source for an unsupported/refusal question")
        else:
            diagnoses.append("Citation Misattribution: Generated citations do not match expected sources")

    # 4. Check Correctness / Content Failure
    if correctness < 1.0:
        if "could not find" in answer and expected_sources & retrieved_sources:
            diagnoses.append("False Refusal: Model refused to answer despite context being available")
        elif "could not find" not in answer and not expected_sources:
            diagnoses.append("Failed Refusal: Model answered without sufficient context instead of refusing")
        else:
            diagnoses.append("Incomplete Answer: Answer missed one or more required expected points")

    if not diagnoses and min(correctness, grounding, citation_accuracy) < 1.0:
        diagnoses.append("Partial Match: One or more evaluation dimensions scored below 1.0")

    return diagnoses


def score_answer(example: Dict[str, Any], rag_fn: Callable = answer_with_citations) -> Dict[str, Any]:
    """
    Score a single test example across:
    - Correctness
    - Grounding
    - Citation Accuracy
    """
    question = example["question"]
    expected_points = example.get("expected_points", [])
    expected_sources = example.get("expected_sources", set())

    # Run RAG answer pipeline
    result = rag_fn(question)

    answer = result.get("answer", "")
    citations = result.get("citations", set())
    retrieved_contexts = result.get("retrieved_contexts", [])
    retrieved_sources = result.get("retrieved_sources", set())

    correctness = judge_expected_points(
        answer=answer,
        expected_points=expected_points
    )

    grounding = judge_grounding(
        answer=answer,
        retrieved_contexts=retrieved_contexts,
        retrieved_sources=citations
    )

    citation_accuracy = check_citations(
        citations=citations,
        expected_sources=expected_sources
    )

    row = {
        "question": question,
        "answer": answer,
        "correctness": correctness,
        "grounding": grounding,
        "citation_accuracy": citation_accuracy,
        "citations": sorted(list(citations)),
        "expected_sources": sorted(list(expected_sources)),
        "expected_points": expected_points,
        "retrieved_sources": sorted(list(retrieved_sources)),
    }

    row["diagnoses"] = diagnose_failure(row)
    return row


def summarize_failures(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate evaluation metrics and pinpoint failures.
    """
    if not rows:
        return {
            "questions": 0,
            "avg_correctness": 0.0,
            "avg_grounding": 0.0,
            "avg_citation_accuracy": 0.0,
            "failures": []
        }

    total = len(rows)
    avg_correctness = sum(r["correctness"] for r in rows) / total
    avg_grounding = sum(r["grounding"] for r in rows) / total
    avg_citation_accuracy = sum(r["citation_accuracy"] for r in rows) / total

    failures = [
        r for r in rows
        if min(r["correctness"], r["grounding"], r["citation_accuracy"]) < 1.0
    ]

    # Collect diagnosis counts
    diagnosis_counts: Dict[str, int] = {}
    for f in failures:
        for d in f.get("diagnoses", []):
            diagnosis_counts[d] = diagnosis_counts.get(d, 0) + 1

    return {
        "questions": total,
        "avg_correctness": round(avg_correctness, 3),
        "avg_grounding": round(avg_grounding, 3),
        "avg_citation_accuracy": round(avg_citation_accuracy, 3),
        "failures": failures,
        "failure_count": len(failures),
        "pass_count": total - len(failures),
        "diagnosis_breakdown": diagnosis_counts,
    }


def diagnose_and_recommend(summary: Dict[str, Any]) -> str:
    """
    Provide targeted recommendations based on the weakest evaluation dimension.
    """
    scores = {
        "Correctness": summary.get("avg_correctness", 0.0),
        "Grounding": summary.get("avg_grounding", 0.0),
        "Citation Accuracy": summary.get("avg_citation_accuracy", 0.0),
    }

    weakest_dim = min(scores, key=scores.get)
    min_score = scores[weakest_dim]

    lines = [
        f"\n=======================================================",
        f"               RAG EVALUATION SUMMARY",
        f"=======================================================",
        f"Total Questions Evaluated : {summary.get('questions', 0)}",
        f"Passed (Perfect 1.0)      : {summary.get('pass_count', 0)}",
        f"Failures (Score < 1.0)    : {summary.get('failure_count', 0)}",
        f"-------------------------------------------------------",
        f"Average Correctness       : {summary.get('avg_correctness', 0.0):.2%}",
        f"Average Grounding         : {summary.get('avg_grounding', 0.0):.2%}",
        f"Average Citation Accuracy : {summary.get('avg_citation_accuracy', 0.0):.2%}",
        f"=======================================================",
    ]

    if summary.get("failures"):
        lines.append("\n[!] NOTABLE FAILURES & DIAGNOSTICS:")
        for idx, f in enumerate(summary["failures"], 1):
            lines.append(f"\n  Failure #{idx}:")
            lines.append(f"  • Question : {f['question']}")
            lines.append(f"  • Scores   : Correctness={f['correctness']} | Grounding={f['grounding']} | Citation={f['citation_accuracy']}")
            lines.append(f"  • Citations: {f['citations']} (Expected: {f['expected_sources']})")
            lines.append(f"  • Answer   : {f['answer'][:120]}..." if len(f['answer']) > 120 else f"  • Answer   : {f['answer']}")
            for diag in f.get("diagnoses", []):
                lines.append(f"    --> Diagnostic: {diag}")

    lines.append(f"\n-------------------------------------------------------")
    lines.append(f"[*] ACTIONABLE REMEDIATION PLAN (Weakest: {weakest_dim} @ {min_score:.2%}):")

    if weakest_dim == "Correctness" and min_score < 1.0:
        lines.append(
            "  -> Improve Retrieval & Prompt Instructions:\n"
            "     1. Check if top_k chunks capture all key terms; increase top_k if needed.\n"
            "     2. Refine chunking size and sentence overlap to prevent splitting essential facts.\n"
            "     3. Ensure prompt explicitly guides the model to cover all multi-part questions."
        )
    elif weakest_dim == "Grounding" and min_score < 1.0:
        lines.append(
            "  -> Strengthen Context-Only Guardrails & Fallback:\n"
            "     1. Enforce strict temperature=0.0 to prevent creative extrapolations.\n"
            "     2. Strengthen the system prompt constraint: 'Answer ONLY using the provided context'.\n"
            "     3. Ensure the fallback message is triggered whenever context similarity is low."
        )
    elif weakest_dim == "Citation Accuracy" and min_score < 1.0:
        lines.append(
            "  -> Fix Metadata and Citation Attribution:\n"
            "     1. Ensure document ingestion attaches clean, non-null 'source' metadata to every chunk.\n"
            "     2. Require structured JSON output mapping answers to exact source filenames.\n"
            "     3. Filter out citations that were not in the retrieved chunk candidate set."
        )
    else:
        lines.append("  -> All dimensions scored perfectly! RAG pipeline is well-tuned.")

    lines.append(f"=======================================================\n")
    return "\n".join(lines)
