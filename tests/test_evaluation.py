import pytest
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import (
    check_citations,
    diagnose_failure,
    judge_expected_points,
    judge_grounding,
    score_answer,
    summarize_failures,
)


class TestCorrectnessScorer:
    def test_full_correctness_match(self):
        answer = "Either party may terminate by giving 30 days written notice or immediately for material breach."
        expected_points = ["30 days", "written notice", "material breach"]
        score = judge_expected_points(answer, expected_points)
        assert score == 1.0

    def test_partial_correctness_match(self):
        answer = "The agreement requires 30 days written notice."
        expected_points = ["30 days", "written notice", "material breach"]
        score = judge_expected_points(answer, expected_points)
        assert score == pytest.approx(0.67, rel=1e-2)

    def test_no_correctness_match(self):
        answer = "The policy allows flexible working hours."
        expected_points = ["30 days", "written notice"]
        score = judge_expected_points(answer, expected_points)
        assert score == 0.0

    def test_refusal_guardrail_scoring(self):
        answer = "I could not find this information in the provided documents."
        expected_points = ["refuse", "say not enough information"]
        score = judge_expected_points(answer, expected_points)
        assert score == 1.0

    def test_false_refusal_penalty(self):
        answer = "I could not find this information in the provided documents."
        expected_points = ["30 days", "written notice"]
        score = judge_expected_points(answer, expected_points)
        assert score == 0.0


class TestGroundingScorer:
    def test_grounded_answer(self):
        context = ["Employees are entitled to 20 days of paid annual leave per year."]
        answer = "Employees receive 20 days of paid annual leave each year."
        score = judge_grounding(answer, retrieved_contexts=context)
        assert score == 1.0

    def test_hallucinated_answer(self):
        context = ["Employees are entitled to 20 days of paid annual leave per year."]
        answer = "Employees receive unlimited paid vacation, free flights, and bonus equity grants."
        score = judge_grounding(answer, retrieved_contexts=context)
        assert score <= 0.40

    def test_valid_refusal_is_grounded(self):
        context = ["Some irrelevant document text."]
        answer = "I could not find this information in the provided documents."
        score = judge_grounding(answer, retrieved_contexts=context)
        assert score == 1.0


class TestCitationAccuracyScorer:
    def test_exact_citation_match(self):
        citations = {"contract.txt"}
        expected = {"contract.txt"}
        score = check_citations(citations, expected)
        assert score == 1.0

    def test_multi_source_citation_match(self):
        citations = {"contract.txt", "employement_policy.md"}
        expected = {"contract.txt", "employement_policy.md"}
        score = check_citations(citations, expected)
        assert score == 1.0

    def test_empty_citations_for_guardrail(self):
        citations = set()
        expected = set()
        score = check_citations(citations, expected)
        assert score == 1.0

    def test_over_citation_penalty(self):
        citations = {"contract.txt", "regulation.html"}
        expected = {"contract.txt"}
        score = check_citations(citations, expected)
        assert score < 1.0
        assert score == pytest.approx(0.67, rel=1e-2)

    def test_mismatched_citation(self):
        citations = {"employement_policy.md"}
        expected = {"contract.txt"}
        score = check_citations(citations, expected)
        assert score == 0.0


class TestSummaryAndDiagnostics:
    def test_summarize_failures_aggregation(self):
        rows = [
            {
                "question": "Q1",
                "answer": "A1",
                "correctness": 1.0,
                "grounding": 1.0,
                "citation_accuracy": 1.0,
                "diagnoses": [],
            },
            {
                "question": "Q2",
                "answer": "A2",
                "correctness": 0.5,
                "grounding": 1.0,
                "citation_accuracy": 0.0,
                "diagnoses": ["Incomplete Answer", "Citation Misattribution"],
            },
        ]
        summary = summarize_failures(rows)
        assert summary["questions"] == 2
        assert summary["avg_correctness"] == 0.75
        assert summary["avg_grounding"] == 1.0
        assert summary["avg_citation_accuracy"] == 0.5
        assert len(summary["failures"]) == 1
        assert summary["failure_count"] == 1
        assert summary["pass_count"] == 1

    def test_diagnose_failure_reasons(self):
        failing_row = {
            "correctness": 0.0,
            "grounding": 0.3,
            "citation_accuracy": 0.0,
            "citations": [],
            "retrieved_sources": ["other_doc.md"],
            "expected_sources": ["contract.txt"],
            "answer": "Fabricated text not in document.",
        }
        diagnoses = diagnose_failure(failing_row)
        assert any("Weak Retrieval" in d for d in diagnoses)
        assert any("Ungrounded" in d for d in diagnoses)
        assert any("Missing Citation" in d for d in diagnoses)


class TestEndToEndScoreAnswer:
    def test_score_answer_with_mock_rag(self):
        example = {
            "question": "What is the notice period for contract termination?",
            "expected_points": ["30 days", "written notice"],
            "expected_sources": {"contract.txt"},
        }

        def mock_rag(q):
            return {
                "question": q,
                "answer": "The agreement may be terminated by providing 30 days written notice.",
                "citations": {"contract.txt"},
                "retrieved_sources": {"contract.txt", "NON.docx"},
                "retrieved_contexts": ["Either party may terminate this agreement by providing 30 days written notice."],
            }

        result = score_answer(example, rag_fn=mock_rag)
        assert result["correctness"] == 1.0
        assert result["grounding"] == 1.0
        assert result["citation_accuracy"] == 1.0
        assert result["diagnoses"] == []
