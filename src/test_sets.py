"""
Test sets for RAG Answer Evaluation.

Contains:
1. CONCEPT_TEST_SET: Canonical demonstration test set from the evaluation rubric.
2. LEXTRACE_TEST_SET: Complete end-to-end test set matching local legal & policy documents in LexTrace.
"""

from typing import Any, Dict, List

# 1. Canonical concept test set
CONCEPT_TEST_SET: List[Dict[str, Any]] = [
    {
        "question": "What evidence is required for project submission?",
        "expected_points": ["PR link", "sample output", "video explanation"],
        "expected_sources": {"submission-rubric.md"},
    },
    {
        "question": "What should the system do when context is missing?",
        "expected_points": ["refuse", "say not enough information"],
        "expected_sources": {"guardrails.md"},
    },
]

# 2. Comprehensive LexTrace RAG test set covering all documents in data/sample/
LEXTRACE_TEST_SET: List[Dict[str, Any]] = [
    {
        "id": "lt-01-termination-notice",
        "question": "What notice period is required for standard contract termination?",
        "expected_points": [
            "30 days",
            "written notice",
        ],
        "expected_sources": {"contract.txt"},
        "document_type": "Service Agreement",
    },
    {
        "id": "lt-02-material-breach",
        "question": "When is immediate termination permitted under the service agreement?",
        "expected_points": [
            "material breach",
            "15 days",
            "written notice",
        ],
        "expected_sources": {"contract.txt"},
        "document_type": "Service Agreement",
    },
    {
        "id": "lt-03-payment-terms",
        "question": "What are the client's payment terms according to the agreement?",
        "expected_points": [
            "30 days",
            "invoice",
        ],
        "expected_sources": {"contract.txt"},
        "document_type": "Service Agreement",
    },
    {
        "id": "lt-04-annual-leave",
        "question": "How many days of paid annual leave are employees entitled to per year?",
        "expected_points": [
            "20 days",
            "annual leave",
        ],
        "expected_sources": {"employement_policy.md"},
        "document_type": "Employment Policy",
    },
    {
        "id": "lt-05-sick-leave-notice",
        "question": "What is the policy for sick leave and notifying managers before taking planned leave?",
        "expected_points": [
            "10 days",
            "sick leave",
            "24 hours",
        ],
        "expected_sources": {"employement_policy.md"},
        "document_type": "Employment Policy",
    },
    {
        "id": "lt-06-nda-duration",
        "question": "How long must confidential information remain protected under the non-disclosure agreement?",
        "expected_points": [
            "three years",
            "termination",
        ],
        "expected_sources": {"NON.docx"},
        "document_type": "Non-Disclosure Agreement",
    },
    {
        "id": "lt-07-data-regulation",
        "question": "What does the data protection regulation specify regarding data retention and data access?",
        "expected_points": [
            "not retained longer than necessary",
            "right to request access",
        ],
        "expected_sources": {"regulation.html"},
        "document_type": "Regulation",
    },
    {
        "id": "lt-08-missing-context-guardrail",
        "question": "What is the company policy on international travel per diem and visa reimbursement?",
        "expected_points": [
            "could not find",
            "not found",
            "no information",
            "refuse",
        ],
        "expected_sources": set(),  # No document contains this; expected citations must be empty
        "document_type": "Guardrail / Missing Context Test",
    },
]
