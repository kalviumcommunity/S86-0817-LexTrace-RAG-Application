import unittest
from unittest.mock import patch

from src.grounded_generation import (
    guarded_answer,
    retrieval_is_strong,
    rewrite_followup,
    conversational_answer,
)


class GuardrailTests(unittest.TestCase):
    def test_retrieval_is_strong_requires_threshold(self):
        self.assertFalse(retrieval_is_strong([]))
        self.assertTrue(retrieval_is_strong([{"score": 0.9}]))
        self.assertFalse(retrieval_is_strong([{"score": 0.7}]))
        self.assertTrue(retrieval_is_strong([{"score": 0.8}, {"score": 0.2}]))

    def test_guarded_answer_refuses_weak_context(self):
        def fake_retrieve(question, top_k=4, metadata_filter=None, **kwargs):
            return [{
                "rank": 1,
                "score": 0.61,
                "text": "This is not relevant enough.",
                "metadata": {"source": "unrelated.txt"},
                "distance": 0.39,
            }]

        def fake_generate(question, chunks, **kwargs):
            raise AssertionError("generation should not run when context is weak")

        with patch("src.grounded_generation.retrieve", side_effect=fake_retrieve), \
             patch("src.grounded_generation.generate_grounded_answer", side_effect=fake_generate):
            result = guarded_answer("What is the refund policy for a product not in this corpus?")

        self.assertEqual(result["status"], "refused_weak_context")
        self.assertEqual(result["answer"], "I don't have enough reliable context to answer that.")
        self.assertEqual(result["sources"], [])

    def test_guarded_answer_produces_answer_when_context_is_strong(self):
        def fake_retrieve(question, top_k=4, metadata_filter=None, **kwargs):
            return [{
                "rank": 1,
                "score": 0.93,
                "text": "A project submission requires all supporting evidence listed in the checklist.",
                "metadata": {"source": "contract.txt"},
                "distance": 0.07,
            }]

        with patch("src.grounded_generation.retrieve", side_effect=fake_retrieve), \
             patch("src.grounded_generation.generate_grounded_answer", return_value={
                 "answer": "The project submission requires all supporting evidence listed in the checklist.",
                 "question": "What evidence is required for project submission?",
                 "sources": [{"source": "contract.txt"}],
                 "context": "context",
                 "sources_count": 1,
                 "status": "success",
             }):
            result = guarded_answer("What evidence is required for project submission?")

        self.assertEqual(result["status"], "answered")
        self.assertIn("supporting evidence", result["answer"])
        self.assertTrue(result["sources"])

    def test_rewrite_followup_uses_conversation_history(self):
        history = [
            {"role": "user", "content": "What evidence is required for project submission?"},
            {"role": "assistant", "content": "The submission needs a PR link, sample output, and a video explanation with source evidence."},
        ]

        with patch("src.grounded_generation.call_llm", return_value="What video explanation is required for project submission?"):
            rewritten = rewrite_followup(history, "What about the video?")

        self.assertEqual(rewritten, "What video explanation is required for project submission?")

    def test_conversational_answer_rewrites_and_appends_history(self):
        history = [
            {"role": "user", "content": "What evidence is required for project submission?"},
            {"role": "assistant", "content": "The submission needs a PR link, sample output, and a video explanation with source evidence."},
        ]

        def fake_retrieve(question, top_k=4, metadata_filter=None, **kwargs):
            return [{
                "rank": 1,
                "score": 0.93,
                "text": "The video explanation must show the final project outcome and supporting evidence.",
                "metadata": {"source": "contract.txt"},
                "distance": 0.07,
            }]

        with patch("src.grounded_generation.call_llm", return_value="What video explanation is required for project submission?"), \
             patch("src.grounded_generation.retrieve", side_effect=fake_retrieve), \
             patch("src.grounded_generation.generate_grounded_answer", return_value={
                 "answer": "The video explanation must show the final project outcome and supporting evidence.",
                 "question": "What about the video?",
                 "sources": [{"source": "contract.txt"}],
                 "context": "context",
                 "sources_count": 1,
                 "status": "success",
             }):
            result = conversational_answer(history, "What about the video?")

        self.assertEqual(result["rewritten_query"], "What video explanation is required for project submission?")
        self.assertIn("video explanation", result["answer"])
        self.assertEqual(history[-2]["role"], "user")
        self.assertEqual(history[-1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
