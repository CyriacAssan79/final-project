import unittest

from verification.verifier import Verifier


class FakeVerifier(Verifier):
    def __init__(self, responses):
        self.responses = list(responses)
        self.id2label = {}
        self.model_name = "fake-nli"

    def predict(self, claim: str, evidence: str) -> dict:
        if not self.responses:
            raise AssertionError("No fake NLI response left.")
        return self.responses.pop(0)


class VerifierTest(unittest.TestCase):
    def test_convert_to_fever_label(self):
        verifier = FakeVerifier([])

        self.assertEqual(verifier.convert_to_fever_label("entailment"), "SUPPORTS")
        self.assertEqual(verifier.convert_to_fever_label("contradiction"), "REFUTES")
        self.assertEqual(verifier.convert_to_fever_label("neutral"), "NOT ENOUGH INFO")

    def test_verify_returns_final_label_and_best_evidence(self):
        verifier = FakeVerifier(
            [
                {
                    "label": "contradiction",
                    "confidence": 0.9,
                    "probabilities": [0.05, 0.9, 0.05],
                },
                {
                    "label": "neutral",
                    "confidence": 0.7,
                    "probabilities": [0.1, 0.2, 0.7],
                },
            ]
        )
        evidence = [
            {
                "chunk_id": 1,
                "doc_id": "Andrew_Kevin_Walker",
                "title": "Andrew Kevin Walker",
                "text": "Andrew Kevin Walker is an American screenwriter.",
                "sentence_ids": [0],
                "score": 0.8,
                "rerank_score": 5.0,
            },
            {
                "chunk_id": 2,
                "doc_id": "Other",
                "title": "Other",
                "text": "Unrelated sentence.",
                "sentence_ids": [1],
                "score": 0.4,
                "rerank_score": 1.0,
            },
        ]

        result = verifier.verify("Andrew Kevin Walker is only Chinese.", evidence)

        self.assertEqual(result["label"], "REFUTES")
        self.assertAlmostEqual(result["confidence"], 0.9 / 1.6)
        self.assertEqual(result["best_evidence"]["doc_id"], "Andrew_Kevin_Walker")
        self.assertEqual(result["best_evidence"]["fever_label"], "REFUTES")
        self.assertEqual(len(result["evidence"]), 2)

    def test_verify_empty_evidence_returns_nei(self):
        verifier = FakeVerifier([])

        result = verifier.verify("A claim with no evidence.", [])

        self.assertEqual(result["label"], "NOT ENOUGH INFO")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIsNone(result["best_evidence"])


if __name__ == "__main__":
    unittest.main()
