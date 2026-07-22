import unittest

from api.pipeline import FactCheckerPipeline, PipelineSettings


class FakeDocumentRetriever:
    def close(self):
        pass


class FakePassageRetriever:
    def retrieve(self, query, top_k, max_chunks_per_document):
        return [
            {
                "chunk_id": 1,
                "doc_id": "Doc_A",
                "title": "Doc A",
                "text": "Evidence text.",
                "sentence_ids": [0],
                "score": 0.8,
            }
        ]


class FakeReranker:
    def rerank(self, query, passages, top_k):
        result = passages[0].copy()
        result["rerank_score"] = 4.2
        return [result]


class FakeVerifier:
    def verify(self, claim, evidence):
        return {
            "label": "SUPPORTS",
            "confidence": 0.88,
            "label_scores": {"SUPPORTS": 0.88, "REFUTES": 0.0, "NOT ENOUGH INFO": 0.0},
            "best_evidence": evidence[0],
            "evidence": evidence,
        }


class PipelineLightTest(unittest.TestCase):
    def test_pipeline_orchestrates_components(self):
        pipeline = FactCheckerPipeline(
            document_retriever=FakeDocumentRetriever(),
            passage_retriever=FakePassageRetriever(),
            reranker=FakeReranker(),
            verifier=FakeVerifier(),
            settings=PipelineSettings(retrieval_top_k=3, rerank_top_k=1),
        )

        result = pipeline.verify_claim("A supported claim.")

        self.assertEqual(result["label"], "SUPPORTS")
        self.assertEqual(result["best_evidence"]["rerank_score"], 4.2)
        self.assertEqual(result["settings"]["retrieval_top_k"], 3)

    def test_pipeline_rejects_empty_claim(self):
        pipeline = FactCheckerPipeline(
            document_retriever=FakeDocumentRetriever(),
            passage_retriever=FakePassageRetriever(),
            reranker=FakeReranker(),
            verifier=FakeVerifier(),
        )

        with self.assertRaises(ValueError):
            pipeline.verify_claim(" ")


if __name__ == "__main__":
    unittest.main()
