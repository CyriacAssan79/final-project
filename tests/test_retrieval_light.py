import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from retrieval.document_retriever import build_results, get_chunk
from retrieval.passage_retriever import PassageRetriever


def create_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE chunks (
            chunk_id INTEGER PRIMARY KEY,
            doc_id TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            start_sentence INTEGER,
            end_sentence INTEGER,
            sentence_ids TEXT,
            num_sentences INTEGER,
            text_length INTEGER,
            text TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            0,
            "Doc_A",
            "Doc A",
            "wikipedia",
            0,
            1,
            json.dumps([0, 1]),
            2,
            12,
            "Known text.",
        ),
    )
    return connection


class FakeDocumentRetriever:
    def search(self, query: str, top_k: int):
        return [
            {"doc_id": "Doc_A", "chunk_id": 0, "title": "Doc A", "source": "wikipedia", "start_sentence": 0, "end_sentence": 0, "sentence_ids": [0], "num_sentences": 1, "text_length": 5, "text": "A", "score": 0.9},
            {"doc_id": "Doc_A", "chunk_id": 1, "title": "Doc A", "source": "wikipedia", "start_sentence": 1, "end_sentence": 1, "sentence_ids": [1], "num_sentences": 1, "text_length": 5, "text": "B", "score": 0.8},
            {"doc_id": "Doc_B", "chunk_id": 2, "title": "Doc B", "source": "wikipedia", "start_sentence": 0, "end_sentence": 0, "sentence_ids": [0], "num_sentences": 1, "text_length": 5, "text": "C", "score": 0.7},
        ]


class RetrievalLightTest(unittest.TestCase):
    def test_get_chunk_and_build_results(self):
        connection = create_connection()

        chunk = get_chunk(connection, 0)
        results = build_results(
            connection,
            scores=np.array([[0.75]], dtype=np.float32),
            indices=np.array([[0]], dtype=np.int64),
        )

        self.assertEqual(chunk["doc_id"], "Doc_A")
        self.assertEqual(chunk["sentence_ids"], [0, 1])
        self.assertEqual(results[0]["score"], 0.75)

    def test_passage_retriever_limits_chunks_per_document(self):
        passage_retriever = PassageRetriever(FakeDocumentRetriever())

        results = passage_retriever.retrieve(
            "query",
            top_k=3,
            max_chunks_per_document=1,
        )

        self.assertEqual([result["doc_id"] for result in results], ["Doc_A", "Doc_B"])


if __name__ == "__main__":
    unittest.main()
