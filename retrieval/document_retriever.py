import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from config.setting import (CHUNKS_DATABASE_FILE, EMBEDDING_MODEL_NAME,FAISS_INDEX_FILE,)

def _missing_dependency_message(package: str) -> str:
    return f"Missing dependency '{package}'. Activate the project environment before running retrieval, for example: conda activate fact-checker."

def load_index(index_file: Path = FAISS_INDEX_FILE) -> Any:
    if not index_file.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_file}")

    try:
        import faiss
    except ImportError as exc:
        raise ImportError(_missing_dependency_message("faiss")) from exc

    return faiss.read_index(str(index_file))

def load_model(model_name: str = EMBEDDING_MODEL_NAME) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(_missing_dependency_message("sentence-transformers")) from exc

    return SentenceTransformer(model_name)

def encode_query(model: Any, query: str) -> np.ndarray:
    query = query.strip()

    if not query:
        raise ValueError("Query must not be empty.")

    embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding.astype(np.float32)

def search_index(index: Any,query_embedding: np.ndarray,top_k: int,) -> tuple[np.ndarray, np.ndarray]:


    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    return index.search(query_embedding, top_k)

def load_database(database_file: Path = CHUNKS_DATABASE_FILE) -> sqlite3.Connection:
    if not database_file.exists():
        raise FileNotFoundError(f"SQLite chunks database not found: {database_file}")

    connection = sqlite3.connect(str(database_file))
    connection.row_factory = sqlite3.Row

    return connection

def get_chunk(connection: sqlite3.Connection,chunk_id: int,) -> dict | None:

    cursor = connection.execute(
        """
        SELECT
            chunk_id,
            doc_id,
            title,
            source,
            start_sentence,
            end_sentence,
            sentence_ids,
            num_sentences,
            text_length,
            text
        FROM chunks
        WHERE chunk_id = ?
        """,
        (chunk_id,),
    )

    row = cursor.fetchone()

    if row is None:
        return None

    return {
        "chunk_id": row["chunk_id"],
        "doc_id": row["doc_id"],
        "title": row["title"],
        "source": row["source"],
        "start_sentence": row["start_sentence"],
        "end_sentence": row["end_sentence"],
        "sentence_ids": json.loads(row["sentence_ids"]),
        "num_sentences": row["num_sentences"],
        "text_length": row["text_length"],
        "text": row["text"],
    }

def build_results(
    connection: sqlite3.Connection,
    scores: np.ndarray,
    indices: np.ndarray,
    ) -> list[dict]:
    results = []

    for score, index in zip(scores[0], indices[0]):

        if index < 0:
            continue

        chunk = get_chunk(connection, int(index))

        if chunk is None:
            continue

        chunk["score"] = float(score)
        results.append(chunk)

    return results

class DocumentRetriever:
    def __init__(self,index_file: Path = FAISS_INDEX_FILE,model_name: str = EMBEDDING_MODEL_NAME,database_file: Path = CHUNKS_DATABASE_FILE,):

        self.index_file = index_file
        self.database_file = database_file
        self.model_name = model_name

        self.index = load_index(index_file)
        self.model = load_model(model_name)

        if getattr(self.index, "ntotal", 0) <= 0:
            raise ValueError(f"FAISS index contains no vectors: {index_file}")

    def search(self,query: str,top_k: int = 10,) -> list[dict]:

        query_embedding = encode_query(self.model,query,)

        scores, indices = search_index(
            self.index,
            query_embedding,
            top_k,
        )

        # Connexion SQLite créée dans le thread courant
        with load_database(self.database_file) as connection:

            return build_results(
                connection,
                scores,
                indices,
            )

    def close(self) -> None:
        # Plus aucune connexion persistante à fermer.
        pass

    def __enter__(self) -> "DocumentRetriever":
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:

        self.close()                                        
