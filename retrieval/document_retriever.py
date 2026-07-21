from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import sqlite3
import json

#Implémentation des chemins
BASE_DIR = Path(__file__).resolve().parents[1]

INDEX_FILE = (BASE_DIR / "data" / "indexes" / "faiss.index")

EMBEDDING_MODEL = ("sentence-transformers/all-MiniLM-L6-v2")

DATABASE_FILE = (BASE_DIR / "data" / "indexes" / "chunks.db")

def load_index(index_file: Path) -> faiss.Index:
    """
    Charge l'index FAISS depuis le disque.
    """

    return faiss.read_index(str(index_file))

def load_model(model_name: str) -> SentenceTransformer:
    """
    Charge le modèle SentenceTransformer.
    """

    return SentenceTransformer(model_name)

def encode_query(model: SentenceTransformer,query: str) -> np.ndarray:
        """
        Encode une requête utilisateur en embedding.
        """

        embedding = model.encode([query],convert_to_numpy=True,normalize_embeddings=True)

        return embedding.astype(np.float32) #Compatibilité

def search_index(index: faiss.Index,query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Recherche les vecteurs les plus proches
    dans l'index FAISS.
    """

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    return scores, indices

def load_database(database_file: Path) -> sqlite3.Connection:
    """
    Ouvre une connexion à la base de données
    contenant les chunks.
    """

    return sqlite3.connect(database_file)

def get_chunk(connection: sqlite3.Connection,chunk_id: int) -> dict | None:
    """
    Récupère un chunk à partir de son identifiant.
    """

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
        (chunk_id,)
    )

    row = cursor.fetchone()

    if row is None:
        return None
    
    return {
        "chunk_id": row[0],
        "doc_id": row[1],
        "title": row[2],
        "source": row[3],
        "start_sentence": row[4],
        "end_sentence": row[5],
        "sentence_ids": json.loads(row[6]),
        "num_sentences": row[7],
        "text_length": row[8],
        "text": row[9],
    }


def build_results(
    connection: sqlite3.Connection,
    scores: np.ndarray,
    indices: np.ndarray
) -> list[dict]:
    """
    Construit les résultats finaux
    à partir des résultats FAISS.
    """

    results = []

    for score, index in zip(
        scores[0],
        indices[0]
    ):

        if index < 0:
            continue

        chunk = get_chunk(
            connection,
            int(index)
        )

        if chunk is None:
            continue

        chunk["score"] = float(
            score
        )

        results.append(
            chunk
        )

    return results


#Regroupement des ressources dans une seule classe
class DocumentRetriever:
    def __init__(self,index_file: Path = INDEX_FILE,model_name: str = EMBEDDING_MODEL,database_file: Path = DATABASE_FILE):

        self.index = load_index(index_file)
        self.model = load_model(model_name)
        self.connection = load_database(database_file)

    
    def search(self,query: str, top_k: int = 10) -> list[dict]:
        """
        Recherche les chunks les plus pertinents
        pour une requête utilisateur.
        """

        query_embedding = encode_query(self.model,query)

        scores, indices = search_index(self.index,query_embedding,top_k)

        results = build_results(self.connection,scores,indices)

        return results


retriever = DocumentRetriever()

results = retriever.search(
    query="Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
    top_k=5
)

for result in results:
    print(f"Title: {result['title']}")
    print(f"Score: {result['score']}")
    print(f"Text: {result['text']}")