from pathlib import Path
from typing import Iterator

import json

import faiss
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]

EMBEDDINGS_DIR = (
    BASE_DIR
    / "data"
    / "indexes"
    / "embeddings"
)

METADATA_FILE = (EMBEDDINGS_DIR/ "metadata.json")

INDEX_FILE = (BASE_DIR / "data" / "indexes" / "faiss.index")


def load_metadata(metadata_file: Path) -> dict:
    """
    Charge les métadonnées des embeddings.
    """

    with metadata_file.open("r",encoding="utf-8") as f:

        return json.load(f)
    
def load_embedding_files(embeddings_dir: Path) -> Iterator[Path]:
    """
    Parcourt tous les fichiers d'embeddings (.npy)
    dans l'ordre de leur création.
    """

    for file_path in sorted(embeddings_dir.glob("embeddings_*.npy")):
        yield file_path


def create_index(embedding_dimension: int) -> faiss.Index:
    """
    Crée un index FAISS basé sur
    la similarité cosinus.
    """

    return faiss.IndexFlatIP(embedding_dimension) #tous les vecteurs ont une norme égale à 1

def add_embeddings(index: faiss.Index, embedding_file: Path ) -> int:
    """
    Charge un fichier d'embeddings
    et les ajoute à l'index FAISS.

    Returns
    -------
    int
        Nombre de vecteurs ajoutés.
    """

    embeddings = np.load(
        embedding_file
    ).astype(np.float32)

    index.add(
        embeddings
    )

    return embeddings.shape[0]

def save_index(index: faiss.Index,output_file: Path ) -> None:
    """
    Sauvegarde l'index FAISS.
    """

    faiss.write_index(
        index,
        str(output_file)
    )


def main():

    metadata = load_metadata(METADATA_FILE)

    index = create_index(metadata["embedding_dimension"])

    total_vectors = 0

    for embedding_file in load_embedding_files(EMBEDDINGS_DIR):
        total_vectors += add_embeddings(index, embedding_file)

    save_index(index, INDEX_FILE)

    print("\n===== Index FAISS terminé =====")
    print(f"Vecteurs indexés : {total_vectors}")
    print(f"Dimension         : {metadata['embedding_dimension']}")

if __name__ == "__main__":
    main()