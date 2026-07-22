from pathlib import Path
from typing import Iterator

import json
import numpy as np

from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "processed_chunks.jsonl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "indexes"
    / "embeddings"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 512

def load_chunks(input_file: Path) -> Iterator[dict]:
    """
    Lit les chunks un par un depuis le fichier JSONL.
    """

    with input_file.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():
                yield json.loads(line)

def load_model(model_name: str) -> SentenceTransformer:
    """
    Charge le modèle SentenceTransformer utilisé
    pour générer les embeddings.
    """
    print("Chargement du modèle...")

    model = SentenceTransformer(model_name)

    print("Modèle chargé.")

    return model

def encode_batch(
    model: SentenceTransformer,
    texts: list[str]
) -> np.ndarray:
    """
    Encode un batch de textes en embeddings.
    """

    embeddings = model.encode(
        texts,
        batch_size=len(texts),
        show_progress_bar=False, #Le pipeline aura déjà sa propre progression dans main().
        convert_to_numpy=True, #On évite donc une conversion supplémentaire.
        normalize_embeddings=True #En normalisant dès cette étape : on évite de le refaire plus tard ; la recherche est plus rapide ; les scores sont plus cohérents. C'est une pratique courante avec les modèles Sentence Transformers.
    )

    return embeddings


# Au lieu d'avoir un seul énorme fichier, nous allons produire quelque chose comme :

# data/
# └── indexes/
#     └── embeddings/
#         embeddings_000001.npy
#         embeddings_000002.npy
#         embeddings_000003.npy
#         ...
def save_batch(
    embeddings: np.ndarray,
    output_dir: Path,
    batch_index: int
) -> None:
    """
    Sauvegarde un batch d'embeddings au format NumPy.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir /
        f"embeddings_{batch_index:06d}.npy"
    )

    np.save(
        output_file,
        embeddings
    )

def save_metadata(
    output_dir: Path,
    model_name: str,
    batch_size: int,
    embedding_dimension: int,
    total_chunks: int,
    total_batches: int
) -> None:
    """
    Sauvegarde les métadonnées des embeddings.
    """

    metadata = {
        "model": model_name,
        "batch_size": batch_size,
        "embedding_dimension": embedding_dimension,
        "total_chunks": total_chunks,
        "total_batches": total_batches
    }

    output_file = output_dir / "metadata.json"

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=4
        )

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model = load_model(MODEL_NAME)
    batch_texts = []
    batch_index = 0
    total_chunks = 0
    embedding_dimension = None

    for chunk in load_chunks(INPUT_FILE):

        batch_texts.append(
            chunk["text"]
        )

        total_chunks += 1

        if len(batch_texts) == BATCH_SIZE:

            embeddings = encode_batch(
                model,
                batch_texts
            )

            if embedding_dimension is None:
                embedding_dimension = int(embeddings.shape[1])

            save_batch(
                embeddings,
                OUTPUT_DIR,
                batch_index
            )

            batch_index += 1

            batch_texts.clear()

    # Traiter le dernier batch s'il n'est pas vide
    if batch_texts:

        embeddings = encode_batch(
            model,
            batch_texts
        )

        if embedding_dimension is None:
            embedding_dimension = int(embeddings.shape[1])

        save_batch(
            embeddings,
            OUTPUT_DIR,
            batch_index
        )

        batch_index += 1

    save_metadata(
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        embedding_dimension=embedding_dimension,
        total_chunks=total_chunks,
        total_batches=batch_index
    )
    print("\n===== Embeddings terminés =====")
    print(f"Chunks traités : {total_chunks}")
    print(f"Batches créés  : {batch_index}")

if __name__ == "__main__":
    main()
