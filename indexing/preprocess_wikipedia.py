from pathlib import Path
from typing import Iterator
from typing import List

import json

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
WIKI_DIR = BASE_DIR / "data" / "wikipedia"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "processed_chunks.jsonl"

SMALL_ARTICLE_THRESHOLD = 5

CHUNK_SIZE = 5
CHUNK_OVERLAP = 2
MAX_FILES = 10  # None = tous les fichiers


# load_articles()
# Responsabilite :
# parcourir tous les fichiers wiki
# lire en streaming
# retourner un article a la fois

def load_articles(wiki_dir: Path) -> Iterator[dict]:
    """
    Parcourt tous les fichiers wiki-*.jsonl et retourne les articles
    un par un afin d'eviter de charger tout le corpus en memoire.
    """

    files = sorted(wiki_dir.glob("wiki-*.jsonl"))
    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    for file_path in files:

        with file_path.open(
            "r",
            encoding="utf-8"
        ) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def extract_sentences(article: dict) -> List[dict]:
    """
    Extrait les phrases d'un article Wikipedia.

    Parameters
    ----------
    article : dict
        Article provenant du dataset Wikipedia.

    Returns
    -------
    List[dict]
        Liste de phrases au format {"id": sentence_id, "text": sentence_text}.
    """

    sentences = []

    lines = article.get("lines", "")

    if not lines:
        return sentences

    for line in lines.split("\n"):

        line = line.strip()

        if not line:
            continue

        parts = line.split("\t", maxsplit=1)

        if len(parts) != 2:
            continue

        sentence_id, sentence = parts

        try:
            sentence_id = int(sentence_id)
        except ValueError:
            continue

        sentence = sentence.strip()

        if not sentence:
            continue

        sentences.append({
            "id": sentence_id,
            "text": sentence,
        })

    return sentences


def create_chunks(
    sentences: list[dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    small_article_threshold: int = SMALL_ARTICLE_THRESHOLD,
) -> list[dict]:
    """
    Decoupe un article en chunks selon une strategie adaptative.

    - Petit article : un seul chunk.
    - Grand article : fenetres glissantes avec chevauchement.
    """

    if not sentences:
        return []

    # Cas 1 : petit article
    if len(sentences) <= small_article_threshold:

        return [{
            "start_sentence": sentences[0]["id"],
            "end_sentence": sentences[-1]["id"],
            "sentences": sentences
        }]

    chunks = []

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    for start in range(0, len(sentences), step):

        window = sentences[start:start + chunk_size]

        if not window:
            break

        chunks.append({
            "start_sentence": window[0]["id"],
            "end_sentence": window[-1]["id"],
            "sentences": window
        })

        # Si la derniere fenetre est plus petite que chunk_size,
        # on s'arrete.
        if len(window) < chunk_size:
            break

    return chunks


def build_chunk(
    chunk_id: int,
    article: dict,
    window: dict,
) -> dict:
    """
    Construit un chunk à partir d'une fenêtre de phrases.
    """

    sentences = window["sentences"]
    text = " ".join(sentence["text"] for sentence in sentences)

    return {

        "chunk_id": chunk_id,

        "doc_id": article["id"],

        "title": article["id"].replace("_", " "),

        "source": "wikipedia",

        "start_sentence": window["start_sentence"],

        "end_sentence": window["end_sentence"],

        "sentence_ids": [s["id"] for s in sentences],

        "num_sentences": len(sentences),

        "text_length": len(text),

        "text": text
    }

def save_chunk(chunk: dict, output_file) -> None:
    """
    Sauvegarde un chunk dans un fichier JSONL.

    Parameters
    ----------
    chunk : dict
        Chunk à sauvegarder.

    output_file : TextIO
        Fichier déjà ouvert en mode écriture.
    """

    output_file.write(
        json.dumps(chunk, ensure_ascii=False) #Le fichier sera beaucoup plus lisible.
    )

    output_file.write("\n")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    article_count = 0
    empty_articles = 0
    chunk_count = 0
    chunk_id = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as output_file:

        for article in load_articles(WIKI_DIR):

            article_count += 1

            sentences = extract_sentences(article)

            if not sentences:
                empty_articles += 1
                continue

            windows = create_chunks(sentences)

            for window in windows:

                chunk = build_chunk(
                    chunk_id=chunk_id,
                    article=article,
                    window=window
                )

                save_chunk(chunk, output_file)

                chunk_id += 1
                chunk_count += 1

    print("\n===== Prétraitement terminé =====")
    print(f"Articles traités : {article_count}")
    print(f"Articles vides   : {empty_articles}")
    print(f"Chunks générés   : {chunk_count}")


if __name__ == "__main__":
    main()
