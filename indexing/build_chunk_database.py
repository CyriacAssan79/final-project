from pathlib import Path

import json
import sqlite3


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
)

DATABASE_FILE = (
    OUTPUT_DIR
    / "chunks.db"
)


#Structure de la base de données
def create_database(connection: sqlite3.Connection) -> None:
    """
    Crée la table contenant les chunks.
    """

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
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

    connection.commit()


def insert_chunk(connection: sqlite3.Connection, chunk: dict) -> None:
    """
    Insère un chunk dans la base de données.
    """

    connection.execute(
        """
        INSERT OR REPLACE INTO chunks (
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
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk["chunk_id"],
            chunk["doc_id"],
            chunk["title"],
            chunk["source"],
            chunk["start_sentence"],
            chunk["end_sentence"],
            json.dumps(chunk["sentence_ids"]),
            chunk["num_sentences"],
            chunk["text_length"],
            chunk["text"]
        )
    )


def load_chunks(
    input_file: Path
):
    """
    Charge les chunks un par un.
    """

    with input_file.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            yield json.loads(line)

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(
        DATABASE_FILE
    )

    create_database(
        connection
    )

    total_chunks = 0

    for chunk in load_chunks(
        INPUT_FILE
    ):

        insert_chunk(
            connection,
            chunk
        )

        total_chunks += 1

    connection.commit()

    connection.close()

    print(
        "\n===== Base de chunks créée ====="
    )

    print(
        f"Chunks insérés : {total_chunks}"
    )

if __name__ == "__main__":
    main()
