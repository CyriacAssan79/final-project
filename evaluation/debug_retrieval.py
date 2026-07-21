import json
from pathlib import Path

from retrieval.document_retriever import DocumentRetriever
from retrieval.passage_retriever import PassageRetriever
from retrieval.reranker import Reranker


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DEV_FILE = (
    BASE_DIR
    / "data"
    / "fever"
    / "shared_task_dev.jsonl"
)


# =============================================================================
# CHARGEMENT D'UN EXEMPLE FEVER
# =============================================================================

def load_examples(file_path: Path) -> list[dict]:

    examples = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():

                examples.append(
                    json.loads(line)
                )

    return examples


# =============================================================================
# EXTRACTION DES PREUVES GOLD
# =============================================================================

def extract_gold_evidence(example: dict) -> set[tuple[str, int]]:

    gold_evidence = set()

    for evidence_set in example["evidence"]:

        for evidence in evidence_set:

            if len(evidence) < 4:
                continue

            doc_id = evidence[2]
            sentence_id = evidence[3]

            if doc_id is None or sentence_id is None:
                continue

            gold_evidence.add(
                (
                    doc_id,
                    sentence_id
                )
            )

    return gold_evidence

def check_gold_coverage(
    example: dict,
    connection
) -> dict:
    """
    Vérifie si les documents contenant les preuves gold
    existent dans la base SQLite.
    """

    gold_evidence = extract_gold_evidence(example)

    coverage = []

    for doc_id, sentence_id in gold_evidence:

        cursor = connection.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE doc_id = ?
            """,
            (doc_id,)
        )

        count = cursor.fetchone()[0]

        coverage.append({
            "doc_id": doc_id,
            "sentence_id": sentence_id,
            "exists": count > 0,
            "num_chunks": count
        })

    return coverage

def gold_evidence_exists_in_database(
    gold_evidence: set[tuple[str, int]],
    connection
) -> bool:
    """
    Vérifie si au moins un document contenant
    une preuve gold existe dans la base.
    """

    for doc_id, sentence_id in gold_evidence:

        cursor = connection.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE doc_id = ?
            """,
            (doc_id,)
        )

        count = cursor.fetchone()[0]

        if count > 0:

            return True

    return False


# =============================================================================
# DEBUG
# =============================================================================
def debug_example(example: dict,passage_retriever: PassageRetriever,reranker: Reranker):
    claim = example["claim"]

    gold_evidence = extract_gold_evidence(
        example
    )

    # =========================================================================
    # CLAIM
    # =========================================================================

    print()
    print("=" * 100)
    print("CLAIM")
    print("=" * 100)

    print(claim)

    # =========================================================================
    # GOLD EVIDENCE
    # =========================================================================

    print()
    print("=" * 100)
    print("GOLD EVIDENCE")
    print("=" * 100)

    if not gold_evidence:

        print("Aucune preuve gold.")

        return

    else:

        for doc_id, sentence_id in gold_evidence:

            print()
            print(f"Document : {doc_id}")
            print(f"Sentence : {sentence_id}")

    # =========================================================================
    # COUVERTURE DU CORPUS
    # =========================================================================

    coverage = check_gold_coverage(
        example,
        passage_retriever.document_retriever.connection
    )

    print()
    print("=" * 100)
    print("COUVERTURE DES PREUVES GOLD")
    print("=" * 100)

    covered_evidence = []

    for item in coverage:

        print()
        print(f"DOC ID : {item['doc_id']}")
        print(f"SENTENCE ID : {item['sentence_id']}")
        print(f"DOCUMENT PRÉSENT : {item['exists']}")
        print(f"NOMBRE DE CHUNKS : {item['num_chunks']}")

        if item["exists"]:

            covered_evidence.append(
                (
                    item["doc_id"],
                    item["sentence_id"]
                )
            )

    # -------------------------------------------------------------------------
    # Vérification de la couverture
    # -------------------------------------------------------------------------

    if not covered_evidence:

        print()
        print("=" * 100)
        print("RÉSULTAT DE LA COUVERTURE")
        print("=" * 100)

        print(
            "❌ Aucune preuve gold n'est présente dans le corpus indexé."
        )

        print(
            "Le retrieval ne peut donc pas récupérer cette preuve."
        )

        return

    else:

        print()
        print("=" * 100)
        print("RÉSULTAT DE LA COUVERTURE")
        print("=" * 100)

        print(
            f"✅ {len(covered_evidence)} preuve(s) gold "
            f"présente(s) dans le corpus."
        )

    # =========================================================================
    # RETRIEVAL FAISS
    # =========================================================================

    passages = passage_retriever.retrieve(
        query=claim,
        top_k=50
    )

    # =========================================================================
    # RECHERCHE DE LA PREUVE GOLD DANS LES RÉSULTATS FAISS
    # =========================================================================

    gold_matches = find_gold_in_results(
        passages,
        gold_evidence
    )

    print()
    print("=" * 100)
    print("RECHERCHE DE LA PREUVE GOLD DANS LES RÉSULTATS FAISS")
    print("=" * 100)

    if gold_matches:

        for match in gold_matches:

            print()
            print(f"RANG FAISS : {match['rank']}")
            print(f"DOC ID : {match['doc_id']}")
            print(f"TITRE : {match['title']}")
            print(f"SENTENCE ID : {match['sentence_id']}")
            print(f"SCORE : {match['score']}")

    else:

        print(
            "❌ La preuve gold n'est pas présente dans les "
            "résultats FAISS."
        )

    # =========================================================================
    # RÉSULTATS FAISS
    # =========================================================================

    print()
    print("=" * 100)
    print("RÉSULTATS FAISS")
    print("=" * 100)

    for rank, passage in enumerate(
        passages,
        start=1
    ):

        is_gold = any(
            (
                passage["doc_id"],
                sentence_id
            )
            in gold_evidence
            for sentence_id in passage["sentence_ids"]
        )

        print()
        print(f"RANG : {rank}")
        print(f"TITRE : {passage['title']}")
        print(f"DOC ID : {passage['doc_id']}")
        print(f"SCORE FAISS : {passage['score']}")
        print(f"SENTENCE IDS : {passage['sentence_ids']}")
        print(f"PREUVE GOLD : {is_gold}")
        print(f"TEXTE : {passage['text'][:500]}")

    # =========================================================================
    # RERANKING
    # =========================================================================

    reranked_results = reranker.rerank(
        query=claim,
        passages=passages,
        top_k=10
    )

    # =========================================================================
    # RÉSULTATS APRÈS RERANKING
    # =========================================================================

    print()
    print("=" * 100)
    print("RÉSULTATS APRÈS RERANKING")
    print("=" * 100)

    for rank, passage in enumerate(
        reranked_results,
        start=1
    ):

        is_gold = any(
            (
                passage["doc_id"],
                sentence_id
            )
            in gold_evidence
            for sentence_id in passage["sentence_ids"]
        )

        print()
        print(f"RANG : {rank}")
        print(f"TITRE : {passage['title']}")
        print(f"DOC ID : {passage['doc_id']}")
        print(f"SCORE FAISS : {passage['score']}")
        print(f"SCORE RERANK : {passage['rerank_score']}")
        print(f"SENTENCE IDS : {passage['sentence_ids']}")
        print(f"PREUVE GOLD : {is_gold}")
        print(f"TEXTE : {passage['text'][:1000]}")


def find_gold_in_results(
    results: list[dict],
    gold_evidence: set[tuple[str, int]]
):

    matches = []

    for rank, result in enumerate(
        results,
        start=1
    ):

        for sentence_id in result["sentence_ids"]:

            if (
                result["doc_id"],
                sentence_id
            ) in gold_evidence:

                matches.append(
                    {
                        "rank": rank,
                        "doc_id": result["doc_id"],
                        "title": result["title"],
                        "sentence_id": sentence_id,
                        "score": result["score"]
                    }
                )

    return matches

def inspect_gold_chunk(
    doc_id: str,
    sentence_id: int,
    document_retriever: DocumentRetriever
):

    print()
    print("=" * 100)
    print("INSPECTION DIRECTE DU CHUNK GOLD")
    print("=" * 100)

    connection = (
        document_retriever.connection
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            chunk_id,
            doc_id,
            title,
            sentence_ids,
            text
        FROM chunks
        WHERE doc_id = ?
        """,
        (doc_id,)
    )

    rows = cursor.fetchall()

    if not rows:

        print(
            "❌ Aucun chunk trouvé pour ce document."
        )

        return

    print(
        f"Nombre de chunks trouvés : {len(rows)}"
    )

    for row in rows:

        print()
        print(
            f"CHUNK ID : {row[0]}"
        )

        print(
            f"DOC ID : {row[1]}"
        )

        print(
            f"TITRE : {row[2]}"
        )

        print(
            f"SENTENCE IDS : {row[3]}"
        )

        print(
            f"TEXTE : {row[4][:1000]}"
        )

    cursor.close()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    examples = load_examples(
        DEV_FILE
    )

    print("Recherche d'un exemple couvert par le corpus...")

    print()

    # =========================================================================
    # CHARGEMENT DU DOCUMENT RETRIEVER
    # =========================================================================

    print(
        "Chargement du DocumentRetriever..."
    )

    document_retriever = DocumentRetriever()

    inspect_gold_chunk(
        doc_id="Andrew_Kevin_Walker",
        sentence_id=0,
        document_retriever=document_retriever
        )

    passage_retriever = PassageRetriever(
        document_retriever
    )

    # =========================================================================
    # RECHERCHE D'UN EXEMPLE COUVERT PAR LE CORPUS
    # =========================================================================

    example = None

    for candidate in examples:

        gold_evidence = extract_gold_evidence(
            candidate
        )

        # On ignore les exemples NOT ENOUGH INFO
        if not gold_evidence:

            continue

        # Vérifie si au moins un document gold
        # existe dans la base SQLite
        coverage = check_gold_coverage(
            candidate,
            document_retriever.connection
        )

        has_coverage = any(
            item["exists"]
            for item in coverage
        )

        if has_coverage:

            example = candidate

            break

    if example is None:

        raise RuntimeError(
            "Aucun exemple avec une preuve présente "
            "dans le corpus indexé."
        )

    # =========================================================================
    # AFFICHAGE DE L'EXEMPLE SÉLECTIONNÉ
    # =========================================================================

    print()

    print(
        "Exemple couvert trouvé :"
    )

    print(
        example["claim"]
    )

    print()

    # =========================================================================
    # CHARGEMENT DU RERANKER
    # =========================================================================

    print(
        "Chargement du Reranker..."
    )

    reranker = Reranker()

    # =========================================================================
    # DEBUG COMPLET
    # =========================================================================

    debug_example(
        example,
        passage_retriever,
        reranker
    )