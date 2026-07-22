import json
from pathlib import Path

from retrieval.document_retriever import DocumentRetriever
from retrieval.passage_retriever import PassageRetriever
from retrieval.reranker import Reranker
from verification.verifier import Verifier
from evaluation.metrics import expected_calibration_error


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
# CHARGEMENT DES EXEMPLES FEVER
# =============================================================================

def load_examples(
    file_path: Path
) -> list[dict]:

    examples = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():

                continue

            examples.append(
                json.loads(line)
            )

    return examples


# =============================================================================
# EXTRACTION DES PREUVES GOLD
# =============================================================================

def extract_gold_evidence(
    example: dict
) -> set[tuple[str, int]]:

    gold_evidence = set()

    for evidence_set in example["evidence"]:

        for evidence in evidence_set:

            if len(evidence) < 4:

                continue

            doc_id = evidence[2]

            sentence_id = evidence[3]

            if (
                doc_id is None
                or sentence_id is None
            ):

                continue

            gold_evidence.add(
                (
                    doc_id,
                    sentence_id
                )
            )

    return gold_evidence


# =============================================================================
# VÉRIFICATION DE LA PRÉSENCE DES PREUVES
# =============================================================================

def check_gold_coverage(
    example: dict,
    connection
) -> list[dict]:

    gold_evidence = extract_gold_evidence(
        example
    )

    coverage = []

    for doc_id, sentence_id in gold_evidence:

        cursor = connection.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE doc_id = ?
            AND EXISTS (
                SELECT 1
                FROM json_each(chunks.sentence_ids)
                WHERE json_each.value = ?
            )
            """,
            (
                doc_id,
                sentence_id
            )
        )

        num_chunks = cursor.fetchone()[0]

        coverage.append(
            {
                "doc_id": doc_id,
                "sentence_id": sentence_id,
                "exists": num_chunks > 0,
                "num_chunks": num_chunks
            }
        )

    return coverage


# =============================================================================
# VÉRIFICATION D'UNE PREUVE DANS LES RÉSULTATS
# =============================================================================

def evidence_matches(
    passage: dict,
    gold_evidence: set[tuple[str, int]]
) -> bool:

    doc_id = passage["doc_id"]

    for sentence_id in passage["sentence_ids"]:

        if (
            doc_id,
            sentence_id
        ) in gold_evidence:

            return True

    return False

def load_indexed_documents(
    connection
) -> set[str]:

    cursor = connection.execute(
        """
        SELECT DISTINCT doc_id
        FROM chunks
        """
    )

    return {
        row[0]
        for row in cursor.fetchall()
    }


# =============================================================================
# ÉVALUATION DU PIPELINE
# =============================================================================

def evaluate_pipeline(
    examples: list[dict],
    document_retriever: DocumentRetriever,
    passage_retriever: PassageRetriever,
    reranker: Reranker,
    verifier: Verifier,
    indexed_documents: set[str],
    max_examples: int = 100,
    retrieval_top_k: int = 50,
    rerank_top_k: int = 5
):

    # -------------------------------------------------------------------------
    # COMPTEURS
    # -------------------------------------------------------------------------

    evaluated_examples = 0

    correct_labels = 0

    claims_with_evidence = 0

    fever_score_count = 0

    reciprocal_ranks = []

    skipped_examples = 0

    detailed_results = []
    confidences = []
    correctness_values = []

    # -------------------------------------------------------------------------
    # ÉVALUATION
    # -------------------------------------------------------------------------

    for index, example in enumerate(examples):

        if evaluated_examples >= max_examples:

            break

        gold_label = example["label"]

        gold_evidence = extract_gold_evidence(
            example
        )

        # ---------------------------------------------------------------------
        # NOT ENOUGH INFO
        # ---------------------------------------------------------------------

        if not gold_evidence:

            skipped_examples += 1

            continue

        # ---------------------------------------------------------------------
        # VÉRIFICATION DE LA COUVERTURE DU CORPUS
        # ---------------------------------------------------------------------

        has_coverage = any(doc_id in indexed_documents for doc_id, sentence_id in gold_evidence)

        if not has_coverage:
            skipped_examples += 1
            continue

        evaluated_examples += 1

        claim = example["claim"]

        # ---------------------------------------------------------------------
        # RETRIEVAL
        # ---------------------------------------------------------------------

        passages = passage_retriever.retrieve(
            query=claim,
            top_k=retrieval_top_k
        )

        # ---------------------------------------------------------------------
        # RERANKING
        # ---------------------------------------------------------------------

        reranked_results = reranker.rerank(
            query=claim,
            passages=passages,
            top_k=rerank_top_k
        )

        # ---------------------------------------------------------------------
        # RECALL DE LA PREUVE
        # ---------------------------------------------------------------------

        evidence_ranks = []

        for rank, passage in enumerate(
            reranked_results,
            start=1
        ):

            if evidence_matches(
                passage,
                gold_evidence
            ):

                evidence_ranks.append(
                    rank
                )

        if evidence_ranks:

            claims_with_evidence += 1

            first_rank = min(
                evidence_ranks
            )

            reciprocal_ranks.append(
                1 / first_rank
            )

        else:

            reciprocal_ranks.append(
                0
            )

        # ---------------------------------------------------------------------
        # VÉRIFICATION NLI
        # ---------------------------------------------------------------------

        verification_result = verifier.verify(
            claim=claim,
            evidence=reranked_results
        )

        predicted_label = (
            verification_result["label"]
        )

        # =========================================================================
        # SAUVEGARDE DU RÉSULTAT DÉTAILLÉ
        # =========================================================================

        first_evidence_rank = None

        if evidence_ranks:

            first_evidence_rank = min(
                evidence_ranks
            )
        best_evidence = None

        if verification_result["evidence"]:

            best_evidence = max(
                verification_result["evidence"],
                key=lambda item: item["confidence"]
            )

        detailed_results.append(
            {
                "claim": claim,
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "label_correct": (
                    predicted_label == gold_label
                ),
                "evidence_found": bool(
                    evidence_ranks
                ),
                "first_evidence_rank": (
                    first_evidence_rank
                ),
                "best_document": (
                    best_evidence["doc_id"]
                    if best_evidence
                    else None
                ),
                "nli_label": (
                    best_evidence["nli_label"]
                    if best_evidence
                    else None
                ),
                "confidence": (
                    best_evidence["confidence"]
                    if best_evidence
                    else 0.0
                )
            }
        )

        # ---------------------------------------------------------------------
        # ACCURACY
        # ---------------------------------------------------------------------

        label_correct = (
            predicted_label == gold_label
        )

        if label_correct:

            correct_labels += 1

        confidences.append(float(verification_result["confidence"]))
        correctness_values.append(label_correct)

        # ---------------------------------------------------------------------
        # FEVER SCORE
        # ---------------------------------------------------------------------

        if (
            label_correct
            and evidence_ranks
        ):

            fever_score_count += 1

        # ---------------------------------------------------------------------
        # PROGRESSION
        # ---------------------------------------------------------------------

        if evaluated_examples % 10 == 0:

            print(
                f"Progression : "
                f"{evaluated_examples}/{max_examples}"
            )

    # =========================================================================
    # CALCUL DES MÉTRIQUES
    # =========================================================================

    accuracy = (

        correct_labels
        / evaluated_examples

        if evaluated_examples > 0

        else 0.0
    )

    evidence_recall = (

        claims_with_evidence
        / evaluated_examples

        if evaluated_examples > 0

        else 0.0
    )

    mrr = (

        sum(reciprocal_ranks)
        / len(reciprocal_ranks)

        if reciprocal_ranks

        else 0.0
    )

    fever_score = (

        fever_score_count
        / evaluated_examples

        if evaluated_examples > 0

        else 0.0
    )

    ece = expected_calibration_error(
        confidences=confidences,
        correctness=correctness_values,
        n_bins=10,
    )

    return {
        "evaluated_examples": evaluated_examples,
        "skipped_examples": skipped_examples,
        "correct_labels": correct_labels,
        "accuracy": accuracy,
        "claims_with_evidence": claims_with_evidence,
        "evidence_recall": evidence_recall,
        "mrr": mrr,
        "fever_score": fever_score,
        "ece": ece,
        "detailed_results": detailed_results
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("Chargement des exemples FEVER...")
    examples = load_examples(DEV_FILE)
    print(f"Exemples chargés : {len(examples)}")
    print()

    # =========================================================================
    # CHARGEMENT DU RETRIEVER
    # =========================================================================

    print("Chargement du DocumentRetriever...")
    document_retriever = DocumentRetriever()
    print(
    "Chargement des documents indexés...")

    indexed_documents = load_indexed_documents(document_retriever.connection)

    print(f"Documents indexés : {len(indexed_documents)}")

    passage_retriever = PassageRetriever(document_retriever)

    # =========================================================================
    # CHARGEMENT DU RERANKER
    # =========================================================================

    print(
        "Chargement du Reranker..."
    )

    reranker = Reranker()

    # =========================================================================
    # CHARGEMENT DU VERIFIER
    # =========================================================================

    print("Chargement du Verifier...")

    verifier = Verifier()

    print()

    print(
        "Début de l'évaluation..."
    )

    # =========================================================================
    # ÉVALUATION
    # =========================================================================

    results = evaluate_pipeline(
        examples=examples,
        document_retriever=document_retriever,
        passage_retriever=passage_retriever,
        reranker=reranker,
        verifier=verifier,
        indexed_documents=indexed_documents,
        max_examples=50,
        retrieval_top_k=50,
        rerank_top_k=5
    )

    # =========================================================================
    # AFFICHAGE DES RÉSULTATS
    # =========================================================================

    print()

    print("=" * 100)
    print("RÉSULTATS DE L'ÉVALUATION")
    print("=" * 100)
    print()
    print(f"Claims évalués : {results['evaluated_examples']}")

    print(f"Claims ignorés : {results['skipped_examples']}")
    print()
    print(f"Labels corrects : {results['correct_labels']}")

    print(f"Accuracy : {results['accuracy']:.4f}")
    print()
    print(f"Claims avec preuve :{results['claims_with_evidence']}")

    print(f"Evidence Recall : {results['evidence_recall']:.4f}")

    print(f"MRR :{results['mrr']:.4f}")
    print()
    print(f"FEVER Score : {results['fever_score']:.4f}")
    print(f"ECE : {results['ece']:.4f}")
    print()
    print("=" * 100)
    print()

    print("=" * 100)
    print("EXEMPLES MAL CLASSÉS")
    print("=" * 100)

    for result in results["detailed_results"]:
        if not result["label_correct"]:
            print()
            print(f"CLAIM : {result['claim']}")
            print(f"LABEL RÉEL : {result['gold_label']}")
            print(f"LABEL PRÉDIT : {result['predicted_label']}")
            print(f"PREUVE TROUVÉE : {result['evidence_found']}")
            print(f"RANG PREUVE GOLD : {result['first_evidence_rank']}")
            print(f"DOCUMENT SÉLECTIONNÉ : {result['best_document']}")
            print(f"LABEL NLI : {result['nli_label']}")
            print(f"CONFIANCE : {result['confidence']:.4f}")
