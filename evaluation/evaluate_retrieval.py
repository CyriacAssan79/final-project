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
# CHARGEMENT DES DONNÉES FEVER
# =============================================================================

def load_fever_data(file_path: Path) -> list[dict]:
    """
    Charge les exemples FEVER depuis un fichier JSONL.
    """

    examples = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            examples.append(json.loads(line))

    return examples


# =============================================================================
# EXTRACTION DES PREUVES DE RÉFÉRENCE
# =============================================================================

def extract_gold_evidence(example: dict) -> set[tuple[str, int]]:
    """
    Extrait les couples :

        (document_id, sentence_id)

    correspondant aux preuves de référence FEVER.
    """

    gold_evidence = set()

    for evidence_set in example["evidence"]:

        for evidence in evidence_set:

            # Format FEVER :
            #
            # [annotator_id, line_id, doc_id, sentence_id]
            #
            # Exemple :
            # [92206, 104971, "Nikolaj_Coster-Waldau", 7]

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


# =============================================================================
# MATCHING DES PREUVES
# =============================================================================

def evidence_matches(
    result: dict,
    gold_evidence: set[tuple[str, int]]
) -> bool:
    """
    Vérifie si le chunk récupéré contient au moins
    une phrase appartenant aux preuves de référence.
    """

    doc_id = result["doc_id"]

    sentence_ids = result["sentence_ids"]

    for sentence_id in sentence_ids:

        if (doc_id, sentence_id) in gold_evidence:

            return True

    return False


# =============================================================================
# ÉVALUATION
# =============================================================================

def evaluate_retrieval(
    examples: list[dict],
    top_k: int = 50,
    rerank_top_k: int = 5
):

    document_retriever = DocumentRetriever()

    passage_retriever = PassageRetriever(
        document_retriever
    )

    reranker = Reranker()

    total_examples = 0

    examples_with_evidence = 0

    retrieved_evidence_count = 0

    total_gold_evidence = 0

    reciprocal_ranks = []

    for index, example in enumerate(examples):

        claim = example["claim"]

        gold_evidence = extract_gold_evidence(
            example
        )

        # Les exemples NOT ENOUGH INFO ne possèdent
        # pas de preuve réelle à récupérer.
        if not gold_evidence:

            continue

        total_examples += 1

        total_gold_evidence += len(
            gold_evidence
        )

        # ---------------------------------------------------------------------
        # RETRIEVAL
        # ---------------------------------------------------------------------

        passages = passage_retriever.retrieve(
            query=claim,
            top_k=top_k
        )

        # ---------------------------------------------------------------------
        # RERANKING
        # ---------------------------------------------------------------------

        reranked_results = reranker.rerank(
            query=claim,
            passages=passages,
            top_k=rerank_top_k
        )

        found_positions = []

        for rank, result in enumerate(
            reranked_results,
            start=1
        ):

            if evidence_matches(
                result,
                gold_evidence
            ):

                retrieved_evidence_count += 1

                found_positions.append(rank)

        # ---------------------------------------------------------------------
        # RECALL PAR CLAIM
        # ---------------------------------------------------------------------

        if found_positions:

            examples_with_evidence += 1

            reciprocal_ranks.append(
                1 / min(found_positions)
            )

        else:

            reciprocal_ranks.append(0)

        # ---------------------------------------------------------------------
        # PROGRESSION
        # ---------------------------------------------------------------------

        if (index + 1) % 100 == 0:

            print(
                f"Progression : {index + 1}/{len(examples)}"
            )

    # =========================================================================
    # MÉTRIQUES
    # =========================================================================

    evidence_recall = (
        examples_with_evidence
        / total_examples
        if total_examples > 0
        else 0
    )

    mean_reciprocal_rank = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
        if reciprocal_ranks
        else 0
    )

    print()
    print("=" * 80)
    print("ÉVALUATION DU RETRIEVAL")
    print("=" * 80)

    print(
        f"Claims évalués : {total_examples}"
    )

    print(
        f"Claims avec preuve récupérée : "
        f"{examples_with_evidence}"
    )

    print(
        f"Evidence Recall : "
        f"{evidence_recall:.4f}"
    )

    print(
        f"MRR : "
        f"{mean_reciprocal_rank:.4f}"
    )

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    examples = load_fever_data(
        DEV_FILE
    )

    # Pour le premier test :
    # on utilise seulement 100 exemples.
    evaluate_retrieval(
        examples[:100],
        top_k=50,
        rerank_top_k=5
    )