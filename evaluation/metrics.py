from collections.abc import Iterable


FEVER_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}


def accuracy_score(gold_labels: Iterable[str], predicted_labels: Iterable[str]) -> float:
    gold = list(gold_labels)
    predicted = list(predicted_labels)
    if len(gold) != len(predicted):
        raise ValueError("gold_labels and predicted_labels must have the same length.")
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, predicted)) / len(gold)


def evidence_recall(
    retrieved_evidence: Iterable[set[tuple[str, int]]],
    gold_evidence: Iterable[set[tuple[str, int]]],
) -> float:
    retrieved_sets = list(retrieved_evidence)
    gold_sets = list(gold_evidence)
    if len(retrieved_sets) != len(gold_sets):
        raise ValueError("retrieved_evidence and gold_evidence must have the same length.")

    eligible = [index for index, gold in enumerate(gold_sets) if gold]
    if not eligible:
        return 0.0

    found = 0
    for index in eligible:
        if retrieved_sets[index] & gold_sets[index]:
            found += 1
    return found / len(eligible)


def evidence_precision(
    retrieved_evidence: Iterable[set[tuple[str, int]]],
    gold_evidence: Iterable[set[tuple[str, int]]],
) -> float:
    retrieved_sets = list(retrieved_evidence)
    gold_sets = list(gold_evidence)
    if len(retrieved_sets) != len(gold_sets):
        raise ValueError("retrieved_evidence and gold_evidence must have the same length.")

    retrieved_count = 0
    correct_count = 0
    for retrieved, gold in zip(retrieved_sets, gold_sets):
        retrieved_count += len(retrieved)
        correct_count += len(retrieved & gold)

    if retrieved_count == 0:
        return 0.0
    return correct_count / retrieved_count


def mean_reciprocal_rank(ranks: Iterable[int | None]) -> float:
    values = [0.0 if rank is None or rank <= 0 else 1 / rank for rank in ranks]
    if not values:
        return 0.0
    return sum(values) / len(values)


def fever_score(
    gold_labels: Iterable[str],
    predicted_labels: Iterable[str],
    evidence_found: Iterable[bool],
) -> float:
    gold = list(gold_labels)
    predicted = list(predicted_labels)
    found = list(evidence_found)
    if not (len(gold) == len(predicted) == len(found)):
        raise ValueError("gold_labels, predicted_labels and evidence_found must align.")
    if not gold:
        return 0.0

    correct = 0
    for gold_label, predicted_label, has_evidence in zip(gold, predicted, found):
        if predicted_label == gold_label and has_evidence:
            correct += 1
    return correct / len(gold)


def expected_calibration_error(
    confidences: Iterable[float],
    correctness: Iterable[bool],
    n_bins: int = 10,
) -> float:
    if n_bins <= 0:
        raise ValueError("n_bins must be greater than zero.")

    confidence_values = list(confidences)
    correctness_values = list(correctness)
    if len(confidence_values) != len(correctness_values):
        raise ValueError("confidences and correctness must have the same length.")
    if not confidence_values:
        return 0.0

    for confidence in confidence_values:
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("All confidences must be between 0 and 1.")

    total = len(confidence_values)
    ece = 0.0

    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins

        if bin_index == n_bins - 1:
            members = [
                index
                for index, confidence in enumerate(confidence_values)
                if lower <= confidence <= upper
            ]
        else:
            members = [
                index
                for index, confidence in enumerate(confidence_values)
                if lower <= confidence < upper
            ]

        if not members:
            continue

        bin_confidence = sum(confidence_values[index] for index in members) / len(members)
        bin_accuracy = sum(correctness_values[index] for index in members) / len(members)
        ece += (len(members) / total) * abs(bin_accuracy - bin_confidence)

    return ece
