import unittest

from evaluation.metrics import (
    accuracy_score,
    evidence_precision,
    evidence_recall,
    expected_calibration_error,
    fever_score,
    mean_reciprocal_rank,
)


class MetricsTest(unittest.TestCase):
    def test_label_and_evidence_metrics(self):
        gold = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        predicted = ["SUPPORTS", "SUPPORTS", "NOT ENOUGH INFO"]
        retrieved = [
            {("Doc_A", 0)},
            {("Doc_B", 2)},
            set(),
        ]
        evidence = [
            {("Doc_A", 0), ("Doc_A", 1)},
            {("Doc_C", 3)},
            set(),
        ]

        self.assertAlmostEqual(accuracy_score(gold, predicted), 2 / 3)
        self.assertAlmostEqual(evidence_recall(retrieved, evidence), 0.5)
        self.assertAlmostEqual(evidence_precision(retrieved, evidence), 0.5)
        self.assertAlmostEqual(fever_score(gold, predicted, [True, False, False]), 1 / 3)

    def test_rank_and_calibration_metrics(self):
        self.assertAlmostEqual(mean_reciprocal_rank([1, 2, None]), 0.5)
        self.assertAlmostEqual(
            expected_calibration_error([0.9, 0.6], [True, False], n_bins=2),
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
