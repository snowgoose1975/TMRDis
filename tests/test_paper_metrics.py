import unittest

from my_predictor.paper_metrics import evaluate


class PaperMetricsTests(unittest.TestCase):
    def test_equal_scores_are_grouped_for_ap_auc_and_ranking(self):
        metrics, ranking = evaluate(
            [(0.9, 1), (0.9, 0), (0.1, 1), (0.1, 0)],
            threshold=0.9,
            probabilities=False,
        )
        self.assertAlmostEqual(metrics["average_precision"], 0.5)
        self.assertAlmostEqual(metrics["auprc_trapezoid"], 0.625)
        self.assertAlmostEqual(metrics["auroc"], 0.5)
        self.assertAlmostEqual(ranking["top_1pct"]["precision"], 0.5)
        self.assertIsNone(metrics["brier_score"])
        self.assertIsNone(metrics["expected_calibration_error"])
        self.assertEqual(metrics["counts"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})

    def test_threshold_tie_chooses_lowest_distinct_score(self):
        metrics, _ = evaluate([(0.8, 1), (0.6, 0), (0.4, 1), (0.2, 0)])
        self.assertEqual(metrics["threshold"], 0.4)

    def test_negative_scores_keep_calibration_metrics_empty(self):
        metrics, _ = evaluate([(-2.0, 1), (-3.0, 0)], threshold=-2.5)
        self.assertEqual(metrics["threshold"], -2.5)
        self.assertIsNone(metrics["brier_score"])

    def test_probability_declaration_rejects_scores_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError, r"within \[0, 1\]"):
            evaluate([(-1.0, 1), (0.5, 0)], threshold=0.0, probabilities=True)


if __name__ == "__main__":
    unittest.main()
