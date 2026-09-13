"""Cobertura OOF e bootstrap de amostras independentes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from models.cross_validation import iter_cv_partitions, run_cross_validation
from models.evaluation import bootstrap_test_metrics
from test_pipeline import sample_frame


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.frame = sample_frame(30)

    def test_cv_partitions_are_disjoint_deterministic_and_cover_development(self):
        partitions = list(iter_cv_partitions(self.frame, folds=3, repeats=2))
        coverage = {1: [], 2: []}
        for repeat, fold, seed, train, val, assessment in partitions:
            self.assertFalse(set(train) & set(val))
            self.assertFalse(set(train) & set(assessment))
            self.assertFalse(set(val) & set(assessment))
            coverage[repeat].extend(assessment)
        for ids in coverage.values():
            self.assertEqual(sorted(ids), sorted(self.frame.index))
        again = list(iter_cv_partitions(self.frame, folds=3, repeats=2))
        for first, second in zip(partitions, again):
            self.assertEqual(first[:3], second[:3])
            for first_ids, second_ids in zip(first[3:], second[3:]):
                np.testing.assert_array_equal(first_ids, second_ids)

    def test_row_bootstrap_preserves_pairs_and_matches_manual_percentiles(self):
        y = np.array([0.1, 0.2, 0.5, 0.7, 0.9])
        pred = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        intervals, differences = bootstrap_test_metrics(y, {"one": pred, "same": pred},
                                                        n_resamples=100, seed=7)
        manual = []
        rng = np.random.default_rng(7)
        indices = rng.integers(0, len(y), size=(100, len(y)))
        manual = np.sqrt(np.mean((y[indices] - pred[indices]) ** 2, axis=1))
        row = intervals[(intervals.model == "one") & (intervals.metric == "RMSE")].iloc[0]
        np.testing.assert_allclose([row.ci_low, row.ci_high], np.quantile(manual, [0.025, 0.975]))
        self.assertEqual(row.resampling_unit, "sample")
        self.assertEqual(row.n_independent_units, len(y))
        self.assertTrue(differences[["difference_a_minus_b", "ci_low", "ci_high"]].eq(0).all().all())

    def test_degenerate_r2_is_reported_not_fabricated(self):
        intervals, _ = bootstrap_test_metrics([0.1, 0.1, 0.1], {"constant": [0.1, 0.1, 0.1]},
                                              n_resamples=100)
        r2 = intervals[intervals.metric == "R2"].iloc[0]
        self.assertTrue(np.isnan(r2.estimate) and np.isnan(r2.ci_low))
        self.assertEqual(r2.valid_resamples, 0)

    def test_oof_runner_records_one_prediction_per_repeat_and_model(self):
        # Stub apenas do treinamento; divisão, PCA, predição e métricas são reais.
        def no_training(model, *args, **kwargs):
            return pd.DataFrame({"epoch": [1]}), 1

        with tempfile.TemporaryDirectory() as directory, patch(
                "models.cross_validation.train_model", side_effect=no_training):
            result = run_cross_validation(self.frame, output_dir=directory, folds=2, repeats=2, augmented_size=0,
                                           mlp_epochs=1, pinn_epochs=1)
            counts = result.predictions.groupby(["repeat", "model", "sample_id"]).size()
            self.assertTrue(counts.eq(1).all())
            self.assertEqual(len(counts), 2 * 5 * len(self.frame))
            self.assertEqual(len(result.repeat_metrics), 2 * 5)
            self.assertIn(result.selected_model, result.predictions.model.unique())
            self.assertTrue((Path(directory) / "selection.json").is_file())


if __name__ == "__main__":
    unittest.main()
