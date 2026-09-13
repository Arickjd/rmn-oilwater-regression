"""Isolamento de grupos, cobertura OOF e bootstrap de unidades independentes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from models.cross_validation import iter_cv_partitions, run_cross_validation
from models.evaluation import bootstrap_test_metrics
from preprocess.groups import load_groups
from preprocess.split import prepare_datasets, prepare_partitions
from test_pipeline import sample_frame


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.frame = sample_frame(60)
        self.groups = pd.Series(np.repeat(np.arange(20), 3).astype(str), index=self.frame.index)

    def test_groups_never_cross_partitions_and_holdout_is_excluded(self):
        data = prepare_datasets(self.frame, groups=self.groups, test_group_fraction=0.3,
                                augmented_size=0)
        parts = [data.train_original, data.validation, data.test]
        group_sets = [set(self.groups.loc[part.index]) for part in parts]
        self.assertFalse(group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2]
                         or group_sets[1] & group_sets[2])
        development = self.frame.drop(index=data.test.index)
        partitions = list(iter_cv_partitions(development, folds=3, repeats=2, groups=self.groups))
        coverage = {1: [], 2: []}
        for repeat, fold, seed, train, val, assessment in partitions:
            ids = [set(self.groups.loc[index]) for index in (train, val, assessment)]
            self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
            self.assertFalse((set(train) | set(val) | set(assessment)) & set(data.test.index))
            coverage[repeat].extend(assessment)
        for ids in coverage.values():
            self.assertEqual(sorted(ids), sorted(development.index))
        again = list(iter_cv_partitions(development, folds=3, repeats=2, groups=self.groups))
        for first, second in zip(partitions, again):
            for a, b in zip(first[3:], second[3:]):
                np.testing.assert_array_equal(a, b)

    def test_explicit_partitions_reject_group_leakage(self):
        with self.assertRaisesRegex(ValueError, "grupo"):
            prepare_partitions(self.frame.iloc[:20], self.frame.iloc[20:40], self.frame.iloc[40:],
                               groups=self.groups)

    def test_mapping_requires_exact_sample_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.csv"
            mapping = pd.DataFrame({"sample_id": self.frame.index, "group": self.groups.values})
            mapping.to_csv(path, index=False)
            pd.testing.assert_series_equal(load_groups(path, self.frame), self.groups.rename("group"),
                                           check_names=False)
            mapping.iloc[:-1].to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_groups(path, self.frame)

    def test_cluster_bootstrap_preserves_pairs_and_whole_groups(self):
        # Grupos de tamanhos desiguais: reproduz manualmente o sorteio de grupos.
        y = np.array([0.1, 0.2, 0.5, 0.7, 0.9])
        pred = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        groups = np.array(["a", "a", "b", "b", "b"])
        intervals, differences = bootstrap_test_metrics(y, {"one": pred, "same": pred},
                                                        groups=groups, n_resamples=100, seed=7)
        manual = []
        rng = np.random.default_rng(7)
        members = [np.array([0, 1]), np.array([2, 3, 4])]
        for _ in range(100):
            idx = np.concatenate([members[i] for i in rng.integers(0, 2, 2)])
            manual.append(np.sqrt(np.mean((y[idx] - pred[idx]) ** 2)))
        row = intervals[(intervals.model == "one") & (intervals.metric == "RMSE")].iloc[0]
        np.testing.assert_allclose([row.ci_low, row.ci_high], np.quantile(manual, [0.025, 0.975]))
        self.assertEqual(row.n_independent_units, 2)
        self.assertTrue(differences[["difference_a_minus_b", "ci_low", "ci_high"]].eq(0).all().all())

    def test_degenerate_r2_is_reported_not_fabricated(self):
        intervals, _ = bootstrap_test_metrics([0.1, 0.1, 0.1], {"constant": [0.1, 0.1, 0.1]},
                                              n_resamples=100)
        r2 = intervals[intervals.metric == "R2"].iloc[0]
        self.assertTrue(np.isnan(r2.estimate) and np.isnan(r2.ci_low))
        self.assertEqual(r2.valid_resamples, 0)
        with self.assertRaises(ValueError):
            bootstrap_test_metrics([1, 2], {"m": [1, 2]}, groups=["a", "a"], n_resamples=100)

    def test_oof_runner_records_one_prediction_per_repeat_and_model(self):
        # Stub apenas do treinamento; divisão, PCA, predição e métricas são reais.
        def no_training(model, *args, **kwargs):
            return pd.DataFrame({"epoch": [1]}), 1

        with tempfile.TemporaryDirectory() as directory, patch(
                "models.cross_validation.train_model", side_effect=no_training):
            result = run_cross_validation(self.frame, groups=self.groups, output_dir=directory,
                                           folds=2, repeats=2, augmented_size=0,
                                           mlp_epochs=1, pinn_epochs=1)
            counts = result.predictions.groupby(["repeat", "model", "sample_id"]).size()
            self.assertTrue(counts.eq(1).all())
            self.assertEqual(len(counts), 2 * 5 * len(self.frame))
            self.assertEqual(len(result.repeat_metrics), 2 * 5)
            self.assertIn(result.selected_model, result.predictions.model.unique())
            self.assertTrue((Path(directory) / "selection.json").is_file())


if __name__ == "__main__":
    unittest.main()
