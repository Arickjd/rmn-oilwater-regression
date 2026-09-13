"""Testes numéricos das etapas sensíveis da migração dos notebooks."""

import unittest

import numpy as np
import pandas as pd
import torch

from models.data_augmentation import augment_training_data
from models.mlp import MLP
from models.pinn import PINN
from models.pinn_log_t2 import PINNLogT2
from models.pinn_weighted import PINNWeighted
from preprocess import echo_columns, prepare_datasets, preprocess_data
from preprocess.signal import dwt_denoise, normalize_rows, truncate_and_bin


def sample_frame(n=20, echoes=8):
    rng = np.random.default_rng(7)
    data = pd.DataFrame(rng.uniform(size=(n, echoes)),
                        columns=[f"Echo_{i + 1}" for i in range(echoes)])
    data["PHIX"] = rng.uniform(0.1, 0.8, n)
    data["Swirr_PHIX"] = rng.uniform(0.1, 0.9, n)
    data["MBVI"] = data["PHIX"] * data["Swirr_PHIX"]
    data["MPHI"] = data["PHIX"] * 0.9
    return data


class PipelineTests(unittest.TestCase):
    def test_signal_operations_and_odd_length(self):
        zero, _ = dwt_denoise(np.zeros(129), level=2)
        self.assertEqual(zero.shape, (129,))
        np.testing.assert_array_equal(zero, 0)
        np.testing.assert_array_equal(normalize_rows([[2, 2], [2, 4]]), [[0, 0], [0, 1]])
        truncated, binned = truncate_and_bin(np.arange(12).reshape(1, 12), trim_last=4, bin_size=4)
        np.testing.assert_array_equal(truncated, np.arange(8).reshape(1, 8))
        np.testing.assert_array_equal(binned, [[1.5, 5.5]])
        with self.assertRaises(ValueError):
            truncate_and_bin(np.ones((2, 9)), trim_last=0, bin_size=4)

    def test_preprocessing_preserves_properties_and_index(self):
        frame = sample_frame(3, 3000)
        frame.index = [10, 20, 30]
        result = preprocess_data(frame)
        self.assertEqual(result.frame.shape, (3, 504))
        for column in ("Swirr_PHIX", "MBVI", "MPHI", "PHIX"):
            pd.testing.assert_series_equal(result.frame[column], frame[column])
        self.assertTrue(np.isfinite(result.frame.to_numpy()).all())
        self.assertEqual(result.frame.columns[499], "Echo_Bin_500")

    def test_mixup_preserves_physical_ratio_and_does_not_mutate(self):
        frame = sample_frame(2)
        original = frame.copy(deep=True)
        augmented = augment_training_data(frame, total_samples=30, seed=5)
        self.assertEqual(len(augmented), 30)
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_frame_equal(augmented, augment_training_data(frame, 30, 5))
        np.testing.assert_allclose(augmented["Swirr_PHIX"], augmented["MBVI"] / augmented["PHIX"])
        # Com dois pais, o peso obtido do eco também deve reconstruir PHIX.
        weight = (augmented["Echo_1"] - frame.iloc[1]["Echo_1"]) / (frame.iloc[0]["Echo_1"] - frame.iloc[1]["Echo_1"])
        np.testing.assert_allclose(augmented["PHIX"], weight * frame.iloc[0]["PHIX"] + (1 - weight) * frame.iloc[1]["PHIX"])
        self.assertTrue(weight.between(-1e-12, 1 + 1e-12).all())

    def test_split_and_pca_exclude_validation_and_test(self):
        frame = sample_frame()
        data = prepare_datasets(frame, test_size=4, validation_size=0.25, augmented_size=30)
        train_ids, val_ids, test_ids = map(set, [data.train_original.index, data.validation.index, data.test.index])
        self.assertFalse(train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)
        self.assertEqual(train_ids | val_ids | test_ids, set(frame.index))
        self.assertEqual((len(train_ids), len(val_ids), len(test_ids)), (12, 4, 4))
        columns = echo_columns(frame)
        np.testing.assert_allclose(data.pca.mean_, data.train[columns].mean().to_numpy())
        # Alterar apenas dados reservados não pode alterar treino ampliado/PCA.
        changed = frame.copy()
        changed.loc[list(val_ids | test_ids), columns] += 100
        other = prepare_datasets(changed, test_size=4, validation_size=0.25, augmented_size=30)
        pd.testing.assert_frame_equal(data.train, other.train)
        np.testing.assert_allclose(data.pca.components_, other.pca.components_)

    def test_pinn_physics_and_single_sample_gradients(self):
        for model_class in (PINN, PINNLogT2, PINNWeighted):
            with self.subTest(model=model_class.__name__):
                model = model_class(8)
                for param in model.parameters():
                    torch.nn.init.zeros_(param)
                x, y = torch.ones(1, 8), torch.tensor([0.3])
                saturation, reconstructed = model.decode(x)
                torch.testing.assert_close(saturation, torch.tensor([0.5]))
                time = torch.arange(8, dtype=torch.float32)
                if model_class is PINN:
                    expected = torch.exp(-time / np.log(2))
                elif model_class is PINNLogT2:
                    expected = torch.exp(-time / 8)
                else:
                    expected = np.log(2) * (0.5 * torch.exp(-time / 8) + 0.5 * torch.exp(-time / 16))
                torch.testing.assert_close(reconstructed[0], expected)
                output, physics = model.loss_components(x, y)
                torch.testing.assert_close(output, torch.tensor(0.04))
                expected_physics = ((1 - expected).square() * torch.exp(-model.time_decay * time / 8)).mean()
                torch.testing.assert_close(physics, expected_physics)
                (model.output_weight * output + model.physics_weight * physics).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        mlp = MLP(3)
        self.assertEqual(mlp(torch.ones(1, 3)).shape, (1,))


if __name__ == "__main__":
    unittest.main()
