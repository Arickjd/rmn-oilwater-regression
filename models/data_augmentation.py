"""MixUp dos ecos e das propriedades físicas, exclusivamente no treino."""

import numpy as np
import pandas as pd

from preprocess.data import TARGET, validate_data


def augment_training_data(frame, total_samples=1000, seed=42):
    """Interpola duas amostras distintas e recalcula Swirr = MBVI / PHIX.

    total_samples é o tamanho final desejado; zero desativa a ampliação.
    Nunca reduz o conjunto original se o tamanho solicitado for menor.
    """
    validate_data(frame)
    if total_samples < 0:
        raise ValueError("total_samples não pode ser negativo.")
    count = max(0, total_samples - len(frame))
    if count and len(frame) < 2:
        raise ValueError("MixUp requer pelo menos duas amostras reais de treino.")
    if not count:
        return frame.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(frame), count)
    second = rng.integers(0, len(frame) - 1, count)
    second += second >= first
    weights = rng.random((count, 1))
    values = frame.to_numpy(dtype=float)
    synthetic = pd.DataFrame(
        weights * values[first] + (1 - weights) * values[second], columns=frame.columns
    )
    synthetic[TARGET] = synthetic["MBVI"] / synthetic["PHIX"]
    augmented = pd.concat([frame, synthetic], ignore_index=True)
    return augmented.sample(frac=1, random_state=seed).reset_index(drop=True)

