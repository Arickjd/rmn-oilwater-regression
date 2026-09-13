"""Divisão antes do MixUp e ajuste do PCA apenas no treino."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .data import echo_columns, validate_data
from .groups import validate_groups


@dataclass
class PreparedData:
    train_original: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    pca: PCA
    train_pca: np.ndarray
    validation_pca: np.ndarray
    test_pca: np.ndarray


def prepare_datasets(frame, *, test_size=175, validation_size=0.2,
                     augmented_size=1000, pca_variance=0.999, seed=42,
                     groups=None, test_group_fraction=0.3):
    """Os índices originais permitem auditar as três partições reais.

    Com groups, os grupos são indivisíveis e os tamanhos dependem deles.
    Sem groups (apenas linhas independentes), 575 amostras dão 320/80/175.
    Somente as amostras reais de treino originam as amostras sintéticas.
    """
    validate_data(frame)
    if not frame.index.is_unique:
        raise ValueError("O índice deve identificar cada amostra de forma única.")
    if not 0 < validation_size < 1 or not 0 < pca_variance < 1:
        raise ValueError("validation_size e pca_variance devem estar entre 0 e 1.")
    if groups is None:
        train_val, test = train_test_split(frame, test_size=test_size, random_state=seed)
        train_original, validation = train_test_split(
            train_val, test_size=validation_size, random_state=seed
        )
    else:
        groups = validate_groups(frame, groups)
        if not 0 < test_group_fraction < 1:
            raise ValueError("test_group_fraction deve estar entre 0 e 1.")
        dev_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_group_fraction,
                                                  random_state=seed).split(frame, groups=groups))
        train_val, test = frame.iloc[dev_idx], frame.iloc[test_idx]
        train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=validation_size,
                                                    random_state=seed).split(
            train_val, groups=groups.loc[train_val.index]))
        train_original, validation = train_val.iloc[train_idx], train_val.iloc[val_idx]
        if groups.loc[test.index].nunique() < 2:
            raise ValueError("Reserve pelo menos dois grupos para estimar a incerteza do teste.")
    return prepare_partitions(train_original, validation, test,
                              augmented_size=augmented_size,
                              pca_variance=pca_variance, seed=seed, groups=groups)


def prepare_partitions(train_original, validation, test, *, augmented_size=1000,
                       pca_variance=0.999, seed=42, groups=None):
    """Ajusta MixUp e PCA em partições explícitas, inclusive dentro de cada fold.

    test também representa o fold externo de avaliação na validação cruzada.
    Índices devem preservar os IDs originais, sem reset entre as partições.
    """
    from models.data_augmentation import augment_training_data

    parts = (train_original, validation, test)
    for part in parts:
        validate_data(part)
        if not part.index.is_unique:
            raise ValueError("Cada partição deve ter índices únicos.")
        if not part.columns.equals(train_original.columns):
            raise ValueError("As partições devem ter as mesmas colunas e ordem.")
    ids = [set(part.index) for part in parts]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("Treino, validação e avaliação não podem compartilhar amostras.")
    if groups is not None:
        group_ids = [set(validate_groups(part, groups)) for part in parts]
        if group_ids[0] & group_ids[1] or group_ids[0] & group_ids[2] or group_ids[1] & group_ids[2]:
            raise ValueError("Um grupo não pode aparecer em mais de uma partição.")
    if not 0 < pca_variance < 1:
        raise ValueError("pca_variance deve estar entre 0 e 1.")
    train = augment_training_data(train_original, total_samples=augmented_size, seed=seed)
    columns = echo_columns(train_original)
    pca = PCA(n_components=pca_variance, svd_solver="full")
    train_pca = pca.fit_transform(train[columns])
    return PreparedData(
        train_original, train, validation, test, pca, train_pca,
        pca.transform(validation[columns]), pca.transform(test[columns]),
    )
