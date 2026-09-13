"""Divisão antes do MixUp e ajuste do PCA apenas no treino."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from .data import echo_columns, validate_data


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
                     augmented_size=1000, pca_variance=0.999, seed=42):
    """Os índices originais permitem auditar as três partições reais.

    As linhas representam amostras distintas do mesmo poço. Com 575 amostras,
    a divisão padrão produz 320 treino, 80 validação e 175 teste.
    Somente as amostras reais de treino originam as amostras sintéticas.
    """
    validate_data(frame)
    if not frame.index.is_unique:
        raise ValueError("O índice deve identificar cada amostra de forma única.")
    if not 0 < validation_size < 1 or not 0 < pca_variance < 1:
        raise ValueError("validation_size e pca_variance devem estar entre 0 e 1.")
    train_val, test = train_test_split(frame, test_size=test_size, random_state=seed)
    train_original, validation = train_test_split(
        train_val, test_size=validation_size, random_state=seed
    )
    return prepare_partitions(train_original, validation, test,
                              augmented_size=augmented_size,
                              pca_variance=pca_variance, seed=seed)


def prepare_partitions(train_original, validation, test, *, augmented_size=1000,
                       pca_variance=0.999, seed=42):
    """Ajusta PCA em partições explícitas, inclusive dentro de cada fold.

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
