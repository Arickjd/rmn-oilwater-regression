"""Leitura e validação do CSV bruto ou previamente processado."""

from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Swirr_PHIX"
PROPERTY_COLUMNS = (TARGET, "MBVI", "MPHI", "PHIX")


def echo_columns(frame: pd.DataFrame) -> list[str]:
    """Seleciona os ecos na ordem do CSV, excluindo as propriedades físicas."""
    return [column for column in frame.columns if column not in PROPERTY_COLUMNS]


def validate_data(frame: pd.DataFrame) -> None:
    missing = set(PROPERTY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Propriedades ausentes no CSV: {sorted(missing)}")
    if frame.empty or not echo_columns(frame):
        raise ValueError("O CSV deve conter amostras e colunas de ecos.")
    if not frame.columns.is_unique:
        raise ValueError("Os nomes das colunas devem ser únicos.")
    try:
        values = frame.to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Todas as colunas devem ser numéricas.") from exc
    if not np.isfinite(values).all():
        raise ValueError("O CSV contém valores ausentes ou infinitos.")
    if (frame["PHIX"] <= 0).any():
        raise ValueError("PHIX deve ser positivo para calcular MBVI / PHIX.")


def load_data(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_data(frame)
    return frame

