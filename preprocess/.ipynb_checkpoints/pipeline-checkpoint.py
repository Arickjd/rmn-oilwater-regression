"""Etapas independentes por amostra extraídas de preprocessing.ipynb."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import PROPERTY_COLUMNS, echo_columns, validate_data
from .signal import dwt_denoise, normalize_rows, truncate_and_bin


@dataclass
class PreprocessingResult:
    frame: pd.DataFrame
    raw: np.ndarray
    denoised: np.ndarray
    truncated: np.ndarray


def preprocess_data(frame, *, wavelet="db6", level=6, trim_last=1000, bin_size=4):
    validate_data(frame)
    raw = frame[echo_columns(frame)].to_numpy(dtype=float)
    denoised = np.stack([dwt_denoise(row, wavelet, level)[0] for row in raw])
    normalized = normalize_rows(denoised)
    truncated, binned = truncate_and_bin(normalized, trim_last, bin_size)
    columns = [f"Echo_Bin_{i + 1}" for i in range(binned.shape[1])]
    processed = pd.DataFrame(binned, index=frame.index, columns=columns)
    processed = pd.concat([processed, frame[list(PROPERTY_COLUMNS)]], axis=1)
    return PreprocessingResult(processed, raw, denoised, truncated)

