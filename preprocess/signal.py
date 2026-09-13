"""DWT, normalização por amostra, truncamento e média dos ecos."""

import numpy as np
import pywt


def dwt_denoise(noisy, wavelet="db6", level=6, mode="soft"):
    """Filtro do notebook; retorna sinal com tamanho original e coeficientes.

    A estimativa de ruído mantém o fator 2 usado em preprocessing.ipynb.
    """
    signal = np.array(noisy, dtype=float, copy=True)
    if signal.ndim != 1 or signal.size < 2 or not np.isfinite(signal).all():
        raise ValueError("O sinal deve ser um vetor finito com pelo menos 2 ecos.")
    max_level = pywt.dwt_max_level(signal.size, pywt.Wavelet(wavelet).dec_len)
    if not 1 <= level <= max_level:
        raise ValueError(f"Nível DWT deve estar entre 1 e {max_level} para este sinal.")
    coefficients = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coefficients[-1])) / 0.6745 * 2
    threshold = sigma * np.sqrt(2 * np.log(signal.size))
    # Um limiar zero deve preservar coeficientes nulos sem divisão 0/0.
    filtered = [coefficients[0]] + [
        pywt.threshold(detail, threshold, mode=mode) if threshold > 0 else detail.copy()
        for detail in coefficients[1:]
    ]
    return pywt.waverec(filtered, wavelet)[:signal.size], filtered


def normalize_rows(signals):
    """Min-max por linha; sinais constantes resultam em zeros."""
    signals = np.asarray(signals, dtype=float)
    minimum = signals.min(axis=1, keepdims=True)
    span = signals.max(axis=1, keepdims=True) - minimum
    return np.divide(signals - minimum, span, out=np.zeros_like(signals), where=span > 0)


def truncate_and_bin(signals, trim_last=1000, bin_size=4):
    """Remove a cauda e agrupa ecos consecutivos por média."""
    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2:
        raise ValueError("signals deve ser uma matriz de amostras por ecos.")
    if not 0 <= trim_last < signals.shape[1] or bin_size < 1:
        raise ValueError("Truncamento ou tamanho de bin inválido.")
    truncated = signals[:, :-trim_last] if trim_last else signals.copy()
    if truncated.shape[1] % bin_size:
        raise ValueError("O número de ecos após truncamento deve ser divisível pelo bin.")
    binned = truncated.reshape(len(signals), -1, bin_size).mean(axis=2)
    return truncated, binned

