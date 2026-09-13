"""Distribuições, denoising, curvas coloridas pelo alvo e diagnóstico do PCA."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from preprocess.data import PROPERTY_COLUMNS


def plot_distributions(frame, columns=PROPERTY_COLUMNS):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    if len(columns) != 4:
        raise ValueError("A distribuição 2x2 requer quatro colunas.")
    for ax, column in zip(axes.flat, columns):
        ax.hist(frame[column], bins=30, color="#007C91", edgecolor="white")
        ax.set(title=column, ylabel="Frequência")
        ax.grid(axis="y", alpha=0.2)
    return fig, axes


def plot_denoising(raw, denoised, *, n_samples=3, seed=42):
    raw, denoised = np.asarray(raw), np.asarray(denoised)
    if raw.shape != denoised.shape or raw.ndim != 2 or n_samples < 1 or not len(raw):
        raise ValueError("Os sinais devem ser matrizes não vazias de mesmo formato.")
    indices = np.random.default_rng(seed).choice(len(raw), min(n_samples, len(raw)), replace=False)
    fig, axes = plt.subplots(len(indices), 1, figsize=(10, 2.7 * len(indices)),
                             sharex=True, squeeze=False, layout="constrained")
    for ax, index in zip(axes[:, 0], indices):
        ax.plot(raw[index], color="0.75", label="Original", linewidth=1)
        ax.plot(denoised[index], color="#CC6677", label="DWT", linewidth=1)
        ax.set(title=f"Amostra {index}", ylabel="Amplitude")
        ax.legend()
    axes[-1, 0].set_xlabel("Índice do eco")
    return fig, axes[:, 0]


def plot_signal_curves(signals, targets, *, max_curves=100, seed=42,
                       title="Decaimentos de RMN processados"):
    signals, targets = np.asarray(signals), np.asarray(targets)
    indices = np.random.default_rng(seed).choice(len(signals), min(max_curves, len(signals)), replace=False)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    norm = Normalize(vmin=targets.min(), vmax=targets.max())
    cmap = plt.get_cmap("viridis")
    for index in indices:
        ax.plot(signals[index], color=cmap(norm(targets[index])), alpha=0.3, linewidth=0.8)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label=r"$Sw_{irr}$")
    ax.set(title=title, xlabel="Índice do bin de ecos", ylabel="Amplitude normalizada")
    ax.grid(alpha=0.2)
    return fig, ax


def plot_pca(pca, transformed, targets, *, variance_target=0.999):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    ratios = pca.explained_variance_ratio_
    axes[0].plot(np.arange(1, len(ratios) + 1), np.cumsum(ratios), "o-", color="#007C91")
    axes[0].axhline(variance_target, color="#CC6677", linestyle="--", label=f"Alvo: {variance_target:.1%}")
    axes[0].set(xlabel="Número de componentes", ylabel="Variância explicada acumulada")
    axes[0].legend()
    if transformed.shape[1] >= 2:
        scatter = axes[1].scatter(transformed[:, 0], transformed[:, 1], c=targets,
                                  cmap="viridis", s=15, alpha=0.6)
        fig.colorbar(scatter, ax=axes[1], label=r"$Sw_{irr}$")
        axes[1].set(xlabel=f"PC1 ({ratios[0]:.1%})", ylabel=f"PC2 ({ratios[1]:.1%})")
    else:
        axes[1].text(0.5, 0.5, "PCA reteve apenas uma componente", ha="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()
    return fig, axes

