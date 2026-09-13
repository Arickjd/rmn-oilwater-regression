"""Dispersão da validação cruzada e incerteza condicional do teste."""

import matplotlib.pyplot as plt
import numpy as np

from models.registry import MODEL_LABELS


def plot_cv_scores(fold_metrics):
    """Pontos dos folds e média; folds correlacionados não são tratados como IC."""
    models = list(fold_metrics.model.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for ax, metric in zip(axes, ("RMSE", "MAE", "R2")):
        for position, model in enumerate(models):
            values = fold_metrics.loc[fold_metrics.model == model, metric].to_numpy()
            offsets = np.linspace(-0.18, 0.18, len(values))
            ax.scatter(position + offsets, values, s=23, alpha=0.7, color="#007C91")
            ax.plot([position - 0.22, position + 0.22], [values.mean()] * 2,
                    color="#CC6677", linewidth=2)
        ax.set_xticks(range(len(models)), [MODEL_LABELS[key] for key in models], rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Validação cruzada: pontos = folds; traço = média (sem IC)")
    return fig, axes


def plot_test_confidence_intervals(intervals):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, metric in zip(axes, ("RMSE", "MAE", "R2")):
        rows = intervals[intervals.metric == metric]
        for position, row in enumerate(rows.itertuples()):
            ax.plot([row.ci_low, row.ci_high], [position, position], color="#007C91", linewidth=2)
            ax.scatter(row.estimate, position, color="#CC6677", zorder=3)
        ax.set_yticks(range(len(rows)), [MODEL_LABELS[key] for key in rows.model])
        ax.set_xlabel(metric)
        ax.grid(axis="x", alpha=0.2)
    confidence = intervals.confidence.iloc[0]
    fig.suptitle(f"Teste: IC {confidence:.0%} por bootstrap de amostras, condicionado ao ajuste")
    return fig, axes
