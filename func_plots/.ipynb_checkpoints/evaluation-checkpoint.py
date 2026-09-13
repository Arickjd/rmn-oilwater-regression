"""Comparação dos quatro modelos no mesmo conjunto de amostras."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_model_scatter_grid(y_true, predictions, *, title="Avaliação no conjunto de teste"):
    """Grid 1x4 de real vs. predito com limites comuns, identidade e métricas.

    predictions é um dict {nome: vetor}, na ordem desejada dos quatro modelos.
    Todos os vetores devem corresponder à mesma ordem de amostras de y_true.
    Retorna (fig, axes), permitindo ajustar e exportar a figura externamente.
    """
    if len(predictions) != 4:
        raise ValueError("O grid 1x4 requer exatamente quatro modelos.")
    y_true = np.asarray(y_true).reshape(-1)
    arrays = [np.asarray(pred).reshape(-1) for pred in predictions.values()]
    if len(y_true) < 2 or any(len(pred) != len(y_true) for pred in arrays):
        raise ValueError("Cada predição deve corresponder às mesmas amostras reais (n >= 2).")
    combined = np.concatenate([y_true, *arrays])
    if not np.isfinite(combined).all():
        raise ValueError("Valores reais e predições devem ser finitos.")
    low, high = combined.min(), combined.max()
    margin = max(float(high - low) * 0.06, 0.01)
    limits = (low - margin, high + margin)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), sharex=True, sharey=True,
                             layout="constrained")
    colors = ("#6F4C9B", "#007C91", "#CC6677", "#447744")
    for ax, name, pred, color in zip(axes, predictions, arrays, colors):
        rmse = np.sqrt(mean_squared_error(y_true, pred))
        mae = mean_absolute_error(y_true, pred)
        r2 = r2_score(y_true, pred)
        ax.scatter(y_true, pred, s=24, alpha=0.65, color=color, edgecolors="white", linewidths=0.3)
        ax.plot(limits, limits, "--", color="0.3", linewidth=1, label="Ideal")
        ax.set(xlim=limits, ylim=limits, title=name, xlabel=r"Real ($Sw_{irr}$)")
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.04, 0.96, f"RMSE = {rmse:.4f}\nMAE = {mae:.4f}\n$R^2$ = {r2:.4f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r"Predito ($Sw_{irr}$)")
    fig.suptitle(title)
    return fig, axes


def plot_training_histories(histories):
    """Uma coluna por modelo, exibindo MSE do alvo e de reconstrução."""
    fig, axes = plt.subplots(1, len(histories), figsize=(4.3 * len(histories), 4),
                             squeeze=False, layout="constrained")
    for ax, (name, history) in zip(axes[0], histories.items()):
        ax.plot(history["epoch"], history["train_output"], label="Treino: alvo")
        ax.plot(history["epoch"], history["val_output"], label="Validação: alvo")
        if history["train_physics"].ne(0).any():
            ax.plot(history["epoch"], history["train_physics"], "--", label="Treino: física")
            ax.plot(history["epoch"], history["val_physics"], "--", label="Validação: física")
        ax.set(title=name, xlabel="Época", ylabel="Perda")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    return fig, axes[0]

