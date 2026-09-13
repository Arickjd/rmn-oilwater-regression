"""Figuras reutilizáveis para o artigo; nenhuma função chama plt.show()."""

from .evaluation import plot_model_scatter_grid, plot_training_histories
from .preprocessing import plot_denoising, plot_distributions, plot_pca, plot_signal_curves
from .utils import save_figure

__all__ = ["plot_model_scatter_grid", "plot_training_histories", "plot_denoising",
           "plot_distributions", "plot_pca", "plot_signal_curves", "save_figure"]

