"""Mesmos modelos, entradas e épocas para holdout e validação cruzada."""

from .mlp import MLP
from .pinn import PINN
from .pinn_log_t2 import PINNLogT2
from .pinn_weighted import PINNWeighted
from preprocess.data import echo_columns

MODEL_LABELS = {"mlp": "MLP + PCA", "pinn": "PINN original",
                "pinn_log_t2": "PINN log(T2)", "pinn_weighted": "PINN ponderada",
                "dummy_mean": "Média do treino"}


def model_specs(prepared, mlp_epochs=40, pinn_epochs=120):
    columns = echo_columns(prepared.train)
    raw = tuple(part[columns].to_numpy() for part in
                (prepared.train, prepared.validation, prepared.test))
    return [
        ("mlp", MODEL_LABELS["mlp"], MLP, mlp_epochs,
         prepared.train_pca, prepared.validation_pca, prepared.test_pca),
        ("pinn", MODEL_LABELS["pinn"], PINN, pinn_epochs, *raw),
        ("pinn_log_t2", MODEL_LABELS["pinn_log_t2"], PINNLogT2, pinn_epochs, *raw),
        ("pinn_weighted", MODEL_LABELS["pinn_weighted"], PINNWeighted, pinn_epochs, *raw),
    ]

