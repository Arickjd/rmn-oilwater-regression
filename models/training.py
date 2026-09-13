"""Treinamento, seleção por validação e avaliação comuns aos quatro modelos."""

import copy
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(x, y, batch_size, shuffle, seed):
    features = torch.as_tensor(np.asarray(x, dtype=np.float32))
    targets = torch.as_tensor(np.asarray(y, dtype=np.float32).reshape(-1))
    if features.ndim != 2 or len(features) != len(targets) or not len(targets):
        raise ValueError("Features e alvos devem ter o mesmo número não nulo de amostras.")
    return DataLoader(TensorDataset(features, targets), batch_size=batch_size,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(seed))


def train_model(model, x_train, y_train, x_val, y_val, *, epochs=120,
                batch_size=256, learning_rate=1e-3, device="cpu", seed=42,
                verbose=True):
    """Restaura o checkpoint de menor perda total na validação real.

    Scheduler e escolha do checkpoint usam os mesmos pesos da perda de treino.
    O conjunto de teste nunca é recebido por esta função.
    """
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size e learning_rate devem ser positivos.")
    model.to(device)
    train_loader = _loader(x_train, y_train, batch_size, True, seed)
    val_loader = _loader(x_val, y_val, batch_size, False, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    ) if model.use_scheduler else None)
    history, best_state, best_epoch, best_loss = [], None, 0, float("inf")
    for epoch in range(1, epochs + 1):
        row = {"epoch": epoch}
        for phase, loader in (("train", train_loader), ("val", val_loader)):
            model.train(phase == "train")
            totals = np.zeros(3)
            with torch.set_grad_enabled(phase == "train"):
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    output, physics = model.loss_components(xb, yb)
                    loss = model.output_weight * output + model.physics_weight * physics
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Perda não finita na época {epoch} ({phase}).")
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    totals += np.array([loss.item(), output.item(), physics.item()]) * len(xb)
            for key, value in zip(("total", "output", "physics"), totals / len(loader.dataset)):
                row[f"{phase}_{key}"] = value
        if row["val_total"] < best_loss:
            best_loss, best_epoch = row["val_total"], epoch
            best_state = copy.deepcopy(model.state_dict())
        if scheduler is not None:
            scheduler.step(row["val_total"])
        history.append(row)
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(f"  Época {epoch:03d}/{epochs}: treino={row['train_total']:.6f} "
                  f"validação={row['val_total']:.6f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return pd.DataFrame(history), best_epoch


def predict(model, x, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    values = torch.as_tensor(np.asarray(x, dtype=np.float32))
    with torch.no_grad():
        return torch.cat([model(batch.to(device)).cpu()
                          for batch in values.split(batch_size)]).numpy().reshape(-1)


def regression_metrics(y_true, y_pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": (float(r2_score(y_true, y_pred))
                   if len(y_true) >= 2 and np.ptp(np.asarray(y_true)) > 0 else float("nan"))}
