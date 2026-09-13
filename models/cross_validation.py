"""K-fold repetido com holdout interno para época/scheduler; teste final excluído.

Não é uma busca de hiperparâmetros por nested K-fold: o nível interno é uma
divisão treino/validação. O fold externo nunca escolhe época, PCA ou scheduler.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, train_test_split

from preprocess.data import TARGET, validate_data
from preprocess.split import prepare_partitions
from .registry import model_specs
from .training import predict, regression_metrics, set_seed, train_model


@dataclass
class CrossValidationResult:
    fold_metrics: pd.DataFrame
    repeat_metrics: pd.DataFrame
    summary: pd.DataFrame
    predictions: pd.DataFrame
    splits: pd.DataFrame
    selected_model: str


def iter_cv_partitions(development, *, folds=5, repeats=3, validation_size=0.2, seed=42):
    """Retorna somente índices reais; cada amostra é avaliada uma vez/repetição."""
    validate_data(development)
    if not development.index.is_unique:
        raise ValueError("O conjunto de desenvolvimento deve ter IDs únicos.")
    if folds < 2 or repeats < 1 or len(development) // folds < 2:
        raise ValueError("Use >=2 folds, >=1 repetição e >=2 amostras por fold externo.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size deve estar entre 0 e 1.")
    cv = RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    outer_splits = cv.split(development)
    for number, (outer_train, outer_test) in enumerate(outer_splits):
        fold_seed = seed + number + 1
        inner_train, inner_val = train_test_split(
            outer_train, test_size=validation_size, random_state=fold_seed
        )
        if min(len(inner_train), len(inner_val), len(outer_test)) < 2:
            raise ValueError("Treino, validação interna e avaliação externa precisam de >=2 amostras.")
        yield (number // folds + 1, number % folds + 1, fold_seed,
               development.index[inner_train], development.index[inner_val],
               development.index[outer_test])


def summarize_scores(scores):
    """Dispersão descritiva, sem tratar folds correlacionados como independentes."""
    long = scores.melt(id_vars=["model"], value_vars=["RMSE", "MAE", "R2"],
                       var_name="metric", value_name="value")
    return long.groupby(["model", "metric"], sort=False)["value"].agg(
        mean="mean", std="std", minimum="min", maximum="max", count="count"
    ).reset_index()


def run_cross_validation(development, *, output_dir, folds=5, repeats=3,
                         validation_size=0.2, augmented_size=1000, pca_variance=0.999,
                         mlp_epochs=40, pinn_epochs=120, batch_size=256,
                         learning_rate=1e-3, device="cpu", seed=42):
    """Recebe exclusivamente o desenvolvimento, nunca o teste final.

    Métricas OOF são calculadas por repetição, sem tirar média das predições
    repetidas (o que avaliaria um ensemble diferente do modelo individual).
    """
    output = Path(output_dir)
    partitions = list(iter_cv_partitions(development, folds=folds, repeats=repeats,
                                         validation_size=validation_size, seed=seed))
    output.mkdir(parents=True, exist_ok=True)
    assignments = []
    for repeat, fold, fold_seed, train_ids, val_ids, test_ids in partitions:
        for split, ids in (("train", train_ids), ("validation", val_ids), ("assessment", test_ids)):
            assignments.append(pd.DataFrame({"repeat": repeat, "fold": fold, "seed": fold_seed,
                                              "sample_id": ids, "split": split}))
    assignments = pd.concat(assignments, ignore_index=True)
    assignments.to_csv(output / "splits.csv", index=False)
    fold_rows, prediction_rows = [], []
    for repeat, fold, fold_seed, train_ids, val_ids, test_ids in partitions:
        print(f"CV: repetição {repeat}/{repeats}, fold {fold}/{folds}", flush=True)
        prepared = prepare_partitions(
            development.loc[train_ids], development.loc[val_ids], development.loc[test_ids],
            augmented_size=augmented_size, pca_variance=pca_variance, seed=fold_seed,
        )
        y_train = prepared.train[TARGET].to_numpy()
        y_val = prepared.validation[TARGET].to_numpy()
        y_test = prepared.test[TARGET].to_numpy()

        def record(key, prediction, best_epoch):
            scores = regression_metrics(y_test, prediction)
            fold_rows.append({"repeat": repeat, "fold": fold, "seed": fold_seed,
                              "model": key, "best_epoch": best_epoch,
                              "n_train_real": len(train_ids), "n_validation": len(val_ids),
                              "n_assessment": len(test_ids), "pca_components": prepared.pca.n_components_,
                              **scores})
            prediction_rows.append(pd.DataFrame({"repeat": repeat, "fold": fold,
                                                 "sample_id": test_ids, "model": key,
                                                 "y_true": y_test, "y_pred": prediction}))
            print(f"  {key}: RMSE={scores['RMSE']:.4f}, R2={scores['R2']:.4f}", flush=True)

        record("dummy_mean", np.full(len(y_test), prepared.train_original[TARGET].mean()), 0)
        for key, _, model_class, epochs, x_train, x_val, x_test in model_specs(prepared, mlp_epochs, pinn_epochs):
            set_seed(fold_seed)
            model = model_class(x_train.shape[1])
            history, best_epoch = train_model(
                model, x_train, y_train, x_val, y_val, epochs=epochs, batch_size=batch_size,
                learning_rate=learning_rate, device=device, seed=fold_seed, verbose=False,
            )
            history.to_csv(output / f"history_r{repeat}_f{fold}_{key}.csv", index=False)
            record(key, predict(model, x_test, batch_size), best_epoch)
        # Checkpoints de progresso tabulares para auditar uma execução interrompida.
        pd.DataFrame(fold_rows).to_csv(output / "fold_metrics.csv", index=False)
        pd.concat(prediction_rows, ignore_index=True).to_csv(output / "predictions_oof.csv", index=False)
    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    repeat_rows = []
    for (repeat, key), part in predictions.groupby(["repeat", "model"]):
        if len(part) != len(development) or not part.sample_id.is_unique:
            raise RuntimeError("Cobertura OOF inválida: cada amostra deve aparecer uma vez por repetição/modelo.")
        repeat_rows.append({"repeat": repeat, "model": key,
                            **regression_metrics(part.y_true, part.y_pred)})
    repeat_metrics = pd.DataFrame(repeat_rows)
    summary = summarize_scores(fold_metrics)
    repeat_metrics.to_csv(output / "repeat_metrics.csv", index=False)
    summary.to_csv(output / "fold_summary.csv", index=False)
    summarize_scores(repeat_metrics).to_csv(output / "repeat_summary.csv", index=False)
    ranking = repeat_metrics.groupby("model").RMSE.mean().sort_values()
    selected = str(ranking.index[0])
    # Seleção registrada antes de ajustar/avaliar os modelos finais no holdout.
    (output / "selection.json").write_text(json.dumps({
        "selected_model": selected, "criterion": "mean_repeat_oof_RMSE",
        "ranking": ranking.to_dict(), "folds": folds, "repeats": repeats,
        "inner_validation_fraction": validation_size, "seed": seed,
        "independence_assumption": "independent_samples_same_well",
        "final_test_used": False,
    }, indent=2), encoding="utf-8")
    return CrossValidationResult(fold_metrics, repeat_metrics, summary, predictions, assignments, selected)
