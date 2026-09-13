"""Executa pré-processamento, MixUp, PCA, quatro modelos e figuras do artigo."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch

from models.cross_validation import run_cross_validation
from models.evaluation import bootstrap_test_metrics, write_evaluation_report
from models.registry import model_specs
from models.training import predict, regression_metrics, set_seed, train_model
from preprocess import TARGET, echo_columns, load_data, prepare_datasets, preprocess_data
from preprocess.groups import load_groups

PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "RMN_data/GulfCoast_RMN_Synthetic.csv")
    parser.add_argument("--preprocessed", action="store_true", help="O CSV de entrada já contém os ecos processados.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output/pipeline")
    parser.add_argument("--seed", type=int, default=42)
    grouping = parser.add_mutually_exclusive_group(required=True)
    grouping.add_argument("--groups-file", type=Path, help="CSV sample_id,group; obrigatório para curvas agrupadas.")
    grouping.add_argument("--independent-samples", action="store_true",
                          help="Somente para dados cujas linhas são comprovadamente independentes.")
    parser.add_argument("--test-size", type=int, default=175, help="Número de amostras de teste no modo independente.")
    parser.add_argument("--test-group-fraction", type=float, default=0.3, help="Fração dos grupos reservada para teste.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Folds externos de CV; 0 desativa CV.")
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--validation-size", type=float, default=0.2, help="Fração do restante reservada para validação.")
    parser.add_argument("--augmented-size", type=int, default=1000, help="Tamanho final do treino; 0 desativa MixUp.")
    parser.add_argument("--pca-variance", type=float, default=0.999)
    parser.add_argument("--mlp-epochs", type=int, default=40)
    parser.add_argument("--pinn-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4, help="Threads PyTorch na CPU.")
    parser.add_argument("--wavelet", default="db6")
    parser.add_argument("--dwt-level", type=int, default=6)
    parser.add_argument("--trim-last", type=int, default=1000)
    parser.add_argument("--bin-size", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def run_pipeline(args):
    if min(args.mlp_epochs, args.pinn_epochs, args.batch_size, args.threads) < 1:
        raise ValueError("Épocas, batch-size e threads devem ser positivos.")
    if (args.cv_folds != 0 and args.cv_folds < 2) or args.cv_repeats < 1:
        raise ValueError("Use cv-folds=0 ou >=2 e cv-repeats>=1.")
    if args.bootstrap_resamples < 100 or not 0 < args.confidence < 1:
        raise ValueError("Use >=100 reamostragens e confiança entre 0 e 1.")
    if args.groups_file is None and not args.independent_samples:
        raise ValueError("Informe os grupos; curvas relacionadas não podem ser divididas por linha.")
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA não está disponível; use --device cpu.")
    print(f"Dispositivo: {device}. Carregando {args.input}", flush=True)
    raw = load_data(args.input)
    groups = load_groups(args.groups_file, raw) if args.groups_file is not None else None
    preprocessing = None
    if args.preprocessed:
        frame = raw
    else:
        print("Aplicando DWT, min-max por sinal, truncamento e média por bin...", flush=True)
        preprocessing = preprocess_data(raw, wavelet=args.wavelet, level=args.dwt_level,
                                         trim_last=args.trim_last, bin_size=args.bin_size)
        frame = preprocessing.frame
    prepared = prepare_datasets(
        frame, test_size=args.test_size, validation_size=args.validation_size,
        augmented_size=args.augmented_size, pca_variance=args.pca_variance, seed=args.seed,
        groups=groups, test_group_fraction=args.test_group_fraction,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cv = None
    if args.cv_folds:
        # Toda a CV recebe apenas desenvolvimento, inclusive seus grupos.
        development = frame.drop(index=prepared.test.index)
        cv = run_cross_validation(
            development, output_dir=output / "cross_validation", folds=args.cv_folds,
            repeats=args.cv_repeats, validation_size=args.validation_size,
            augmented_size=args.augmented_size, pca_variance=args.pca_variance,
            mlp_epochs=args.mlp_epochs, pinn_epochs=args.pinn_epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate, device=device,
            seed=args.seed, groups=groups.loc[development.index] if groups is not None else None,
        )
        print(f"Modelo selecionado pela CV (antes do teste): {cv.selected_model}", flush=True)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    frame.to_csv(output / "model_input.csv", index=False)
    # Índice da linha do CSV original (base zero), antes do embaralhamento/MixUp.
    splits = pd.concat([
        pd.DataFrame({"sample_id": part.index, "split": name})
        for name, part in (("train", prepared.train_original),
                           ("validation", prepared.validation), ("test", prepared.test))
    ], ignore_index=True)
    if groups is not None:
        splits["group"] = splits.sample_id.map(groups)
    splits.to_csv(output / "splits.csv", index=False)
    prepared.train.to_csv(output / "train_augmented.csv", index=False)
    joblib.dump(prepared.pca, checkpoints / "pca.joblib")
    print(f"Treino real: {len(prepared.train_original)}; aumentado: {len(prepared.train)}; "
          f"validação: {len(prepared.validation)}; teste: {len(prepared.test)}; "
          f"PCA: {prepared.pca.n_components_} componentes.", flush=True)
    columns = echo_columns(frame)
    train_x = prepared.train[columns].to_numpy()
    train_y = prepared.train[TARGET].to_numpy()
    val_y = prepared.validation[TARGET].to_numpy()
    test_y = prepared.test[TARGET].to_numpy()
    specs = model_specs(prepared, args.mlp_epochs, args.pinn_epochs)
    predictions, histories, metrics, best_epochs = {}, {}, [], {}
    prediction_frame = pd.DataFrame({"sample_id": prepared.test.index, "y_true": test_y})
    if groups is not None:
        prediction_frame["group"] = prediction_frame.sample_id.map(groups)
    for key, label, model_class, epochs, x_train, x_val, x_test in specs:
        print(f"Treinando {label}...", flush=True)
        set_seed(args.seed)
        model = model_class(x_train.shape[1])
        history, best_epoch = train_model(
            model, x_train, train_y, x_val, val_y, epochs=epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate, device=device, seed=args.seed,
        )
        prediction = predict(model, x_test, args.batch_size)
        predictions[label], histories[label], best_epochs[key] = prediction, history, best_epoch
        prediction_frame[key] = prediction
        history.to_csv(output / f"history_{key}.csv", index=False)
        for split, y_true, y_pred in (("validation", val_y, predict(model, x_val, args.batch_size)),
                                      ("test", test_y, prediction)):
            metrics.append({"model": key, "split": split, **regression_metrics(y_true, y_pred)})
        torch.save({"model_name": key, "input_dim": x_train.shape[1],
                    "state_dict": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
                    "best_epoch": best_epoch, "echo_columns": columns, "seed": args.seed},
                   checkpoints / f"{key}.pt")
    baseline = float(prepared.train_original[TARGET].mean())
    prediction_frame["dummy_mean"] = baseline
    for split, values in (("validation", val_y), ("test", test_y)):
        metrics.append({"model": "dummy_mean", "split": split,
                        **regression_metrics(values, np.full(len(values), baseline))})
    intervals, comparisons = bootstrap_test_metrics(
        test_y, {key: prediction_frame[key].to_numpy() for key in [*(spec[0] for spec in specs), "dummy_mean"]},
        n_resamples=args.bootstrap_resamples, confidence=args.confidence, seed=args.seed,
        groups=groups.loc[prepared.test.index].to_numpy() if groups is not None else None,
    )
    intervals.to_csv(output / "test_confidence_intervals.csv", index=False)
    comparisons.to_csv(output / "test_paired_comparisons.csv", index=False)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output / "metrics.csv", index=False)
    prediction_frame.to_csv(output / "predictions_test.csv", index=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(device_used=device, pca_components=int(prepared.pca.n_components_), best_epochs=best_epochs,
                  train_real=len(prepared.train_original), train_augmented=len(prepared.train),
                  validation_samples=len(prepared.validation), test_samples=len(prepared.test))
    config.update(selected_model_by_cv=cv.selected_model if cv is not None else None,
                  split_unit="group" if groups is not None else "row",
                  baseline_train_mean=baseline,
                  test_groups=groups.loc[prepared.test.index].nunique() if groups is not None else None)
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    write_evaluation_report(output / "evaluation_report.md", intervals, cv=cv, config=config)
    if not args.no_plots:
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from func_plots import (plot_denoising, plot_distributions, plot_model_scatter_grid,
                                plot_pca, plot_signal_curves, plot_training_histories, save_figure)
        from func_plots.validation import plot_cv_scores, plot_test_confidence_intervals

        def export(name, plot):
            figure, _ = plot
            save_figure(figure, output / "figures" / name)
            plt.close(figure)

        export("scatter_models_1x4", plot_model_scatter_grid(test_y, predictions))
        export("training_histories", plot_training_histories(histories))
        export("test_confidence_intervals", plot_test_confidence_intervals(intervals))
        if cv is not None:
            export("cv_fold_scores", plot_cv_scores(cv.fold_metrics))
            # Primeira repetição: uma predição externa por amostra, sem ensemble.
            oof = cv.predictions[cv.predictions["repeat"] == 1]
            wide = oof.pivot(index="sample_id", columns="model", values="y_pred")
            truth = oof.drop_duplicates("sample_id").set_index("sample_id").loc[wide.index, "y_true"]
            export("scatter_oof_1x4", plot_model_scatter_grid(
                truth, {label: wide[key] for key, label, *_ in specs},
                title="Predições fora do fold — primeira repetição da CV"))
        export("distributions", plot_distributions(frame))
        export("pca", plot_pca(prepared.pca, prepared.train_pca, train_y, variance_target=args.pca_variance))
        export("augmented_signals", plot_signal_curves(train_x, train_y, seed=args.seed,
                                                       title="Decaimentos de RMN — treino aumentado"))
        if preprocessing is not None:
            export("denoising", plot_denoising(preprocessing.raw, preprocessing.denoised, seed=args.seed))
    print("\nMétricas no teste reservado:", flush=True)
    print(metrics_frame[metrics_frame["split"] == "test"].to_string(index=False))
    print(f"\nResultados salvos em {output}", flush=True)
    return metrics_frame


def main():
    args = build_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
