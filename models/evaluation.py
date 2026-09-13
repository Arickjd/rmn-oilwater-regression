"""Bootstrap pareado do teste, condicional aos modelos já ajustados."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _scores(y, prediction):
    """Métricas na última dimensão; R² indefinido permanece NaN."""
    residuals = y - prediction
    mse = np.mean(residuals ** 2, axis=-1)
    variance = np.mean((y - y.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
    # Mesmo um vetor constante pode ter variância residual de arredondamento.
    ratio = np.divide(mse, variance, out=np.full_like(mse, np.nan),
                      where=(variance > 0) & (np.ptp(y, axis=-1) > 0))
    return {"RMSE": np.sqrt(mse), "MAE": np.mean(np.abs(residuals), axis=-1), "R2": 1 - ratio}


def bootstrap_test_metrics(y_true, predictions, *, n_resamples=2000,
                           confidence=0.95, seed=42):
    """IC percentil com reamostragem de pares (alvo/predições) da mesma linha.

    Retorna métricas/IC e diferenças pareadas entre modelos. Não reestima
    treinamento, não produz IC de CV nem intervalos de predição individuais.
    Os IC das diferenças são exploratórios, sem ajuste por múltiplas comparações.
    Reamostras com alvo constante são excluídas apenas do IC de R² e contadas.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    arrays = {key: np.asarray(value, dtype=float).reshape(-1) for key, value in predictions.items()}
    if len(y) < 2 or not arrays or not np.isfinite(y).all():
        raise ValueError("Forneça pelo menos dois alvos finitos e um modelo.")
    if any(value.shape != y.shape or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Predições finitas devem estar alinhadas com os alvos.")
    if n_resamples < 100 or not 0 < confidence < 1:
        raise ValueError("Use pelo menos 100 reamostragens e confiança entre 0 e 1.")
    rng = np.random.default_rng(seed)
    boot = {key: {metric: [] for metric in ("RMSE", "MAE", "R2")} for key in arrays}
    def resampling_indices():
        # Lotes limitam a memória no bootstrap de amostras independentes.
        for start in range(0, n_resamples, 256):
            yield rng.integers(0, len(y), size=(min(256, n_resamples - start), len(y)))

    for indices in resampling_indices():
        for key, prediction in arrays.items():
            for metric, values in _scores(y[indices], prediction[indices]).items():
                boot[key][metric].append(np.atleast_1d(values))
    boot = {key: {metric: np.concatenate(values) for metric, values in scores.items()}
            for key, scores in boot.items()}
    points = {key: _scores(y, value) for key, value in arrays.items()}
    quantiles = [(1 - confidence) / 2, (1 + confidence) / 2]

    def interval(values):
        finite = values[np.isfinite(values)]
        low, high = np.quantile(finite, quantiles) if len(finite) else (np.nan, np.nan)
        return {"ci_low": low, "ci_high": high, "valid_resamples": len(finite),
                "n_resamples": n_resamples, "confidence": confidence,
                "resampling_unit": "sample", "n_independent_units": len(y)}

    rows, differences = [], []
    for key in arrays:
        for metric in ("RMSE", "MAE", "R2"):
            rows.append({"model": key, "metric": metric, "estimate": float(points[key][metric]),
                         "n_test": len(y), **interval(boot[key][metric])})
    for first, second in combinations(arrays, 2):
        for metric in ("RMSE", "MAE", "R2"):
            differences.append({"model_a": first, "model_b": second, "metric": metric,
                                "difference_a_minus_b": float(points[first][metric] - points[second][metric]),
                                **interval(boot[first][metric] - boot[second][metric])})
    return pd.DataFrame(rows), pd.DataFrame(differences)


def write_evaluation_report(path, intervals, *, cv, config):
    """Relatório distingue dispersão de CV e IC condicional do teste."""
    lines = ["# Avaliação dos modelos", "",
             "Unidade de separação e bootstrap: amostra de RMN.",
             f"Treino real final: {config['train_real']}; validação interna: {config['validation_samples']}; "
             f"teste: {config['test_samples']} amostras."]
    lines += ["", "## Protocolo", "",
              "DWT e min-max são operações individuais por curva. MixUp e PCA são ajustados "
              "somente no treino de cada divisão. Validação interna escolhe época e scheduler; "
              "o conjunto externo não participa dessas escolhas."]
    if cv is not None:
        lines += [f"CV externa com {config['cv_folds']} folds × {config['cv_repeats']} repetições, "
                  "somente no desenvolvimento. A divisão interna é um holdout, não uma busca "
                  "de hiperparâmetros por nested K-fold.",
                  f"Modelo escolhido pelo menor RMSE OOF médio por repetição, antes do teste: **{cv.selected_model}**.",
                  "", "## CV: métricas OOF por repetição (média ± desvio-padrão)", "",
                  "| Modelo | RMSE | MAE | R² |", "| --- | --- | --- | --- |"]
        for model, scores in cv.repeat_metrics.groupby("model"):
            cells = []
            for metric in ("RMSE", "MAE", "R2"):
                values = scores[metric]
                cells.append(f"{values.mean():.4f} ± {values.std():.4f}" if len(values) > 1
                             else f"{values.mean():.4f} (uma repetição)")
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines += ["", "Cada curva tem uma predição externa por repetição. As métricas são calculadas "
                  "separadamente por repetição; não se misturam linhas repetidas como observações novas. "
                  "O desvio-padrão descreve sensibilidade às divisões e seeds, não é um IC. "
                  "Folds e repetições compartilham dados, portanto não foi usado desvio/raiz(n). "
                  "O desempenho CV do modelo escolhido pode ser otimista devido à seleção entre candidatos."]
    else:
        lines += ["CV desativada nesta execução; não houve seleção de modelo pela CV."]
    confidence = config["confidence"]
    lines += ["", f"## Teste: métricas e IC de {confidence:.0%}", "",
              "| Modelo | Métrica | Estimativa | IC inferior | IC superior | Reamostras válidas |",
              "| --- | --- | --- | --- | --- | --- |"]
    for row in intervals.itertuples():
        lines.append(f"| {row.model} | {row.metric} | {row.estimate:.4f} | {row.ci_low:.4f} | "
                     f"{row.ci_high:.4f} | {row.valid_resamples}/{row.n_resamples} |")
    lines += ["", "## Interpretação e limites", "",
              f"Bootstrap percentil com {config['bootstrap_resamples']} reamostragens pareadas de amostras. "
              "As métricas dão o mesmo peso a cada amostra. "
              "Os intervalos são condicionais aos modelos ajustados e não abrangem toda a incerteza "
              "do treinamento, seleção de arquitetura ou mudança de distribuição.",
              "R² é indefinido em reamostras de alvo constante; elas são excluídas apenas do IC de R² "
              "e sua quantidade é registrada. Um IC não é uma garantia nem um intervalo de predição individual.",
              "test_paired_comparisons.csv contém diferenças A − B nas mesmas reamostras. "
              "Para RMSE/MAE, valores negativos favorecem A; para R², positivos favorecem A. "
              "Esses IC são exploratórios e não corrigidos por múltiplas comparações.",
              "O conjunto de dados é composto por amostras de um único poço. O protocolo avalia "
              "a generalização para novas amostras desse mesmo poço, e não para novos poços. "
              "Além disso, ele já foi usado nos notebooks e no teste da versão anterior. "
              "A nova separação não apaga esse histórico. Para confirmação externa do resultado, "
              "use amostras de poços ainda não consultados e mantenha arquitetura/hiperparâmetros fixos.",
              "", "## Referências metodológicas", "",
              "- [Separação entre seleção e avaliação — scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)",
              "- [Validação cruzada — scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)",
              "- [Bootstrap pareado e percentil — SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
