"""Protocolo das rodadas e os experimentos 1 a 7.

Em cada rodada:

1. Dividir em treino, validação e teste, estratificando por classe.
2. Ajustar cada configuração da grade só no treino e medir a validação.
3. Escolher a configuração pela regra fixa (só validação).
4. Avaliar a escolhida no teste, sem re-treinar com a validação.
5. O ``StandardScaler``, quando existe, fica dentro do ``Pipeline``.

O teste de todas as combinações com normalização é calculado depois da escolha
e serve apenas à curva de sensibilidade. Ele não volta para a seleção.
"""

from __future__ import annotations

import time

import numpy as np

from src.metrics import compute_metrics
from src.models import (
    config_to_string,
    find_default_configuration,
    iter_grid,
    make_baseline,
    make_knn_pipeline,
    select_best_configuration,
)
from src.splitter import repeated_stratified_holdout

def run_experiments(
    x: np.ndarray,
    y: np.ndarray,
    n_rounds: int = 5,
    base_seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    neighbors: list[int] | None = None,
    quick: bool = False,
) -> dict:
    """Executa todas as variantes nas mesmas divisões e devolve as linhas brutas."""
    if n_rounds < 2:
        raise ValueError(
            "n_rounds deve ser pelo menos 2 para o desvio padrão amostral (ddof=1)."
        )
    features = np.asarray(x, dtype=float)
    labels = np.asarray(y, dtype=int)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("X e y têm números de linhas diferentes.")

    neighbor_grid = list(neighbors) if neighbors is not None else [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    started = time.perf_counter()
    splits = repeated_stratified_holdout(
        labels,
        n_rounds=n_rounds,
        base_seed=base_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    ratios = (train_ratio, val_ratio, test_ratio)
    per_round: list[dict] = []
    grid_rows: list[dict] = []

    print(
        f"Amostras: {labels.shape[0]} | normais: {int(np.sum(labels == 0))} | "
        f"anormais: {int(np.sum(labels == 1))}",
        flush=True,
    )
    if quick:
        print(
            f"Modo rápido: {n_rounds} rodadas e k em {neighbor_grid}.",
            flush=True,
        )

    for split in splits:
        _assert_known_vertebral_split(labels, split, ratios)
        round_started = time.perf_counter()
        print(
            f"Rodada {split['round']}/{n_rounds} | semente {split['seed']} | "
            f"treino {split['train_idx'].size}, validação {split['val_idx'].size}, "
            f"teste {split['test_idx'].size}",
            flush=True,
        )
        round_records, round_grid = run_round(features, labels, split, neighbor_grid)
        per_round.extend(round_records)
        grid_rows.extend(round_grid)
        for record in round_records:
            print(f"  {_progress_line(record)}", flush=True)
        print(f"  tempo da rodada: {time.perf_counter() - round_started:.1f} s", flush=True)

    elapsed = time.perf_counter() - started
    print(f"Tempo total: {elapsed:.1f} s", flush=True)
    return {
        "per_round": per_round,
        "grid_rows": grid_rows,
        "n_rounds": n_rounds,
        "base_seed": base_seed,
        "seeds": [split["seed"] for split in splits],
        "neighbors": neighbor_grid,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "quick": quick,
        "elapsed_seconds": elapsed,
    }


def run_round(
    x: np.ndarray,
    y: np.ndarray,
    split: dict,
    neighbors: list[int],
) -> tuple[list[dict], list[dict]]:
    """Roda baseline, KNN padrão, KNN ajustado e KNN sem normalização em uma divisão."""
    x_train, y_train = x[split["train_idx"]], y[split["train_idx"]]
    x_val, y_val = x[split["val_idx"]], y[split["val_idx"]]
    x_test, y_test = x[split["test_idx"]], y[split["test_idx"]]

    records: list[dict] = []

    baseline = make_baseline()
    baseline.fit(x_train, y_train)
    records.append(
        _record(
            "baseline",
            split,
            n_neighbors=None,
            weights="",
            metric="",
            scale=False,
            val_metrics=compute_metrics(y_val, baseline.predict(x_val)),
            test_metrics=compute_metrics(y_test, baseline.predict(x_test)),
            strategy="most_frequent",
        )
    )

    # Grade com normalização: a escolha usa só a validação. O teste de cada
    # combinação é medido depois, para a análise descritiva de sensibilidade.
    scaled = _fit_grid(x_train, y_train, x_val, y_val, neighbors, scale=True)
    best_scaled = select_best_configuration(scaled)
    default_scaled = find_default_configuration(scaled)
    grid_rows: list[dict] = []
    for item in scaled:
        test_metrics = compute_metrics(y_test, item["pipeline"].predict(x_test))
        item["test_metrics"] = test_metrics
        grid_rows.append(_grid_row(split, item, test_metrics))

    records.append(
        _record(
            "knn_default",
            split,
            default_scaled["n_neighbors"],
            default_scaled["weights"],
            default_scaled["metric"],
            True,
            default_scaled["val_metrics"],
            default_scaled["test_metrics"],
            strategy="",
        )
    )
    records.append(
        _record(
            "knn_tuned",
            split,
            best_scaled["n_neighbors"],
            best_scaled["weights"],
            best_scaled["metric"],
            True,
            best_scaled["val_metrics"],
            best_scaled["test_metrics"],
            strategy="",
        )
    )

    # Sem normalização: o teste é calculado uma vez, só na configuração escolhida.
    unscaled = _fit_grid(x_train, y_train, x_val, y_val, neighbors, scale=False)
    best_unscaled = select_best_configuration(unscaled)
    unscaled_test = compute_metrics(y_test, best_unscaled["pipeline"].predict(x_test))
    records.append(
        _record(
            "knn_tuned_no_scale",
            split,
            best_unscaled["n_neighbors"],
            best_unscaled["weights"],
            best_unscaled["metric"],
            False,
            best_unscaled["val_metrics"],
            unscaled_test,
            strategy="",
        )
    )
    return records, grid_rows


def _fit_grid(x_train, y_train, x_val, y_val, neighbors: list[int], scale: bool) -> list[dict]:
    """Treina cada configuração no treino e mede somente a validação."""
    fitted: list[dict] = []
    for n_neighbors, weights, metric in iter_grid(neighbors):
        pipeline = make_knn_pipeline(n_neighbors, weights, metric, scale=scale)
        pipeline.fit(x_train, y_train)
        val_metrics = compute_metrics(y_val, pipeline.predict(x_val))
        fitted.append(
            {
                "n_neighbors": n_neighbors,
                "weights": weights,
                "metric": metric,
                "scale": scale,
                "pipeline": pipeline,
                "val_f1_macro": val_metrics["f1_macro"],
                "val_recall_abnormal": val_metrics["recall_abnormal"],
                "val_metrics": val_metrics,
            }
        )
    return fitted


def _record(
    experiment: str,
    split: dict,
    n_neighbors: int | None,
    weights: str,
    metric: str,
    scale: bool,
    val_metrics: dict,
    test_metrics: dict,
    strategy: str,
) -> dict:
    cm = test_metrics["confusion_matrix"]
    return {
        "experiment": experiment,
        "round": split["round"],
        "seed": split["seed"],
        "n_neighbors": n_neighbors,
        "weights": weights,
        "metric": metric,
        "scale": scale,
        "strategy": strategy,
        "val_accuracy": val_metrics["accuracy"],
        "val_recall_abnormal": val_metrics["recall_abnormal"],
        "val_f1_macro": val_metrics["f1_macro"],
        "test_accuracy": test_metrics["accuracy"],
        "test_recall_abnormal": test_metrics["recall_abnormal"],
        "test_f1_macro": test_metrics["f1_macro"],
        "test_cm_00": int(cm[0, 0]),
        "test_cm_01": int(cm[0, 1]),
        "test_cm_10": int(cm[1, 0]),
        "test_cm_11": int(cm[1, 1]),
    }


def _grid_row(split: dict, item: dict, test_metrics: dict) -> dict:
    val_metrics = item["val_metrics"]
    return {
        "round": split["round"],
        "seed": split["seed"],
        "n_neighbors": item["n_neighbors"],
        "weights": item["weights"],
        "metric": item["metric"],
        "scale": True,
        "val_accuracy": val_metrics["accuracy"],
        "val_recall_abnormal": val_metrics["recall_abnormal"],
        "val_f1_macro": val_metrics["f1_macro"],
        "test_accuracy": test_metrics["accuracy"],
        "test_recall_abnormal": test_metrics["recall_abnormal"],
        "test_f1_macro": test_metrics["f1_macro"],
    }


def _progress_line(record: dict) -> str:
    base = (
        f"{record['experiment']}: acurácia teste={record['test_accuracy']:.4f}, "
        f"recall anormal={record['test_recall_abnormal']:.4f}, "
        f"F1 macro teste={record['test_f1_macro']:.4f}"
    )
    if record["experiment"] in {"knn_tuned", "knn_tuned_no_scale"}:
        chosen = config_to_string(
            int(record["n_neighbors"]),
            record["weights"],
            record["metric"],
            bool(record["scale"]),
        )
        return f"{base} | {chosen}"
    return base


def _assert_known_vertebral_split(y: np.ndarray, split: dict, ratios: tuple[float, float, float]) -> None:
    """Confere 60/20/20 e 126/42/42 quando a base e as proporções são as do protocolo."""
    if ratios != (0.6, 0.2, 0.2):
        return
    if y.shape[0] != 310 or int(np.sum(y == 0)) != 100 or int(np.sum(y == 1)) != 210:
        return
    expected = {
        "train_idx": (60, 126),
        "val_idx": (20, 42),
        "test_idx": (20, 42),
    }
    for key, (n_normal, n_abnormal) in expected.items():
        part = y[split[key]]
        got_normal = int(np.sum(part == 0))
        got_abnormal = int(np.sum(part == 1))
        if got_normal != n_normal or got_abnormal != n_abnormal:
            raise RuntimeError(
                f"Partição {key} da rodada {split['round']} saiu "
                f"{got_normal} normais e {got_abnormal} anormais; "
                f"esperado {n_normal} e {n_abnormal}."
            )
