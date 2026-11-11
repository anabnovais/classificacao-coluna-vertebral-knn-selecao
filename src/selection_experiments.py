"""Protocolo da atividade de seleção de atributos.

Em cada rodada (mesmas divisões 60/20/20 do restante do projeto, mesmas sementes):

1. Dividir em treino, validação e teste, estratificando por classe.
2. Referência: KNN padrão com todos os atributos, treinado no treino.
3. Para cada família (filtro, wrapper, embutido): selecionar atributos **só com o
   treino**, treinar o mesmo KNN nas colunas escolhidas.
4. Avaliar cada modelo no teste **uma única vez**.

A validação não participa desta atividade: a seleção usa validação cruzada
dentro do treino, e nenhum modelo é re-treinado com a validação.
"""

from __future__ import annotations

import time

import numpy as np

from src.data import FEATURE_COLUMNS
from src.feature_selection import SELECTORS, make_reference_knn, run_selector
from src.metrics import compute_metrics
from src.splitter import repeated_stratified_holdout

REFERENCE = "reference"
SELECTION_EXPERIMENTS = [REFERENCE, *SELECTORS]


def run_selection_experiments(
    x: np.ndarray,
    y: np.ndarray,
    n_rounds: int = 5,
    base_seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> dict:
    """Roda referência, filtro, wrapper e embutido nas mesmas divisões."""
    if n_rounds < 2:
        raise ValueError(
            "n_rounds deve ser pelo menos 2 para o desvio padrão amostral (ddof=1)."
        )
    features = np.asarray(x, dtype=float)
    labels = np.asarray(y, dtype=int)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("X e y têm números de linhas diferentes.")

    started = time.perf_counter()
    splits = repeated_stratified_holdout(
        labels,
        n_rounds=n_rounds,
        base_seed=base_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    print(
        f"Amostras: {labels.shape[0]} | atributos: {features.shape[1]} | "
        f"normais: {int(np.sum(labels == 0))} | anormais: {int(np.sum(labels == 1))}",
        flush=True,
    )

    per_round: list[dict] = []
    for split in splits:
        print(
            f"Rodada {split['round']}/{n_rounds} | semente {split['seed']} | "
            f"treino {split['train_idx'].size}, teste {split['test_idx'].size}",
            flush=True,
        )
        records = run_selection_round(features, labels, split)
        per_round.extend(records)
        for record in records:
            print(f"  {_progress_line(record)}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"Tempo total: {elapsed:.1f} s", flush=True)
    return {
        "per_round": per_round,
        "n_rounds": n_rounds,
        "base_seed": base_seed,
        "seeds": [split["seed"] for split in splits],
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_features_total": int(features.shape[1]),
        "elapsed_seconds": elapsed,
    }


def run_selection_round(x: np.ndarray, y: np.ndarray, split: dict) -> list[dict]:
    """Uma rodada: seleciona no treino, treina e avalia no teste uma vez por modelo."""
    x_train, y_train = x[split["train_idx"]], y[split["train_idx"]]
    x_test, y_test = x[split["test_idx"]], y[split["test_idx"]]
    all_columns = list(range(x.shape[1]))

    records = [
        _fit_and_test(REFERENCE, split, all_columns, 0.0, x_train, y_train, x_test, y_test)
    ]
    for name in SELECTORS:
        columns, select_seconds = run_selector(name, x_train, y_train, split["seed"])
        records.append(
            _fit_and_test(name, split, columns, select_seconds, x_train, y_train, x_test, y_test)
        )
    return records


def _fit_and_test(
    experiment: str,
    split: dict,
    columns: list[int],
    select_seconds: float,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    model = make_reference_knn()
    fit_started = time.perf_counter()
    model.fit(x_train[:, columns], y_train)
    fit_seconds = time.perf_counter() - fit_started

    # Única avaliação no teste deste modelo.
    metrics = compute_metrics(y_test, model.predict(x_test[:, columns]))
    cm = metrics["confusion_matrix"]
    record = {
        "experiment": experiment,
        "round": split["round"],
        "seed": split["seed"],
        "n_features": len(columns),
        "features": "|".join(FEATURE_COLUMNS[c] for c in columns),
        "select_seconds": select_seconds,
        "fit_seconds": fit_seconds,
        "test_accuracy": metrics["accuracy"],
        "test_recall_abnormal": metrics["recall_abnormal"],
        "test_f1_macro": metrics["f1_macro"],
        "test_cm_00": int(cm[0, 0]),
        "test_cm_01": int(cm[0, 1]),
        "test_cm_10": int(cm[1, 0]),
        "test_cm_11": int(cm[1, 1]),
    }
    return record


def _progress_line(record: dict) -> str:
    return (
        f"{record['experiment']}: {record['n_features']} atributos "
        f"({record['features']}) | acurácia={record['test_accuracy']:.4f}, "
        f"recall anormal={record['test_recall_abnormal']:.4f}, "
        f"F1 macro={record['test_f1_macro']:.4f} | "
        f"seleção {record['select_seconds'] * 1000:.1f} ms"
    )
