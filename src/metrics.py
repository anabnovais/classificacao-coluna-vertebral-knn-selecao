"""Métricas da tarefa binária e agregação entre rodadas.

Em cada partição calculamos acurácia, recall da classe anormal (``pos_label=1``)
e F1 macro. A matriz de confusão usa a ordem de rótulos ``[normal=0, anormal=1]``.

Agregação de ``a_1, ..., a_R`` medidos no teste (a mesma fórmula vale para as
três métricas):

- ``media = (a_1 + ... + a_R) / R``
- ``desvio`` = desvio padrão amostral, com divisor ``R - 1`` (``ddof=1``).
  Com as 5 rodadas padrão, esse divisor é 4.
- ``media_mais_desvio = media + desvio``

``media_mais_desvio`` não é um intervalo de confiança. Como as métricas estão
entre 0 e 1, a soma pode passar de 1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcula acurácia, recall da classe anormal, F1 macro e a matriz de confusão."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall_abnormal": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm,
    }


def aggregate_values(values: list[float] | np.ndarray) -> dict:
    """Agrega uma métrica entre rodadas.

    Exige pelo menos dois valores, porque o desvio padrão amostral (``ddof=1``)
    divide por ``R - 1``. Com um único número o desvio não está definido.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(
            "A agregação com desvio padrão amostral (ddof=1) exige pelo menos 2 valores "
            f"(recebidos: {arr.size})."
        )
    n_rounds = int(arr.size)
    media = float(np.sum(arr) / n_rounds)
    # Divisor R - 1. No protocolo padrão, R = 5 e o divisor é 4.
    desvio = float(np.std(arr, ddof=1))
    return {
        "media": media,
        "desvio": desvio,
        "media_mais_desvio": media + desvio,
    }


def format_mean_pm(media: float, desvio: float, digits: int = 4) -> str:
    """Formata a média e o desvio como ``media ± desvio`` para as tabelas."""
    return f"{media:.{digits}f} ± {desvio:.{digits}f}"
