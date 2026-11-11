"""Seleção de atributos: um método de cada família (filtro, wrapper, embutido).

O classificador é sempre o mesmo KNN padrão do projeto (k=5, pesos uniformes,
distância euclidiana, ``StandardScaler`` dentro do ``Pipeline``). Assim, a
diferença entre os experimentos vem só do conjunto de atributos.

Todos os métodos recebem **apenas o treino**. O teste não aparece aqui.
Onde é preciso escolher quantos atributos manter, a escolha é feita por
validação cruzada estratificada dentro do treino (``StratifiedKFold``, com a
semente da rodada). O ``StandardScaler`` e, no filtro, o ``SelectKBest``
ficam dentro do ``Pipeline`` avaliado em cada dobra, para a dobra de
validação não influenciar o ranking nem a média e o desvio.

Famílias:

- filtro: ``SelectKBest`` com a estatística F da ANOVA (``f_classif``). O
  ranking não depende do classificador. O número ``k`` de atributos é o que dá
  o maior F1 macro na validação cruzada do treino.
- wrapper: seleção sequencial para frente (forward). Parte do conjunto vazio e
  adiciona, a cada passo, o atributo que mais aumenta o F1 macro do KNN na
  validação cruzada do treino. Entre os tamanhos 1..p, fica o de maior F1.
- embutido: ``RandomForestClassifier`` e ``SelectFromModel`` com limiar
  ``"mean"``. Fica quem tem importância (redução de impureza) maior ou igual
  à média. A seleção acontece durante o treino da floresta, sem validação
  cruzada e sem ajuste de limiar.

Em empate de F1 (filtro e wrapper), vence o conjunto com menos atributos. No
wrapper, empate entre candidatos de um passo vai para o menor índice de coluna.
Cada função devolve os índices das colunas escolhidas, em ordem crescente.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.models import (
    DEFAULT_METRIC,
    DEFAULT_N_NEIGHBORS,
    DEFAULT_WEIGHTS,
    make_knn_pipeline,
)

CV_FOLDS = 5
RF_TREES = 200

FILTER = "filter"
WRAPPER = "wrapper"
EMBEDDED = "embedded"
SELECTORS = [FILTER, WRAPPER, EMBEDDED]


def make_reference_knn() -> Pipeline:
    """KNN padrão com ``StandardScaler``: o mesmo classificador de todos os experimentos."""
    return make_knn_pipeline(
        DEFAULT_N_NEIGHBORS, DEFAULT_WEIGHTS, DEFAULT_METRIC, scale=True
    )


def _cv_f1_macro(estimator, x: np.ndarray, y: np.ndarray, seed: int, n_splits: int) -> float:
    """F1 macro médio na validação cruzada estratificada. As dobras só dependem da semente."""
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, x, y, cv=folds, scoring="f1_macro")
    return float(np.mean(scores))


def select_filter(
    x_train: np.ndarray, y_train: np.ndarray, seed: int, n_splits: int = CV_FOLDS
) -> list[int]:
    """Filtro: ``SelectKBest(f_classif)`` com ``k`` escolhido por validação cruzada no treino."""
    n_features = x_train.shape[1]
    best_k, best_score = 1, -np.inf
    for k in range(1, n_features + 1):
        knn = make_reference_knn()
        estimator = Pipeline(
            [("select", SelectKBest(f_classif, k=k)), *knn.steps]
        )
        score = _cv_f1_macro(estimator, x_train, y_train, seed, n_splits)
        # ">" estrito: em empate, fica o k menor, já avaliado.
        if score > best_score:
            best_k, best_score = k, score
    selector = SelectKBest(f_classif, k=best_k).fit(x_train, y_train)
    return np.flatnonzero(selector.get_support()).tolist()


def select_wrapper(
    x_train: np.ndarray, y_train: np.ndarray, seed: int, n_splits: int = CV_FOLDS
) -> list[int]:
    """Wrapper: seleção sequencial para frente guiada pelo F1 macro do próprio KNN."""
    n_features = x_train.shape[1]
    chosen: list[int] = []
    remaining = list(range(n_features))
    best_subset: list[int] = []
    best_score = -np.inf
    while remaining:
        step_best, step_score = None, -np.inf
        for column in remaining:  # índices crescentes: empate vai para o menor
            candidate = sorted(chosen + [column])
            score = _cv_f1_macro(
                make_reference_knn(), x_train[:, candidate], y_train, seed, n_splits
            )
            if score > step_score:
                step_best, step_score = column, score
        chosen.append(step_best)
        remaining.remove(step_best)
        # ">" estrito: tamanhos maiores só vencem se melhorarem o F1.
        if step_score > best_score:
            best_subset, best_score = sorted(chosen), step_score
    return best_subset


def select_embedded(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> list[int]:
    """Embutido: importâncias da Random Forest, com limiar igual à média."""
    forest = RandomForestClassifier(n_estimators=RF_TREES, random_state=seed)
    forest.fit(x_train, y_train)
    selector = SelectFromModel(forest, threshold="mean", prefit=True)
    return np.flatnonzero(selector.get_support()).tolist()


def run_selector(
    name: str, x_train: np.ndarray, y_train: np.ndarray, seed: int
) -> tuple[list[int], float]:
    """Executa um método e devolve ``(colunas, segundos)``. O tempo é só o da seleção."""
    started = time.perf_counter()
    if name == FILTER:
        columns = select_filter(x_train, y_train, seed)
    elif name == WRAPPER:
        columns = select_wrapper(x_train, y_train, seed)
    elif name == EMBEDDED:
        columns = select_embedded(x_train, y_train, seed)
    else:
        raise ValueError(f"Método de seleção desconhecido: {name!r}.")
    elapsed = time.perf_counter() - started
    if not columns:
        raise RuntimeError(f"O método {name} não selecionou nenhum atributo.")
    return columns, elapsed
