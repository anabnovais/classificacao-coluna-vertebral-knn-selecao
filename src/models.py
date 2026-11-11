"""KNN, pipelines e a regra fixa de seleção da grade.

O único classificador do experimento é ``KNeighborsClassifier``.
``DummyClassifier(strategy="most_frequent")`` entra só como linha de base trivial.
A busca da grade é um laço próprio: não usamos ``GridSearchCV``.
"""

from __future__ import annotations

from itertools import product

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_N_NEIGHBORS = 5
DEFAULT_WEIGHTS = "uniform"
DEFAULT_METRIC = "euclidean"

# k ímpar evita empate de votação na classificação binária com peso uniforme.
FULL_NEIGHBORS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
QUICK_NEIGHBORS = [1, 5, 11]
WEIGHTS_GRID = ["uniform", "distance"]
METRICS_GRID = ["euclidean", "manhattan"]

# Na desempate, o número maior é o preferido (depois do F1 e do recall).
WEIGHT_PRIORITY = {"uniform": 1, "distance": 0}
METRIC_PRIORITY = {"euclidean": 1, "manhattan": 0}


def make_knn_pipeline(
    n_neighbors: int,
    weights: str = DEFAULT_WEIGHTS,
    metric: str = DEFAULT_METRIC,
    scale: bool = True,
) -> Pipeline:
    """Monta o KNN. Com ``scale=True``, o ``StandardScaler`` é o primeiro passo.

    O scaler só vê o treino porque ``Pipeline.fit`` ajusta cada passo com o
    conjunto passado a ``fit``, e o experimento chama ``fit`` só com o treino.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
    if scale:
        return Pipeline([("scaler", StandardScaler()), ("knn", knn)])
    return Pipeline([("knn", knn)])


def make_baseline() -> DummyClassifier:
    """Linha de base que sempre prevê a classe mais frequente do treino."""
    return DummyClassifier(strategy="most_frequent")


def iter_grid(
    neighbors: list[int] | None = None,
    weights: list[str] | None = None,
    metrics: list[str] | None = None,
):
    """Gera as combinações ``(n_neighbors, weights, metric)`` da grade."""
    neighbor_values = FULL_NEIGHBORS if neighbors is None else list(neighbors)
    weight_values = WEIGHTS_GRID if weights is None else list(weights)
    metric_values = METRICS_GRID if metrics is None else list(metrics)
    yield from product(neighbor_values, weight_values, metric_values)


def select_best_configuration(candidates: list[dict]) -> dict:
    """Escolhe a configuração pela regra definida antes de olhar o teste.

    Ordem, só com métricas de validação:

    1. maior F1 macro;
    2. empate: maior recall da classe anormal;
    3. novo empate: configuração mais simples, nesta ordem:
       menor ``n_neighbors``, depois ``weights="uniform"``, depois ``metric="euclidean"``.

    Campos de teste, se existirem no dicionário, não entram na chave.
    """
    if not candidates:
        raise ValueError("A lista de candidatos está vazia.")
    return max(candidates, key=_selection_key)


def _selection_key(candidate: dict) -> tuple:
    weights = candidate["weights"]
    metric = candidate["metric"]
    if weights not in WEIGHT_PRIORITY:
        raise ValueError(f"Peso desconhecido na seleção: {weights!r}.")
    if metric not in METRIC_PRIORITY:
        raise ValueError(f"Métrica de distância desconhecida na seleção: {metric!r}.")
    return (
        float(candidate["val_f1_macro"]),
        float(candidate["val_recall_abnormal"]),
        -int(candidate["n_neighbors"]),
        WEIGHT_PRIORITY[weights],
        METRIC_PRIORITY[metric],
    )


def config_to_string(n_neighbors: int, weights: str, metric: str, scale: bool) -> str:
    """Descreve uma configuração de KNN em uma linha."""
    scale_text = "com StandardScaler" if scale else "sem StandardScaler"
    return f"n_neighbors={n_neighbors}, weights={weights}, metric={metric}, {scale_text}"


def find_default_configuration(candidates: list[dict]) -> dict:
    """Localiza k=5, peso uniforme e distância euclidiana dentro da grade já ajustada."""
    for candidate in candidates:
        if (
            int(candidate["n_neighbors"]) == DEFAULT_N_NEIGHBORS
            and candidate["weights"] == DEFAULT_WEIGHTS
            and candidate["metric"] == DEFAULT_METRIC
        ):
            return candidate
    raise RuntimeError(
        "A configuração padrão do KNN (k=5, uniform, euclidean) não está na grade."
    )
