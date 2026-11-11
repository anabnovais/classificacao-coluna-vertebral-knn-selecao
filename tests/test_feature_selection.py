"""Testes da seleção de atributos: seleção só com treino, tamanhos e determinismo."""

import numpy as np
import pytest

from src.data import load_dataset
from src.feature_selection import (
    select_embedded,
    select_filter,
    select_wrapper,
)
from src.selection_experiments import run_selection_experiments
from src.splitter import stratified_holdout_indices


def _toy_data(seed: int = 0):
    """Duas colunas informativas e quatro de ruído."""
    rng = np.random.default_rng(seed)
    n = 200
    y = np.repeat([0, 1], n // 2)
    informative = rng.normal(size=(n, 2)) + 3.0 * y[:, None]
    noise = rng.normal(size=(n, 4))
    return np.hstack([informative, noise]), y


@pytest.mark.parametrize("selector", [select_filter, select_wrapper, select_embedded])
def test_selectors_find_informative_columns(selector):
    x, y = _toy_data()
    columns = selector(x, y, 42)
    assert columns == sorted(columns)
    assert 1 <= len(columns) < x.shape[1]
    assert {0, 1} & set(columns)


@pytest.mark.parametrize("selector", [select_filter, select_wrapper, select_embedded])
def test_selectors_are_deterministic(selector):
    x, y = _toy_data()
    assert selector(x, y, 7) == selector(x, y, 7)


def test_selection_ignores_rows_outside_train():
    """Mudar as linhas de teste não pode alterar a seleção: ela só enxerga o treino."""
    csv = "vertebral_column.csv"
    x, y = load_dataset(csv)
    train, _val, test = stratified_holdout_indices(y, 42)
    base = select_filter(x[train], y[train], 42), select_wrapper(x[train], y[train], 42)
    x_changed = x.copy()
    x_changed[test] = x_changed[test] * 100 + 5
    again = select_filter(x_changed[train], y[train], 42), select_wrapper(x_changed[train], y[train], 42)
    assert base == again


def test_full_run_has_four_experiments_and_valid_counts():
    x, y = load_dataset("vertebral_column.csv")
    result = run_selection_experiments(x, y, n_rounds=2)
    records = result["per_round"]
    assert {r["experiment"] for r in records} == {"reference", "filter", "wrapper", "embedded"}
    for record in records:
        assert 1 <= record["n_features"] <= 6
        assert record["select_seconds"] >= 0
        if record["experiment"] == "reference":
            assert record["n_features"] == 6
        total = sum(record[f"test_cm_{i}{j}"] for i in (0, 1) for j in (0, 1))
        assert total == 62
