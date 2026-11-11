"""Testes da agregação, da regra de seleção e do ajuste do scaler."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.metrics import aggregate_values
from src.models import make_knn_pipeline, select_best_configuration


def test_aggregate_known_values():
    result = aggregate_values([0.8, 0.9, 1.0])
    assert result["media"] == pytest.approx(0.9)
    assert result["desvio"] == pytest.approx(0.1)
    assert result["media_mais_desvio"] == pytest.approx(1.0)
    # O desvio populacional (ddof=0) deste exemplo não é 0.1.
    population = float(np.std([0.8, 0.9, 1.0], ddof=0))
    assert result["desvio"] != pytest.approx(population)


def test_aggregate_requires_two_values():
    with pytest.raises(ValueError):
        aggregate_values([0.5])


def test_selection_tie_on_f1_prefers_abnormal_recall():
    candidates = [
        {
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "euclidean",
            "val_f1_macro": 0.8,
            "val_recall_abnormal": 0.70,
        },
        {
            "n_neighbors": 9,
            "weights": "distance",
            "metric": "manhattan",
            "val_f1_macro": 0.8,
            "val_recall_abnormal": 0.90,
        },
    ]
    chosen = select_best_configuration(candidates)
    assert chosen["n_neighbors"] == 9
    assert chosen["val_recall_abnormal"] == pytest.approx(0.90)


def test_selection_full_tie_prefers_smaller_k():
    candidates = [
        {
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "euclidean",
            "val_f1_macro": 0.8,
            "val_recall_abnormal": 0.9,
        },
        {
            "n_neighbors": 1,
            "weights": "distance",
            "metric": "manhattan",
            "val_f1_macro": 0.8,
            "val_recall_abnormal": 0.9,
        },
    ]
    chosen = select_best_configuration(candidates)
    assert chosen["n_neighbors"] == 1


def test_selection_then_prefers_uniform_and_euclidean():
    same_score = {"val_f1_macro": 0.8, "val_recall_abnormal": 0.9, "n_neighbors": 5}
    by_weight = select_best_configuration(
        [
            {**same_score, "weights": "distance", "metric": "euclidean"},
            {**same_score, "weights": "uniform", "metric": "manhattan"},
        ]
    )
    assert by_weight["weights"] == "uniform"

    by_metric = select_best_configuration(
        [
            {**same_score, "weights": "uniform", "metric": "manhattan"},
            {**same_score, "weights": "uniform", "metric": "euclidean"},
        ]
    )
    assert by_metric["metric"] == "euclidean"


def test_selection_ignores_test_metrics():
    candidates = [
        {
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "euclidean",
            "val_f1_macro": 0.70,
            "val_recall_abnormal": 0.60,
            "test_f1_macro": 0.99,
        },
        {
            "n_neighbors": 11,
            "weights": "distance",
            "metric": "manhattan",
            "val_f1_macro": 0.80,
            "val_recall_abnormal": 0.60,
            "test_f1_macro": 0.10,
        },
    ]
    chosen = select_best_configuration(candidates)
    assert chosen["n_neighbors"] == 11


def test_scaler_is_fit_only_on_train():
    rng = np.random.default_rng(0)
    x_train = rng.normal(loc=5.0, scale=2.0, size=(40, 6))
    y_train = np.array([0] * 20 + [1] * 20)
    x_val = rng.normal(loc=-3.0, scale=1.0, size=(15, 6))

    pipeline = make_knn_pipeline(n_neighbors=3, weights="uniform", metric="euclidean", scale=True)
    pipeline.fit(x_train, y_train)
    # Prever a validação não pode reajustar o scaler.
    pipeline.predict(x_val)

    scaler = pipeline.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert int(scaler.n_samples_seen_) == len(x_train)
    np.testing.assert_allclose(scaler.mean_, x_train.mean(axis=0))
    np.testing.assert_allclose(scaler.scale_, x_train.std(axis=0, ddof=0))

    combined = np.vstack([x_train, x_val])
    assert not np.allclose(scaler.mean_, combined.mean(axis=0))
