"""Testes do hold-out estratificado por classe."""

from pathlib import Path

import numpy as np
import pytest

from src.data import encode_label, load_dataset
from src.splitter import class_split_sizes, repeated_stratified_holdout, stratified_holdout_indices

ROOT = Path(__file__).resolve().parents[1]


def _counts(y, idx):
    part = y[idx]
    return int(np.sum(part == 0)), int(np.sum(part == 1))


def _assert_disjoint_and_complete(n, train_idx, val_idx, test_idx):
    train, val, test = set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(n))
    assert len(train) + len(val) + len(test) == n


def test_class_split_sizes_examples():
    assert class_split_sizes(90) == (54, 18, 18)
    assert class_split_sizes(10) == (6, 2, 2)
    assert class_split_sizes(100) == (60, 20, 20)
    assert class_split_sizes(210) == (126, 42, 42)


def test_holdout_90_10():
    y = np.array([0] * 90 + [1] * 10)
    train_idx, val_idx, test_idx = stratified_holdout_indices(y, seed=42)
    assert _counts(y, train_idx) == (54, 6)
    assert _counts(y, val_idx) == (18, 2)
    assert _counts(y, test_idx) == (18, 2)
    assert len(train_idx) == 60
    assert len(val_idx) == 20
    assert len(test_idx) == 20
    _assert_disjoint_and_complete(len(y), train_idx, val_idx, test_idx)
    original = (y == 1).mean()
    for idx in (train_idx, val_idx, test_idx):
        assert (y[idx] == 1).mean() == pytest.approx(original)


def test_holdout_real_class_sizes():
    y = np.array([0] * 100 + [1] * 210)
    train_idx, val_idx, test_idx = stratified_holdout_indices(y, seed=42)
    assert _counts(y, train_idx) == (60, 126)
    assert _counts(y, val_idx) == (20, 42)
    assert _counts(y, test_idx) == (20, 42)
    assert (len(train_idx), len(val_idx), len(test_idx)) == (186, 62, 62)
    _assert_disjoint_and_complete(len(y), train_idx, val_idx, test_idx)
    original = (y == 1).mean()
    for idx in (train_idx, val_idx, test_idx):
        assert (y[idx] == 1).mean() == pytest.approx(original)


def test_same_seed_is_deterministic_and_other_seed_differs():
    y = np.array([0] * 100 + [1] * 210)
    first = stratified_holdout_indices(y, seed=42)
    second = stratified_holdout_indices(y, seed=42)
    third = stratified_holdout_indices(y, seed=43)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
    assert not all(np.array_equal(left, right) for left, right in zip(first, third))


def test_five_rounds_use_seeds_42_to_46_and_differ():
    y = np.array([0] * 100 + [1] * 210)
    splits = repeated_stratified_holdout(y, n_rounds=5, base_seed=42)
    assert len(splits) == 5
    assert [split["seed"] for split in splits] == [42, 43, 44, 45, 46]
    assert [split["round"] for split in splits] == [1, 2, 3, 4, 5]
    trains = [tuple(split["train_idx"].tolist()) for split in splits]
    assert len(set(trains)) == 5
    for split in splits:
        _assert_disjoint_and_complete(len(y), split["train_idx"], split["val_idx"], split["test_idx"])
        assert _counts(y, split["train_idx"]) == (60, 126)
        assert _counts(y, split["val_idx"]) == (20, 42)
        assert _counts(y, split["test_idx"]) == (20, 42)


def test_small_class_raises_clear_error():
    y = np.array([0] * 90 + [1] * 2)
    with pytest.raises(ValueError, match="insuficiente"):
        stratified_holdout_indices(y, seed=1)


def test_encode_binary_labels():
    assert encode_label("NO") == 0
    assert encode_label("Normal") == 0
    assert encode_label("AB") == 1
    assert encode_label("Hernia") == 1
    assert encode_label("Spondylolisthesis") == 1
    with pytest.raises(ValueError, match="desconhecido"):
        encode_label("talvez")


def test_real_csv_has_expected_binary_counts():
    features, labels = load_dataset(ROOT / "vertebral_column.csv")
    assert features.shape == (310, 6)
    assert int(np.sum(labels == 0)) == 100
    assert int(np.sum(labels == 1)) == 210
    # Primeira linha do CSV é Hernia, portanto anormal.
    assert labels[0] == 1
    assert features[0, 0] == pytest.approx(63.0278175)
    assert features[0, 5] == pytest.approx(-0.254399986)


def test_dat_format_ab_no(tmp_path):
    lines = ["1 2 3 4 5 6 NO"] * 100 + ["1 2 3 4 5 6 AB"] * 210
    path = tmp_path / "column_2C.dat"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    features, labels = load_dataset(path)
    assert features.shape == (310, 6)
    assert int(np.sum(labels == 0)) == 100
    assert int(np.sum(labels == 1)) == 210
    assert labels[0] == 0
    assert labels[-1] == 1
