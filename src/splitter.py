"""Hold-out estratificado por classe, repetido com sementes.

A divisão é feita dentro de cada classe e só depois as partes são unidas.
Não usa ``train_test_split``, ``KFold`` nem ``StratifiedKFold``.

Arredondamento, para uma classe com ``n`` amostras e proporções
``train_ratio``, ``val_ratio``, ``test_ratio`` (padrão 0.6, 0.2, 0.2):

- ``n_train = round(train_ratio * n)``
- ``n_val = round(val_ratio * n)``
- ``n_test = n - n_train - n_val``

``round`` é o da biblioteca padrão do Python 3 (meio para o par mais próximo).
Nos tamanhos desta base o produto cai em inteiro, então o meio do arredondamento
não entra: 100 → 60/20/20 e 210 → 126/42/42. Cada partição precisa ter pelo
menos uma amostra da classe; se o tamanho não permitir, a função lança erro.

Cada rodada ``i`` (começando em 0) usa a semente ``base_seed + i``. O embaralhamento
de cada classe usa um único ``numpy.random.Generator`` criado com essa semente.
As classes são percorridas em ordem crescente de rótulo, para a sequência do
gerador ser estável.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.2


def class_split_sizes(
    n: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> tuple[int, int, int]:
    """Devolve ``(n_train, n_val, n_test)`` para uma classe de tamanho ``n``.

    Exemplos com as proporções padrão: 90 → 54/18/18; 10 → 6/2/2;
    100 → 60/20/20; 210 → 126/42/42.
    """
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    if n < 0:
        raise ValueError(f"O tamanho da classe não pode ser negativo (recebido {n}).")
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_test = int(n - n_train - n_val)
    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"A classe tem {n} amostras, insuficiente para garantir pelo menos "
            f"1 amostra em treino, validação e teste "
            f"(obtido {n_train}/{n_val}/{n_test} com proporções "
            f"{train_ratio}/{val_ratio}/{test_ratio})."
        )
    return n_train, n_val, n_test


def stratified_holdout_indices(
    y: np.ndarray,
    seed: int,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide os índices posicionais de ``y`` em treino, validação e teste.

    O gerador é ``numpy.random.default_rng(seed)``, um ``numpy.random.Generator``.
    A mesma semente reproduz os mesmos índices. Os índices são posicionais
    (a linha 0 é o primeiro elemento de ``y``).
    """
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y deve ser um vetor de rótulos, um por amostra.")
    n_samples = int(labels.shape[0])

    # Os tamanhos são resolvidos antes de consumir o gerador. Uma classe pequena
    # falha com a mesma mensagem, qualquer que seja a semente.
    plans: list[tuple[object, np.ndarray, int, int, int]] = []
    for cls in sorted(np.unique(labels).tolist()):
        class_idx = np.flatnonzero(labels == cls)
        try:
            n_train, n_val, n_test = class_split_sizes(
                int(class_idx.shape[0]), train_ratio, val_ratio, test_ratio
            )
        except ValueError as exc:
            raise ValueError(f"Classe {cls!r}: {exc}") from exc
        plans.append((cls, class_idx, n_train, n_val, n_test))

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for _cls, class_idx, n_train, n_val, _n_test in plans:
        shuffled = rng.permutation(class_idx)
        train_parts.append(shuffled[:n_train])
        val_parts.append(shuffled[n_train : n_train + n_val])
        test_parts.append(shuffled[n_train + n_val :])

    train_idx = np.concatenate(train_parts).astype(int, copy=False)
    val_idx = np.concatenate(val_parts).astype(int, copy=False)
    test_idx = np.concatenate(test_parts).astype(int, copy=False)
    ensure_partition(n_samples, train_idx, val_idx, test_idx)
    return train_idx, val_idx, test_idx


def repeated_stratified_holdout(
    y: np.ndarray,
    n_rounds: int = 5,
    base_seed: int = 42,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> list[dict]:
    """Repete o hold-out ``n_rounds`` vezes.

    A rodada ``i`` (0 ≤ i < n_rounds) usa a semente ``base_seed + i``.
    Com o padrão ``n_rounds=5`` e ``base_seed=42``, as sementes são
    42, 43, 44, 45 e 46. Cada rodada embaralha e redivide as classes.
    """
    if n_rounds < 1:
        raise ValueError("n_rounds deve ser pelo menos 1.")
    splits: list[dict] = []
    for i in range(n_rounds):
        seed = int(base_seed) + i
        train_idx, val_idx, test_idx = stratified_holdout_indices(
            y,
            seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        splits.append(
            {
                "round": i + 1,
                "seed": seed,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }
        )
    return splits


def ensure_partition(
    n_samples: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Garante que as três partições não se sobrepõem e cobrem todas as amostras."""
    train = np.asarray(train_idx)
    val = np.asarray(val_idx)
    test = np.asarray(test_idx)
    for name, idx in (("treino", train), ("validação", val), ("teste", test)):
        if idx.size != np.unique(idx).size:
            raise RuntimeError(f"Há índices repetidos na partição de {name}.")
    train_set = set(train.tolist())
    val_set = set(val.tolist())
    test_set = set(test.tolist())
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Treino, validação e teste se sobrepõem.")
    union = train_set | val_set | test_set
    if union != set(range(n_samples)):
        raise RuntimeError(
            "A união de treino, validação e teste não é o conjunto original "
            f"({len(union)} índices distintos; esperado {n_samples})."
        )


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("As proporções de treino, validação e teste devem ser positivas.")
    total = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"As proporções devem somar 1; a soma obtida foi {total}.")
