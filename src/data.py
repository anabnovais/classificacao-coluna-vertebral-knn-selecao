"""Leitura e validação da base Vertebral Column (tarefa binária).

O repositório traz ``vertebral_column.csv``, exportado do UCI pelo
``ucimlrepo`` (id 212), com três rótulos: Normal, Hernia e Spondylolisthesis.
A tarefa deste trabalho é a binária, a mesma de ``column_2C.dat``. Na definição
da UCI, Hérnia e Espondilolistese formam a classe anormal. Conferimos o arquivo
oficial: a mesma ordem de linhas, os mesmos rótulos binários e os mesmos
números arredondados a duas casas. O CSV guarda mais casas decimais, e é essa
versão que o experimento usa quando o caminho aponta para ele.

Codificação fixa: anormal = 1 (classe positiva), normal = 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "pelvic_incidence",
    "pelvic_tilt",
    "lumbar_lordosis_angle",
    "sacral_slope",
    "pelvic_radius",
    "degree_spondylolisthesis",
]

# Rótulos aceitos. Hernia e Spondylolisthesis (e as siglas DH/SL do arquivo
# de três classes) entram em anormal, como na tarefa binária da UCI.
NORMAL_LABELS = {"NO", "Normal", "normal"}
ABNORMAL_LABELS = {
    "AB",
    "Abnormal",
    "abnormal",
    "Hernia",
    "Spondylolisthesis",
    "DH",
    "SL",
}

EXPECTED_N_SAMPLES = 310
EXPECTED_N_NORMAL = 100
EXPECTED_N_ABNORMAL = 210


def encode_label(raw: object) -> int:
    """Converte um rótulo bruto em 0 (normal) ou 1 (anormal).

    Lança ``ValueError`` se o texto não pertence à tarefa binária.
    """
    text = str(raw).strip()
    if text in NORMAL_LABELS:
        return 0
    if text in ABNORMAL_LABELS:
        return 1
    raise ValueError(
        f"Rótulo desconhecido: {raw!r}. Esperado normal (NO/Normal) ou anormal "
        "(AB/Abnormal/Hernia/Spondylolisthesis)."
    )


def validate_binary_counts(y: np.ndarray) -> None:
    """Interrompe o experimento se a base não tiver 310 linhas, 100 normais e 210 anormais."""
    y = np.asarray(y)
    n_samples = int(y.shape[0])
    n_normal = int(np.sum(y == 0))
    n_abnormal = int(np.sum(y == 1))
    if (
        n_samples != EXPECTED_N_SAMPLES
        or n_normal != EXPECTED_N_NORMAL
        or n_abnormal != EXPECTED_N_ABNORMAL
        or n_normal + n_abnormal != n_samples
    ):
        raise ValueError(
            "A base não confere com a tarefa binária da UCI (column_2C): "
            f"esperado {EXPECTED_N_SAMPLES} linhas, {EXPECTED_N_NORMAL} normais (0) e "
            f"{EXPECTED_N_ABNORMAL} anormais (1); obtido {n_samples} linhas, "
            f"{n_normal} normais e {n_abnormal} anormais. O experimento foi interrompido."
        )


def load_dataset(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Lê a base e devolve ``X`` (n, 6) e ``y`` (n,) com rótulos 0/1.

    Aceita o CSV com cabeçalho (``vertebral_column.csv``) e o ``column_2C.dat``
    separado por espaços, sem cabeçalho, com rótulo ``AB`` ou ``NO``.
    Não normaliza os atributos: isso fica a cargo do ``Pipeline``, depois da divisão.
    """
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        features, labels = _read_csv(data_path)
    elif suffix == ".dat":
        features, labels = _read_dat(data_path)
    else:
        raise ValueError(
            f"Extensão não suportada ({suffix or 'sem extensão'}). "
            "Use vertebral_column.csv ou column_2C.dat."
        )

    x = np.asarray(features, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Esperadas {len(FEATURE_COLUMNS)} características numéricas; obtido shape {x.shape}."
        )
    if not np.isfinite(x).all():
        raise ValueError("Há valores ausentes ou não numéricos nas características.")
    validate_binary_counts(y)
    return x, y


def _label_column(frame: pd.DataFrame) -> str:
    if "class" in frame.columns:
        return "class"
    extra = [column for column in frame.columns if column not in FEATURE_COLUMNS]
    if len(extra) != 1:
        raise ValueError(
            "Não foi possível identificar a coluna de rótulo. "
            f"Colunas encontradas: {list(frame.columns)}."
        )
    return extra[0]


def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, encoding="utf-8")
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Colunas ausentes no CSV: {missing}. Colunas encontradas: {list(frame.columns)}."
        )
    label_column = _label_column(frame)
    features = frame.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    labels = np.asarray([encode_label(value) for value in frame[label_column]], dtype=int)
    return features, labels


def _read_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Lê ``column_2C.dat``: 6 números e um rótulo, separados por espaço, sem cabeçalho."""
    features: list[list[float]] = []
    labels: list[int] = []
    text = path.read_text(encoding="utf-8")
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 7:
            raise ValueError(
                f"Linha {line_number} de {path.name} tem {len(parts)} campos; "
                "o formato esperado é 6 números e 1 rótulo, separados por espaço."
            )
        try:
            features.append([float(value) for value in parts[:6]])
        except ValueError as exc:
            raise ValueError(f"Valor numérico inválido na linha {line_number} de {path.name}.") from exc
        labels.append(encode_label(parts[6]))
    return np.asarray(features, dtype=float), np.asarray(labels, dtype=int)
