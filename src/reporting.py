"""Tabelas, figuras e ``config.json``.

As figuras usam o estilo padrão do matplotlib, com títulos e rótulos em português.
A matriz somada conta predições de teste ao longo das rodadas. Como o hold-out
é refeito, o mesmo paciente pode cair no teste mais de uma vez: a soma não é
uma contagem de pacientes distintos.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import aggregate_values, format_mean_pm
from src.models import DEFAULT_METRIC, DEFAULT_N_NEIGHBORS, DEFAULT_WEIGHTS, METRICS_GRID, WEIGHTS_GRID

EXPERIMENT_ORDER = ["baseline", "knn_default", "knn_tuned", "knn_tuned_no_scale"]
EXPERIMENT_LABELS = {
    "baseline": "Linha de base",
    "knn_default": "KNN padrão",
    "knn_tuned": "KNN ajustado",
    "knn_tuned_no_scale": "KNN sem normalização",
}
TEST_METRICS = ["test_accuracy", "test_recall_abnormal", "test_f1_macro"]
GRID_METRICS = [
    "val_accuracy",
    "val_recall_abnormal",
    "val_f1_macro",
    "test_accuracy",
    "test_recall_abnormal",
    "test_f1_macro",
]
CURVE_STYLE = {
    ("uniform", "euclidean"): ("o", "-"),
    ("uniform", "manhattan"): ("s", "--"),
    ("distance", "euclidean"): ("^", "-."),
    ("distance", "manhattan"): ("D", ":"),
}
CURVE_LABELS = {
    ("uniform", "euclidean"): "uniforme, euclidiana",
    ("uniform", "manhattan"): "uniforme, Manhattan",
    ("distance", "euclidean"): "distância, euclidiana",
    ("distance", "manhattan"): "distância, Manhattan",
}


def write_all_results(result: dict, output_dir: str | Path, data_path: str | Path) -> pd.DataFrame:
    """Grava CSVs, figuras e ``config.json``. Devolve a tabela resumo do teste."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_round = pd.DataFrame(result["per_round"])
    per_round["n_neighbors"] = per_round["n_neighbors"].astype("Int64")
    per_round = per_round[
        [
            "experiment",
            "round",
            "seed",
            "n_neighbors",
            "weights",
            "metric",
            "scale",
            "strategy",
            "val_accuracy",
            "val_recall_abnormal",
            "val_f1_macro",
            "test_accuracy",
            "test_recall_abnormal",
            "test_f1_macro",
            "test_cm_00",
            "test_cm_01",
            "test_cm_10",
            "test_cm_11",
        ]
    ]
    per_round.to_csv(out / "per_round.csv", index=False)

    summary = build_summary(per_round)
    summary.to_csv(out / "summary.csv", index=False)

    grid = build_grid_summary(result["grid_rows"])
    grid.to_csv(out / "grid_sensitivity.csv", index=False)

    cm = summed_confusion(per_round, "knn_tuned")
    write_summary_markdown(out / "summary.md", summary, per_round, cm, result)
    plot_confusion_matrix(out / "confusion_matrix_knn.png", cm, result["n_rounds"])
    plot_f1_boxplot(out / "boxplot_f1_macro.png", per_round, result["n_rounds"])
    plot_k_sensitivity(out / "k_sensitivity.png", grid, result["n_rounds"])
    write_config(out / "config.json", result, data_path)
    return summary


def build_summary(per_round: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por variante, com média, desvio e média + desvio no teste."""
    rows: list[dict] = []
    for experiment in EXPERIMENT_ORDER:
        group = per_round.loc[per_round["experiment"] == experiment]
        if group.empty:
            raise RuntimeError(f"Não há linhas do experimento {experiment}.")
        row: dict = {"experiment": experiment, "split": "test", "n_rounds": int(group.shape[0])}
        for column in TEST_METRICS:
            agg = aggregate_values(group[column].tolist())
            short = column.removeprefix("test_")
            row[f"{short}_media"] = agg["media"]
            row[f"{short}_desvio"] = agg["desvio"]
            row[f"{short}_media_mais_desvio"] = agg["media_mais_desvio"]
            row[f"{short}_media_pm_desvio"] = format_mean_pm(agg["media"], agg["desvio"])
        rows.append(row)
    return pd.DataFrame(rows)


def build_grid_summary(grid_rows: list[dict]) -> pd.DataFrame:
    """Média, desvio e média + desvio de cada combinação da grade, entre as rodadas."""
    frame = pd.DataFrame(grid_rows)
    grouped = frame.groupby(["n_neighbors", "weights", "metric", "scale"], sort=True)
    rows: list[dict] = []
    for key, group in grouped:
        n_neighbors, weights, metric, scale = key
        row: dict = {
            "n_neighbors": int(n_neighbors),
            "weights": weights,
            "metric": metric,
            "scale": bool(scale),
            "n_rounds": int(group.shape[0]),
        }
        for column in GRID_METRICS:
            agg = aggregate_values(group[column].tolist())
            row[f"{column}_media"] = agg["media"]
            row[f"{column}_desvio"] = agg["desvio"]
            row[f"{column}_media_mais_desvio"] = agg["media_mais_desvio"]
        rows.append(row)
    return pd.DataFrame(rows)


def summed_confusion(per_round: pd.DataFrame, experiment: str) -> np.ndarray:
    """Soma as matrizes de confusão do teste de um experimento."""
    group = per_round.loc[per_round["experiment"] == experiment]
    cm = np.zeros((2, 2), dtype=int)
    cm[0, 0] = int(group["test_cm_00"].sum())
    cm[0, 1] = int(group["test_cm_01"].sum())
    cm[1, 0] = int(group["test_cm_10"].sum())
    cm[1, 1] = int(group["test_cm_11"].sum())
    return cm


def row_normalize(cm: np.ndarray) -> np.ndarray:
    """Divide cada linha pelo seu total. Linha vazia permanece zerada."""
    totals = cm.sum(axis=1, keepdims=True).astype(float)
    normalized = np.zeros(cm.shape, dtype=float)
    np.divide(cm, totals, out=normalized, where=totals != 0)
    return normalized


def write_summary_markdown(
    path: Path,
    summary: pd.DataFrame,
    per_round: pd.DataFrame,
    cm: np.ndarray,
    result: dict,
) -> None:
    """Tabela legível para o texto, mais os k escolhidos e a matriz somada."""
    lines = [
        "# Resumo do teste",
        "",
        "Agregação das métricas de **teste** entre as rodadas. "
        "A validação escolhe a configuração; o teste não entra nessa escolha.",
        "",
        "Fórmulas, com `a_1 ... a_R` os valores da métrica no teste:",
        "",
        "- `media = (a_1 + ... + a_R) / R`",
        "- `desvio` = desvio padrão amostral (`ddof=1`, divisor `R - 1`). "
        f"Nesta execução, R = {result['n_rounds']}"
        + (" e o divisor é 4." if result["n_rounds"] == 5 else "."),
        "- `media_mais_desvio = media + desvio`",
        "",
        "Com poucas rodadas, média e desvio são estimativas ruidosas. "
        "`media_mais_desvio` pode passar de 1, porque não é um intervalo de confiança.",
        "",
        "| Experimento | Acurácia | Recall anormal | F1 macro |",
        "| --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        label = EXPERIMENT_LABELS[row["experiment"]]
        lines.append(
            f"| {label} | {row['accuracy_media_pm_desvio']} | "
            f"{row['recall_abnormal_media_pm_desvio']} | {row['f1_macro_media_pm_desvio']} |"
        )

    lines.extend(
        [
            "",
            "Colunas numéricas correspondentes estão em `summary.csv`: "
            "`media`, `desvio` e `media_mais_desvio` de cada métrica.",
            "",
            "## Configurações escolhidas do KNN ajustado",
            "",
            "Regra: maior F1 macro na validação; empate no recall da classe anormal; "
            "novo empate no menor k, depois peso uniforme, depois distância euclidiana.",
            "",
            "| Rodada | Semente | k | Pesos | Distância | F1 macro (validação) | F1 macro (teste) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    tuned = per_round.loc[per_round["experiment"] == "knn_tuned"].sort_values("round")
    for _, row in tuned.iterrows():
        lines.append(
            f"| {int(row['round'])} | {int(row['seed'])} | {int(row['n_neighbors'])} | "
            f"{row['weights']} | {row['metric']} | {row['val_f1_macro']:.4f} | "
            f"{row['test_f1_macro']:.4f} |"
        )

    counts = tuned["n_neighbors"].value_counts().sort_index()
    lines.extend(["", "### Frequência de k", "", "| k | Rodadas |", "| --- | --- |"])
    for k, count in counts.items():
        lines.append(f"| {int(k)} | {int(count)} |")

    normalized = row_normalize(cm)
    lines.extend(
        [
            "",
            "## Matriz de confusão somada (KNN ajustado, teste)",
            "",
            "Cada célula soma as predições de teste das rodadas. "
            "Não é uma contagem de pacientes distintos: o mesmo registro pode "
            "entrar no teste em mais de uma rodada.",
            "",
            "Ordem dos rótulos: linha e coluna `[normal, anormal]`.",
            "",
            "```",
            str(cm),
            "```",
            "",
            "Normalizada por linha (proporção dentro da classe verdadeira):",
            "",
            "```",
            np.array2string(normalized, precision=4, suppress_small=True),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_confusion_matrix(path: Path, cm: np.ndarray, n_rounds: int) -> None:
    """Desenha a matriz somada e a mesma matriz normalizada por linha."""
    normalized = row_normalize(cm)
    labels = ["Normal", "Anormal"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _draw_matrix(
        axes[0],
        cm.astype(float),
        labels,
        f"Matriz somada ({n_rounds} rodadas de teste)",
        "{:.0f}",
    )
    _draw_matrix(
        axes[1],
        normalized,
        labels,
        "Normalizada por linha",
        "{:.2f}",
    )
    fig.suptitle("KNN ajustado — matriz de confusão do teste")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _draw_matrix(ax, matrix: np.ndarray, labels: list[str], title: str, fmt: str) -> None:
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    threshold = float(np.max(matrix)) / 2 if np.max(matrix) > 0 else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = "white" if value > threshold else "black"
            ax.text(j, i, fmt.format(value), ha="center", va="center", color=color)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_f1_boxplot(path: Path, per_round: pd.DataFrame, n_rounds: int) -> None:
    """Boxplot do F1 macro no teste, com um ponto por rodada."""
    series = [
        per_round.loc[per_round["experiment"] == name, "test_f1_macro"].to_numpy(dtype=float)
        for name in EXPERIMENT_ORDER
    ]
    labels = [EXPERIMENT_LABELS[name] for name in EXPERIMENT_ORDER]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(series, tick_labels=labels, showfliers=False)
    for position, values in enumerate(series, start=1):
        # Espaçamento fixo no eixo x para os pontos da mesma rodada não se cobrirem
        # quando a métrica é constante, como na linha de base.
        offsets = np.linspace(-0.18, 0.18, values.shape[0])
        ax.scatter(np.full(values.shape[0], position) + offsets, values, color="black", s=24, zorder=3)
    ax.set_title(f"F1 macro no teste por variante ({n_rounds} rodadas)")
    ax.set_ylabel("F1 macro (teste)")
    ax.set_xlabel("Variante")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_k_sensitivity(path: Path, grid: pd.DataFrame, n_rounds: int) -> None:
    """Curva do F1 macro médio em função de k.

    O painel da validação é o que a seleção enxerga. O painel do teste é
    descritivo e não foi usado para escolher a configuração.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    panels = (
        (axes[0], "val_f1_macro_media", "Validação (usada na seleção)"),
        (axes[1], "test_f1_macro_media", "Teste (somente descritivo)"),
    )
    for ax, column, title in panels:
        for weights in WEIGHTS_GRID:
            for metric in METRICS_GRID:
                key = (weights, metric)
                subset = grid.loc[(grid["weights"] == weights) & (grid["metric"] == metric)]
                subset = subset.sort_values("n_neighbors")
                marker, linestyle = CURVE_STYLE[key]
                ax.plot(
                    subset["n_neighbors"],
                    subset[column],
                    marker=marker,
                    linestyle=linestyle,
                    label=CURVE_LABELS[key],
                )
        ax.set_title(title)
        ax.set_xlabel("k (n_neighbors)")
        ax.set_ylabel("F1 macro (média entre rodadas)")
        ax.set_xticks(sorted(grid["n_neighbors"].unique()))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Sensibilidade do KNN a k, peso e distância ({n_rounds} rodadas)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_config(path: Path, result: dict, data_path: str | Path) -> None:
    """Registra sementes, proporções e versões usadas nesta execução."""
    data_file = Path(data_path)
    payload = {
        "data_path": str(data_file),
        "data_sha256": hashlib.sha256(data_file.read_bytes()).hexdigest(),
        "n_rounds": result["n_rounds"],
        "base_seed": result["base_seed"],
        "seeds": result["seeds"],
        "train_ratio": result["train_ratio"],
        "val_ratio": result["val_ratio"],
        "test_ratio": result["test_ratio"],
        "quick": result["quick"],
        "n_neighbors_grid": result["neighbors"],
        "weights_grid": list(WEIGHTS_GRID),
        "metric_grid": list(METRICS_GRID),
        "default_knn": {
            "n_neighbors": DEFAULT_N_NEIGHBORS,
            "weights": DEFAULT_WEIGHTS,
            "metric": DEFAULT_METRIC,
            "scale": True,
        },
        "positive_class": 1,
        "positive_class_name": "anormal",
        "selection_rule": [
            "maior F1 macro na validação",
            "empate: maior recall da classe anormal",
            "novo empate: menor n_neighbors, depois weights=uniform, depois metric=euclidean",
        ],
        "aggregation": {
            "media": "(a_1 + ... + a_R) / R",
            "desvio": "desvio padrão amostral com ddof=1 (divisor R-1; com R=5, divisor 4)",
            "media_mais_desvio": "media + desvio",
            "split": "test",
        },
        "elapsed_seconds": result["elapsed_seconds"],
        "versions": library_versions(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def library_versions() -> dict:
    """Versões do interpretador e das bibliotecas numéricas."""
    import matplotlib as mpl
    import numpy
    import pandas
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": mpl.__version__,
    }
