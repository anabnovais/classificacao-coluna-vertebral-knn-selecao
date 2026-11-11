"""Tabelas, figuras e ``config_selecao.json`` da atividade de seleção de atributos.

Mesmo estilo do ``reporting.py``: agregação ``media``, ``desvio`` (ddof=1) e
``media_mais_desvio`` das métricas de **teste** entre as rodadas.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import FEATURE_COLUMNS
from src.feature_selection import CV_FOLDS, RF_TREES
from src.metrics import aggregate_values, format_mean_pm
from src.models import DEFAULT_METRIC, DEFAULT_N_NEIGHBORS, DEFAULT_WEIGHTS
from src.reporting import library_versions, row_normalize, summed_confusion
from src.selection_experiments import SELECTION_EXPERIMENTS

SELECTION_LABELS = {
    "reference": "Referência (todos)",
    "filter": "Filtro (ANOVA F)",
    "wrapper": "Wrapper (forward)",
    "embedded": "Embutido (Random Forest)",
}
SELECTION_METRICS = ["test_accuracy", "test_recall_abnormal", "test_f1_macro"]
OTHER_COLUMNS = ["n_features", "select_seconds", "fit_seconds"]


def write_selection_results(result: dict, output_dir: str | Path, data_path: str | Path) -> pd.DataFrame:
    """Grava CSVs, markdown, figuras e config. Devolve a tabela resumo do teste."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_round = pd.DataFrame(result["per_round"])
    per_round.to_csv(out / "per_round_selecao.csv", index=False)

    summary = build_selection_summary(per_round)
    summary.to_csv(out / "summary_selecao.csv", index=False)

    frequency = feature_frequency(per_round)
    frequency.to_csv(out / "frequencia_atributos.csv")

    write_selection_markdown(out / "summary_selecao.md", summary, per_round, frequency, result)
    plot_selection_boxplot(out / "boxplot_f1_selecao.png", per_round, result["n_rounds"])
    plot_feature_frequency(out / "frequencia_atributos.png", frequency, result["n_rounds"])
    write_selection_config(out / "config_selecao.json", result, data_path)
    return summary


def build_selection_summary(per_round: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por experimento: métricas de teste, número de atributos e tempos."""
    rows: list[dict] = []
    for experiment in SELECTION_EXPERIMENTS:
        group = per_round.loc[per_round["experiment"] == experiment]
        if group.empty:
            raise RuntimeError(f"Não há linhas do experimento {experiment}.")
        row: dict = {"experiment": experiment, "split": "test", "n_rounds": int(group.shape[0])}
        for column in SELECTION_METRICS + OTHER_COLUMNS:
            agg = aggregate_values(group[column].tolist())
            short = column.removeprefix("test_")
            row[f"{short}_media"] = agg["media"]
            row[f"{short}_desvio"] = agg["desvio"]
            row[f"{short}_media_mais_desvio"] = agg["media_mais_desvio"]
            digits = 4
            scale = 1.0
            if column.endswith("_seconds"):  # tempos em milissegundos no texto
                scale, digits = 1000.0, 1
            elif column == "n_features":
                digits = 2
            row[f"{short}_media_pm_desvio"] = format_mean_pm(
                agg["media"] * scale, agg["desvio"] * scale, digits
            )
        rows.append(row)
    return pd.DataFrame(rows)


def feature_frequency(per_round: pd.DataFrame) -> pd.DataFrame:
    """Em quantas rodadas cada atributo foi escolhido por cada experimento."""
    table = pd.DataFrame(0, index=FEATURE_COLUMNS, columns=SELECTION_EXPERIMENTS, dtype=int)
    for _, row in per_round.iterrows():
        for name in str(row["features"]).split("|"):
            table.loc[name, row["experiment"]] += 1
    return table


def write_selection_markdown(
    path: Path,
    summary: pd.DataFrame,
    per_round: pd.DataFrame,
    frequency: pd.DataFrame,
    result: dict,
) -> None:
    n_rounds = result["n_rounds"]
    lines = [
        "# Seleção de atributos: resumo do teste",
        "",
        "Classificador único: KNN padrão "
        f"(k={DEFAULT_N_NEIGHBORS}, pesos {DEFAULT_WEIGHTS}, distância {DEFAULT_METRIC}) "
        "com `StandardScaler` dentro do `Pipeline`. A seleção usa só o treino; "
        "cada modelo é avaliado no teste uma única vez por rodada.",
        "",
        f"Agregação entre {n_rounds} rodadas: `media ± desvio` (desvio amostral, `ddof=1`). "
        "Colunas `media`, `desvio` e `media_mais_desvio` estão em `summary_selecao.csv`.",
        "",
        "| Experimento | Acurácia | Recall anormal | F1 macro | Nº de atributos | Tempo de seleção (ms) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        label = SELECTION_LABELS[row["experiment"]]
        lines.append(
            f"| {label} | {row['accuracy_media_pm_desvio']} | "
            f"{row['recall_abnormal_media_pm_desvio']} | {row['f1_macro_media_pm_desvio']} | "
            f"{row['n_features_media_pm_desvio']} | {row['select_seconds_media_pm_desvio']} |"
        )

    lines.extend(
        [
            "",
            "O tempo de seleção é só o da busca (a referência não seleciona, tempo 0). "
            "O treino do KNN está em `fit_seconds` de `per_round_selecao.csv`.",
            "",
            "## Atributos escolhidos por rodada",
            "",
            "| Experimento | Rodada | Semente | Nº | Atributos |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    chosen = per_round.loc[per_round["experiment"] != "reference"].copy()
    chosen["ordem"] = chosen["experiment"].map({"filter": 0, "wrapper": 1, "embedded": 2})
    chosen = chosen.sort_values(["ordem", "round"])
    for _, row in chosen.iterrows():
        lines.append(
            f"| {SELECTION_LABELS[row['experiment']]} | {int(row['round'])} | "
            f"{int(row['seed'])} | {int(row['n_features'])} | {row['features'].replace('|', ', ')} |"
        )

    lines.extend(
        [
            "",
            f"## Frequência de cada atributo (em {n_rounds} rodadas)",
            "",
            "| Atributo | " + " | ".join(SELECTION_LABELS[e] for e in SELECTION_EXPERIMENTS) + " |",
            "| --- | " + " | ".join("---" for _ in SELECTION_EXPERIMENTS) + " |",
        ]
    )
    for name, row in frequency.iterrows():
        lines.append(f"| `{name}` | " + " | ".join(str(int(row[e])) for e in SELECTION_EXPERIMENTS) + " |")

    lines.extend(["", "## Matrizes de confusão somadas (teste)", "",
                  "Soma das predições de teste das rodadas; ordem `[normal, anormal]`. "
                  "Não é contagem de pacientes distintos.", ""])
    for experiment in SELECTION_EXPERIMENTS:
        cm = summed_confusion(per_round, experiment)
        lines.extend([f"**{SELECTION_LABELS[experiment]}**", "", "```", str(cm), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_selection_boxplot(path: Path, per_round: pd.DataFrame, n_rounds: int) -> None:
    """Boxplot do F1 macro no teste por experimento, com um ponto por rodada."""
    series = [
        per_round.loc[per_round["experiment"] == name, "test_f1_macro"].to_numpy(dtype=float)
        for name in SELECTION_EXPERIMENTS
    ]
    labels = [SELECTION_LABELS[name].replace(" (", "\n(") for name in SELECTION_EXPERIMENTS]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(series, tick_labels=labels, showfliers=False)
    for position, values in enumerate(series, start=1):
        offsets = np.linspace(-0.18, 0.18, values.shape[0])
        ax.scatter(np.full(values.shape[0], position) + offsets, values, color="black", s=24, zorder=3)
    ax.set_title(f"F1 macro no teste por método de seleção ({n_rounds} rodadas)")
    ax.set_ylabel("F1 macro (teste)")
    ax.set_xlabel("Experimento")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_frequency(path: Path, frequency: pd.DataFrame, n_rounds: int) -> None:
    """Barras agrupadas: quantas rodadas cada método escolheu cada atributo."""
    methods = [e for e in SELECTION_EXPERIMENTS if e != "reference"]
    x = np.arange(len(FEATURE_COLUMNS))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, method in enumerate(methods):
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, frequency[method].to_numpy(), width,
               label=SELECTION_LABELS[method])
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=20, ha="right")
    ax.set_ylabel(f"Rodadas em que foi escolhido (de {n_rounds})")
    ax.set_yticks(range(0, n_rounds + 1))
    ax.set_title("Atributos escolhidos por cada método")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_selection_config(path: Path, result: dict, data_path: str | Path) -> None:
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
        "classifier": {
            "n_neighbors": DEFAULT_N_NEIGHBORS,
            "weights": DEFAULT_WEIGHTS,
            "metric": DEFAULT_METRIC,
            "scale": True,
        },
        "selection": {
            "uses": "somente o conjunto de treino",
            "cv_folds": CV_FOLDS,
            "filter": "SelectKBest(f_classif); k por F1 macro em CV estratificada no treino",
            "wrapper": "forward selection com o próprio KNN; tamanho por F1 macro em CV estratificada no treino",
            "embedded": f"RandomForest({RF_TREES} árvores) + SelectFromModel(threshold='mean')",
            "tie_break": "menor número de atributos",
        },
        "positive_class": 1,
        "positive_class_name": "anormal",
        "aggregation": {
            "media": "(a_1 + ... + a_R) / R",
            "desvio": "desvio padrão amostral com ddof=1",
            "media_mais_desvio": "media + desvio",
            "split": "test",
        },
        "elapsed_seconds": result["elapsed_seconds"],
        "versions": library_versions(),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
