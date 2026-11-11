"""Linha de comando da atividade de seleção de atributos.

Exemplo::

    python -m src.main_selecao --data-path vertebral_column.csv --n-rounds 5 --seed 42 --output-dir results_selecao
"""

from __future__ import annotations

import argparse
import sys

from src.data import load_dataset
from src.main import _configure_stdio
from src.selection_experiments import run_selection_experiments
from src.selection_reporting import SELECTION_LABELS, write_selection_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seleção de atributos (filtro, wrapper e embutido) com KNN na base "
            "Vertebral Column, com hold-out estratificado por classe."
        )
    )
    parser.add_argument("--data-path", required=True, help="Caminho de vertebral_column.csv ou column_2C.dat.")
    parser.add_argument("--n-rounds", type=int, default=5, help="Número de rodadas. Padrão: 5.")
    parser.add_argument("--seed", type=int, default=42, help="Semente base. A rodada i usa seed + i. Padrão: 42.")
    parser.add_argument("--output-dir", default="results_selecao", help="Pasta de saída. Padrão: results_selecao.")
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    if args.n_rounds < 2:
        print(
            "Erro: --n-rounds deve ser pelo menos 2, porque o desvio padrão amostral (ddof=1) exige isso.",
            file=sys.stderr,
        )
        return 1
    try:
        features, labels = load_dataset(args.data_path)
    except (OSError, ValueError) as exc:
        print(f"Erro ao ler a base: {exc}", file=sys.stderr)
        return 1

    result = run_selection_experiments(features, labels, n_rounds=args.n_rounds, base_seed=args.seed)
    summary = write_selection_results(result, args.output_dir, args.data_path)
    print(f"Resultados gravados em: {args.output_dir}", flush=True)
    print("Teste (média ± desvio): F1 macro | nº de atributos | tempo de seleção (ms)", flush=True)
    for _, row in summary.iterrows():
        print(
            f"  {SELECTION_LABELS[row['experiment']]}: {row['f1_macro_media_pm_desvio']} | "
            f"{row['n_features_media_pm_desvio']} | {row['select_seconds_media_pm_desvio']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
