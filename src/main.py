"""Linha de comando do experimento de KNN na coluna vertebral.

Exemplo::

    python -m src.main --data-path vertebral_column.csv --n-rounds 5 --seed 42 --output-dir results
"""

from __future__ import annotations

import argparse
import sys

from src.data import load_dataset
from src.experiments import run_experiments
from src.models import FULL_NEIGHBORS, QUICK_NEIGHBORS
from src.reporting import EXPERIMENT_LABELS, write_all_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classificação binária da coluna vertebral com KNN e "
            "hold-out estratificado por classe."
        )
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Caminho de vertebral_column.csv ou de column_2C.dat.",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=None,
        help="Número de rodadas. Padrão: 5. Com --quick e sem este argumento, o padrão passa a 2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente base. A rodada i (a partir de 0) usa seed + i. Padrão: 42.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Pasta de saída. Padrão: results.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Grade reduzida, k em {1, 5, 11}. Se --n-rounds for omitido, usa 2 rodadas.",
    )
    return parser


def _configure_stdio() -> None:
    """Pede UTF-8 no terminal para os acentos do progresso aparecerem no Windows."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = build_parser().parse_args(argv)
    if args.n_rounds is None:
        n_rounds = 2 if args.quick else 5
    else:
        n_rounds = args.n_rounds
    if n_rounds < 2:
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

    neighbors = list(QUICK_NEIGHBORS if args.quick else FULL_NEIGHBORS)
    result = run_experiments(
        features,
        labels,
        n_rounds=n_rounds,
        base_seed=args.seed,
        neighbors=neighbors,
        quick=args.quick,
    )
    summary = write_all_results(result, args.output_dir, args.data_path)
    print(f"Resultados gravados em: {args.output_dir}", flush=True)
    print("F1 macro no teste (média ± desvio):", flush=True)
    for _, row in summary.iterrows():
        label = EXPERIMENT_LABELS[row["experiment"]]
        print(f"  {label}: {row['f1_macro_media_pm_desvio']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
