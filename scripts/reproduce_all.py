from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from moe_market.data.registry import IMAGE_DATASETS, TABULAR_DATASETS


def run(command: List[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the unified paper protocol.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the one-seed smoke protocol for every selected dataset.",
    )
    parser.add_argument(
        "--stage",
        default="table",
        choices=["table", "figures", "all"],
        help="The full Figure 3-7 sweep is intentionally opt-in because it is expensive.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(TABULAR_DATASETS) + list(IMAGE_DATASETS),
        help="Run a subset; by default all registered paper datasets are used.",
    )
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args()
    python = sys.executable
    datasets = arguments.datasets or list(TABULAR_DATASETS) + list(IMAGE_DATASETS)
    for dataset_name in datasets:
        command = [
            python,
            "experiments/run_dataset.py",
            "--dataset",
            dataset_name,
            "--device",
            arguments.device,
            "--stage",
            arguments.stage,
        ]
        if arguments.quick:
            command.append("--quick")
        if arguments.no_plots:
            command.append("--no-plots")
        run(command)


if __name__ == "__main__":
    main()
