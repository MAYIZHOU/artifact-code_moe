from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from plot_beta_sensitivity import beta_sensitivity_figure
from make_paper_style_revenue_figures import (
    allocation_behavior_figures,
    allocation_comparison_figure,
    allocation_runtime_figure,
    lambda_sensitivity_figure,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-style Figures 3-7 from one dataset's saved summaries."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    arguments = parser.parse_args()
    input_dir = arguments.input_dir.resolve()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    beta_sensitivity_figure(
        pd.read_csv(input_dir / "beta_sensitivity_summary.csv"),
        output_dir / "figure3_beta_sensitivity",
    )
    allocation_behavior_figures(
        pd.read_csv(input_dir / "allocation_behavior_summary.csv"),
        output_dir,
    )
    allocation_comparison_figure(
        pd.read_csv(input_dir / "allocation_comparison_summary.csv"),
        output_dir / "figure5_allocation_comparison",
    )
    lambda_sensitivity_figure(
        pd.read_csv(input_dir / "lambda_sensitivity_summary.csv"),
        output_dir / "figure6_lambda_sensitivity",
    )
    allocation_runtime_figure(
        pd.read_csv(input_dir / "allocation_runtime_runs.csv"),
        output_dir / "figure7_allocation_runtime",
    )
    print(f"Wrote paper-style Figures 3-7 to {output_dir}")


if __name__ == "__main__":
    main()
