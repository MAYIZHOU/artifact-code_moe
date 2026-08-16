from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_FONT_SIZE = 15
LEGEND_FONT = {"weight": "bold", "size": LABEL_FONT_SIZE}
EXPERT_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "lightgray",
        "grid.linestyle": "--",
        "font.family": "serif",
        "font.size": 12,
        "font.weight": "bold",
        "axes.labelweight": "bold",
    }
)


def _expert_key(value: str) -> Tuple[int, str]:
    suffix = str(value).lstrip("E")
    return (int(suffix), str(value)) if suffix.isdigit() else (10**9, str(value))


def _save(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(figure)


def _style_axis(axis: plt.Axes, ylabel: str, experts: Sequence[str]) -> None:
    positions = np.arange(len(experts))
    axis.set_xlabel("Expert", fontweight="bold", fontsize=LABEL_FONT_SIZE, labelpad=10)
    axis.set_ylabel(ylabel, fontweight="bold", fontsize=LABEL_FONT_SIZE, labelpad=10)
    axis.set_xticks(positions)
    axis.set_xticklabels(experts, fontsize=LABEL_FONT_SIZE, fontweight="bold")
    plt.setp(axis.get_yticklabels(), fontsize=LABEL_FONT_SIZE, fontweight="bold")


def _draw_behavior_axis(
    axis: plt.Axes,
    experts: Sequence[str],
    values: np.ndarray,
    errors: np.ndarray,
    ylabel: str,
) -> None:
    positions = np.arange(len(experts))
    bars = axis.bar(
        positions,
        values,
        yerr=errors,
        capsize=4,
        width=0.45,
        color=EXPERT_COLORS[: len(experts)],
        edgecolor="black",
        linewidth=1.2,
        alpha=0.9,
    )
    axis.bar_label(bars, fmt="%.3f", padding=4, fontweight="bold", fontsize=12)
    _style_axis(axis, ylabel, experts)
    maximum = float(np.nanmax(values + errors)) if len(values) else 0.0
    axis.set_ylim(0, maximum * 1.25 if maximum > 0 else 1.0)


def allocation_behavior_figures(frame: pd.DataFrame, output_dir: Path) -> None:
    frame = frame.sort_values("expert", key=lambda values: values.map(_expert_key))
    experts = frame["expert"].astype(str).tolist()
    panels = [
        ("participation_mean", "Avg Participation Weight", "weight"),
        ("unit_cost_mean", "Expected Cost", "cost"),
        ("revenue_share_mean", "Revenue Share", "share"),
    ]

    combined, axes = plt.subplots(1, len(panels), figsize=(18, 4.5))
    for axis, (column, ylabel, safe_name) in zip(axes, panels):
        values = frame[column].to_numpy(dtype=float)
        errors = frame[column.replace("_mean", "_std")].fillna(0.0).to_numpy(dtype=float)
        _draw_behavior_axis(axis, experts, values, errors, ylabel)

        panel, panel_axis = plt.subplots(figsize=(6, 4.5))
        _draw_behavior_axis(panel_axis, experts, values, errors, ylabel)
        panel.tight_layout()
        _save(panel, output_dir / f"figure4_allocation_behavior_{safe_name}")

    combined.tight_layout()
    _save(combined, output_dir / "figure4_allocation_behavior")


def allocation_comparison_figure(frame: pd.DataFrame, destination: Path) -> None:
    method_order = ["Participation-only", "Shapley", "Cost-adjusted"]
    display_names: Dict[str, str] = {
        "Participation-only": "Participation-only",
        "Shapley": "Shapley",
        "Cost-adjusted": "Cost-adjusted (Ours)",
    }
    colors = ["tab:red", "tab:green", "tab:blue"]
    experts = sorted(frame["expert"].astype(str).unique(), key=_expert_key)
    values = frame.pivot(index="expert", columns="method", values="share_mean").reindex(experts)
    errors = frame.pivot(index="expert", columns="method", values="share_std").reindex(experts)

    figure, axis = plt.subplots(figsize=(7, 5))
    positions = np.arange(len(experts))
    width = 0.25
    label_groups = []
    for index, method in enumerate(method_order):
        offset = (index - (len(method_order) - 1) / 2) * width
        bars = axis.bar(
            positions + offset,
            values[method].to_numpy(dtype=float),
            width,
            yerr=errors[method].fillna(0.0).to_numpy(dtype=float),
            capsize=3,
            label=display_names[method],
            color=colors[index],
            edgecolor="black",
            linewidth=1.2,
            alpha=1.0 if method == "Cost-adjusted" else 0.55,
        )
        labels = axis.bar_label(
            bars,
            fmt="%.2f",
            padding=4,
            fontweight="bold",
            fontsize=12,
            zorder=5,
        )
        for label in labels:
            label.set_path_effects(
                [path_effects.Stroke(linewidth=1.6, foreground="white"), path_effects.Normal()]
            )
        label_groups.append(labels)

    # Stagger only labels whose values are close enough to collide horizontally.
    for expert_index, expert in enumerate(experts):
        group_values = np.asarray(
            [values.loc[expert, method] for method in method_order], dtype=float
        )
        pairwise_distances = np.abs(group_values[:, None] - group_values[None, :])
        close_pair = np.any((pairwise_distances > 0) & (pairwise_distances < 0.03))
        if close_pair:
            for rank, method_index in enumerate(np.argsort(group_values)):
                label_groups[method_index][expert_index].set_position((0, 4 + 8 * rank))

    _style_axis(axis, "Revenue Share", experts)
    maximum = float(values[method_order].max().max())
    axis.set_ylim(0, maximum * 1.35 if maximum > 0 else 1.0)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=True,
        edgecolor="black",
        framealpha=1.0,
        prop={"weight": "bold", "size": 12},
    )
    figure.tight_layout()
    _save(figure, destination)


def lambda_sensitivity_figure(frame: pd.DataFrame, destination: Path) -> None:
    experts = sorted(frame["expert"].astype(str).unique(), key=_expert_key)
    lambda_values = sorted(frame["lambda"].astype(float).unique())
    values = frame.pivot(index="lambda", columns="expert", values="share_mean").reindex(lambda_values)
    errors = frame.pivot(index="lambda", columns="expert", values="share_std").reindex(lambda_values)
    markers = ["o", "s", "D", "^"]
    positions = np.arange(len(lambda_values))

    figure, axis = plt.subplots(figsize=(7, 5))
    for index, expert in enumerate(experts):
        axis.plot(
            positions,
            values[expert].to_numpy(dtype=float),
            label=expert,
            color=EXPERT_COLORS[index],
            marker=markers[index],
            linestyle="-",
            linewidth=2.5,
            markersize=8,
            alpha=0.9,
        )
        mean = values[expert].to_numpy(dtype=float)
        std = errors[expert].fillna(0.0).to_numpy(dtype=float)
        axis.fill_between(
            positions,
            mean - std,
            mean + std,
            color=EXPERT_COLORS[index],
            alpha=0.12,
            linewidth=0,
        )

    axis.set_xlabel(
        r"Sensitivity Parameter $\lambda$",
        fontweight="bold",
        fontsize=LABEL_FONT_SIZE,
        labelpad=10,
    )
    axis.set_ylabel("Revenue Share", fontweight="bold", fontsize=LABEL_FONT_SIZE, labelpad=10)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{value:g}" for value in lambda_values],
        fontsize=LABEL_FONT_SIZE,
        fontweight="bold",
    )
    plt.setp(axis.get_yticklabels(), fontsize=LABEL_FONT_SIZE, fontweight="bold")
    maximum = float(values[experts].max().max())
    axis.set_ylim(-0.05, maximum * 1.25 if maximum > 0 else 1.0)
    axis.legend(
        loc="best",
        ncol=2,
        frameon=True,
        edgecolor="black",
        framealpha=1.0,
        prop=LEGEND_FONT,
    )
    figure.tight_layout()
    _save(figure, destination)


def allocation_runtime_figure(frame: pd.DataFrame, destination: Path) -> None:
    method_order = ["Exact Shapley", "Cost-adjusted"]
    display_names = ["Exact\nShapley", "Cost-adjusted\n(Ours)"]
    colors = [
        matplotlib.colors.to_rgba("tab:green", alpha=0.55),
        matplotlib.colors.to_rgba("tab:blue", alpha=1.0),
    ]
    summary = (
        frame.groupby("method", as_index=True)["runtime_ms"]
        .agg(["mean", "std"])
        .reindex(method_order)
    )
    if summary["mean"].isnull().any():
        raise ValueError("Runtime results must contain Exact Shapley and Cost-adjusted runs.")
    summary["std"] = summary["std"].fillna(0.0)

    means = summary["mean"].to_numpy(dtype=float)
    standard_deviations = summary["std"].to_numpy(dtype=float)
    positions = np.arange(len(method_order))

    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(
        positions,
        means,
        width=0.52,
        yerr=standard_deviations,
        capsize=6,
        color=colors,
        edgecolor="black",
        linewidth=1.3,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "ecolor": "black"},
        zorder=2,
    )

    for method_index, method in enumerate(method_order):
        method_runs = frame.loc[frame["method"] == method, "runtime_ms"].to_numpy(dtype=float)
        jitter = np.linspace(-0.07, 0.07, len(method_runs)) if len(method_runs) > 1 else [0.0]
        axis.scatter(
            positions[method_index] + jitter,
            method_runs,
            s=36,
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            zorder=4,
        )

    for bar, mean, std in zip(bars, means, standard_deviations):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            (mean + std) * 1.16,
            rf"{mean:.4f} $\pm$ {std:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            path_effects=[
                path_effects.Stroke(linewidth=1.6, foreground="white"),
                path_effects.Normal(),
            ],
            zorder=5,
        )

    axis.set_yscale("log")
    axis.set_ylabel(
        "Allocation Runtime (ms, log scale)",
        fontweight="bold",
        fontsize=LABEL_FONT_SIZE,
        labelpad=10,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(display_names, fontsize=LABEL_FONT_SIZE, fontweight="bold")
    plt.setp(axis.get_yticklabels(), fontsize=LABEL_FONT_SIZE, fontweight="bold")
    axis.set_ylim(means.min() / 2.5, (means + standard_deviations).max() * 3.0)
    axis.grid(True, which="major", axis="y")
    axis.grid(False, axis="x")

    runtime_by_seed = (
        frame.pivot(index="seed", columns="method", values="runtime_ms")
        .reindex(columns=method_order)
        .sort_index()
    )
    speedups = runtime_by_seed["Exact Shapley"] / runtime_by_seed["Cost-adjusted"]
    speedup_mean = float(speedups.mean())
    speedup_std = float(speedups.std()) if len(speedups) > 1 else 0.0
    speedup_range = float(speedups.max() - speedups.min())
    speedup_padding = max(speedup_range * 0.35, speedup_mean * 0.015)

    inset = axis.inset_axes([0.42, 0.32, 0.55, 0.46])
    inset.axhspan(
        speedup_mean - speedup_std,
        speedup_mean + speedup_std,
        color="tab:blue",
        alpha=0.12,
        zorder=0,
    )
    inset.axhline(speedup_mean, color="tab:blue", linestyle="--", linewidth=1.5, zorder=1)
    inset.plot(
        speedups.index.to_numpy(dtype=int),
        speedups.to_numpy(dtype=float),
        color="tab:red",
        marker="o",
        markersize=8,
        linewidth=2.6,
        markeredgecolor="black",
        zorder=3,
    )
    for seed, speedup in speedups.items():
        inset.text(
            int(seed),
            float(speedup) + speedup_padding * 0.10,
            f"{speedup:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    inset.text(
        0.97,
        0.06,
        f"mean={speedup_mean:.1f}x",
        transform=inset.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "black"},
    )
    inset.set_title("Per-seed Speedup", fontsize=14, fontweight="bold", pad=6)
    inset.set_xlabel("Seed", fontsize=12, fontweight="bold", labelpad=1)
    inset.set_ylabel("Speedup (x)", fontsize=12, fontweight="bold", labelpad=4)
    inset.set_xticks(speedups.index.to_numpy(dtype=int))
    inset.set_ylim(
        float(speedups.min()) - speedup_padding,
        float(speedups.max()) + speedup_padding,
    )
    inset.tick_params(axis="both", labelsize=11, width=1.3, length=4)
    inset.grid(True, linestyle="--", color="lightgray", linewidth=0.8)
    for spine in inset.spines.values():
        spine.set_linewidth(1.0)

    figure.tight_layout()
    _save(figure, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-style Figures 4-6 from allocation summaries."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "raw" / "breast_cancer_figures_4_6_hybrid",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures" / "breast_cancer_figures_4_6_hybrid",
    )
    parser.add_argument("--skip-beta", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    behavior = pd.read_csv(arguments.input_dir / "allocation_behavior_summary.csv")
    comparison = pd.read_csv(arguments.input_dir / "allocation_comparison_summary.csv")
    sensitivity = pd.read_csv(arguments.input_dir / "lambda_sensitivity_summary.csv")

    allocation_behavior_figures(behavior, arguments.output_dir)
    allocation_comparison_figure(comparison, arguments.output_dir / "figure5_allocation_comparison")
    lambda_sensitivity_figure(sensitivity, arguments.output_dir / "figure6_lambda_sensitivity")
    runtime_path = arguments.input_dir / "allocation_runtime_runs.csv"
    if runtime_path.exists():
        allocation_runtime_figure(
            pd.read_csv(runtime_path), arguments.output_dir / "figure7_allocation_runtime"
        )
    print(f"Wrote paper-style Figures 4-7 to {arguments.output_dir}")


if __name__ == "__main__":
    main()
