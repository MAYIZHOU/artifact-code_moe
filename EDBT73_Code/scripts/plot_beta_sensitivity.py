from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "Welfare-aware Static Selection",
    "Uniform Average",
    "Standard MoE",
    "MoE Market",
]
METHOD_STYLES: Dict[str, Dict[str, Any]] = {
    "Welfare-aware Static Selection": {
        "label": "WSS",
        "color": "tab:orange",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.5,
        "alpha": 0.6,
        "markersize": 7,
    },
    "Uniform Average": {
        "label": "Uniform Avg",
        "color": "tab:red",
        "marker": "v",
        "linestyle": "--",
        "linewidth": 1.5,
        "alpha": 0.6,
        "markersize": 7,
    },
    "Standard MoE": {
        "label": "Standard MoE",
        "color": "tab:green",
        "marker": "D",
        "linestyle": "--",
        "linewidth": 1.5,
        "alpha": 0.6,
        "markersize": 6,
    },
    "MoE Market": {
        "label": "MoE Market",
        "color": "tab:blue",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 3.0,
        "alpha": 1.0,
        "markersize": 10,
    },
}


def _save(figure: plt.Figure, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _panels(frame: pd.DataFrame) -> List[Tuple[str, str, str]]:
    panels = [
        ("training_welfare", "Welfare (Train)", "final_ablation_welfare_train"),
        ("deployment_welfare", "Welfare (Real)", "final_ablation_welfare_real"),
        ("utility", "Utility", "final_ablation_utility"),
        ("cost", "Cost", "final_ablation_cost"),
    ]
    if "accuracy_mean" in frame.columns:
        panels.extend(
            [
                ("ce", "CE (Loss)", "final_ablation_ce"),
                ("accuracy", "Accuracy", "final_ablation_acc"),
                ("f1", "Macro-F1", "final_ablation_f1"),
                ("auc", "AUC", "final_ablation_auc"),
            ]
        )
    elif "mse_mean" in frame.columns:
        panels.extend(
            [
                ("mse", "MSE", "final_ablation_mse"),
                ("rmse", "RMSE", "final_ablation_rmse"),
                ("mae", "MAE", "final_ablation_mae"),
            ]
        )
    else:
        raise ValueError("The beta summary contains neither classification nor regression metrics.")
    return panels


def beta_sensitivity_figure(frame: pd.DataFrame, destination: Path) -> None:
    panels = _panels(frame)
    betas = sorted(frame["training_beta"].astype(float).unique())
    output_dir = destination.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    def draw_panel(
        axis: plt.Axes,
        metric: str,
        ylabel: str,
        label_font_size: float,
        legend_font_size: float,
        show_legend: bool,
    ) -> None:
        lower_values: List[np.ndarray] = []
        upper_values: List[np.ndarray] = []
        for method in METHOD_ORDER:
            subset = frame[frame["method"] == method].sort_values("training_beta")
            if subset.empty:
                continue
            style = METHOD_STYLES[method]
            x = subset["training_beta"].to_numpy(dtype=float)
            mean = subset[f"{metric}_mean"].to_numpy(dtype=float)
            std_column = f"{metric}_std"
            std = (
                subset[std_column].fillna(0.0).to_numpy(dtype=float)
                if std_column in subset.columns
                else np.zeros_like(mean)
            )
            lower_values.append(mean - std)
            upper_values.append(mean + std)
            axis.plot(
                x,
                mean,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                alpha=style["alpha"],
            )
            if np.any(std > 0):
                axis.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    color=style["color"],
                    alpha=0.08,
                    linewidth=0,
                )

        axis.set_xlabel(r"Parameter $\beta$", fontweight="bold", fontsize=label_font_size, labelpad=10)
        axis.set_ylabel(ylabel, fontweight="bold", fontsize=label_font_size, labelpad=10)
        axis.set_xticks(betas)
        axis.set_xticklabels(
            [f"{value:g}" for value in betas],
            fontsize=label_font_size,
            fontweight="bold",
        )
        plt.setp(axis.get_yticklabels(), fontsize=label_font_size, fontweight="bold")
        minimum = float(min(values.min() for values in lower_values))
        maximum = float(max(values.max() for values in upper_values))
        difference = maximum - minimum
        padding = difference * 0.20 if difference > 0 else max(abs(maximum) * 0.10, 0.1)
        axis.set_ylim(minimum - padding, maximum + padding)
        axis.set_axisbelow(True)
        axis.grid(True, color="lightgray", linestyle="--", linewidth=0.9)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.5)
        if show_legend:
            axis.legend(
                loc="best",
                ncol=2,
                frameon=True,
                edgecolor="black",
                framealpha=1.0,
                prop={"weight": "bold", "size": legend_font_size},
            )

    style = {
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.weight": "bold",
        "axes.labelweight": "bold",
    }
    with plt.rc_context(style):
        for panel_index, (metric, ylabel, filename) in enumerate(panels):
            figure, axis = plt.subplots(figsize=(7, 5))
            draw_panel(axis, metric, ylabel, 15, 15, panel_index == 0)
            figure.tight_layout()
            _save(figure, output_dir / filename)

        columns = 4
        rows = int(np.ceil(len(panels) / columns))
        preview, axes = plt.subplots(rows, columns, figsize=(18, 4.3 * rows), squeeze=False)
        for panel_index, (axis, (metric, ylabel, _)) in enumerate(zip(axes.flat, panels)):
            draw_panel(axis, metric, ylabel, 10, 7.5, panel_index == 0)
        for axis in axes.flat[len(panels) :]:
            axis.set_visible(False)
        preview.tight_layout()
        preview.savefig(destination.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(preview)
