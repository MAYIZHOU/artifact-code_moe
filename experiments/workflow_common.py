from __future__ import annotations

import copy
import time
from statistics import median
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from common import build_model

from moe_market.allocation import (
    allocation_shares,
    coalition_values,
    exact_shapley,
    payment_shares_from_shapley,
)
from moe_market.costs import service_cost
from moe_market.data.cleaning import make_provider_loaders
from moe_market.evaluation import collect_predictions
from moe_market.models.experts import fit_native_experts
from moe_market.models.moe import MoE
from moe_market.reproducibility import set_seed
from moe_market.training import copy_experts, train_experts_independently, train_moe


METHOD_ORDER = [
    "Welfare-aware Static Selection",
    "Uniform Average",
    "Standard MoE",
    "MoE Market",
]


def reset_loader(loader: Iterable, seed: int) -> None:
    generator = getattr(loader, "generator", None)
    if generator is not None:
        generator.manual_seed(seed)


def train_model(
    config: Dict[str, Any],
    input_shape: Any,
    output_dim: int,
    train_loader: Iterable,
    validation_loader: Iterable,
    seed: int,
    device: torch.device,
    objective_name: str,
    beta: float,
    expert_bank: Optional[MoE] = None,
) -> MoE:
    set_seed(seed)
    model = build_model(config, input_shape, output_dim)
    if expert_bank is None:
        reset_loader(train_loader, seed)
        fit_native_experts(model, train_loader, seed)
    else:
        copy_experts(expert_bank, model, freeze=True)
    objective = copy.deepcopy(config["objective"])
    objective["beta"] = float(beta)
    reset_loader(train_loader, seed)
    return train_moe(
        model,
        train_loader,
        validation_loader,
        config["task"],
        objective_name,
        device,
        config["training"],
        objective,
        config["cost"],
    )


def fit_expert_bank(
    config: Dict[str, Any],
    data: Any,
    seed: int,
    device: torch.device,
) -> Optional[MoE]:
    if config["model"].get("training_mode", "joint") != "independent_frozen":
        return None
    set_seed(seed)
    model_input = getattr(data, "input_dim", None)
    if model_input is None:
        model_input = data.input_shape
    model = build_model(config, model_input, data.num_outputs)
    reset_loader(data.train_loader, seed)
    fit_native_experts(model, data.train_loader, seed)
    expert_loaders = make_provider_loaders(
        data.train_dataset,
        len(model.experts),
        config["model"].get("provider_split", "full"),
        int(config["data"].get("batch_size", 128)),
        seed,
        int(config["data"].get("num_workers", 0)),
    )
    return train_experts_independently(
        model,
        expert_loaders,
        data.validation_loader,
        config["task"],
        device,
        int(config["training"].get("expert_fit_epochs", 10)),
        float(config["training"].get("learning_rate", 1e-3)),
        float(config["training"].get("weight_decay", 0.0)),
    )


def _time_call(function: Callable[[], Any], repeats: int) -> Tuple[Any, float]:
    durations: List[float] = []
    result: Any = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = function()
        durations.append((time.perf_counter() - start) * 1000.0)
    return result, float(median(durations))


def allocation_outputs(
    config: Dict[str, Any],
    model: MoE,
    test_loader: Iterable,
    seed: int,
    device: torch.device,
    dataset_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    collected = collect_predictions(
        model,
        test_loader,
        device,
        config["cost"]["values"],
        float(config["cost"].get("input_scale", 0.0)),
    )
    weights = collected.weights.numpy()
    costs = np.asarray(config["cost"]["values"], dtype=float)
    allocation = config["allocation"]
    discount = float(allocation["lambda"])

    proposed, proposed_runtime = _time_call(
        lambda: allocation_shares(weights, costs, discount),
        int(allocation.get("proposed_runtime_repeats", 1000)),
    )

    def shapley_computation() -> Tuple[np.ndarray, np.ndarray]:
        values = coalition_values(
            config["task"],
            collected.expert_predictions,
            collected.weights,
            collected.target,
            float(config["objective"]["alpha"]),
            float(config["objective"]["loss_reference"]),
        )
        shapley_values = exact_shapley(values, collected.weights.shape[1])
        return shapley_values, payment_shares_from_shapley(shapley_values)

    (shapley_values, shapley_shares), shapley_runtime = _time_call(
        shapley_computation,
        int(allocation.get("shapley_runtime_repeats", 3)),
    )
    proposed["Shapley"] = shapley_shares
    expected_cost = float(
        service_cost(
            collected.weights,
            collected.expert_costs,
            collected.active_mask,
            config["cost"].get("evaluation_mode", config["cost"].get("mode", "weighted")),
        ).mean()
    )
    residual = max(
        0.0,
        float(allocation["service_price"])
        - float(allocation["broker_fee"])
        - expected_cost,
    )
    participation = weights.mean(axis=0)
    score = participation * np.exp(-discount * costs)
    cost_adjusted_share = score / score.sum()

    behavior_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    for index in range(len(costs)):
        behavior_rows.append(
            {
                "dataset": dataset_name,
                "seed": seed,
                "expert": f"E{index + 1}",
                "participation": float(participation[index]),
                "unit_cost": float(costs[index]),
                "cost_adjusted_score": float(score[index]),
                "revenue_share": float(cost_adjusted_share[index]),
                "residual_revenue": residual,
                "expected_service_cost": expected_cost,
                "shapley_value": float(shapley_values[index]),
            }
        )
        for method, shares in proposed.items():
            comparison_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "expert": f"E{index + 1}",
                    "method": method,
                    "share": float(shares[index]),
                    "payment": float(residual * shares[index]),
                }
            )

    sensitivity_rows: List[Dict[str, Any]] = []
    for lambda_value in allocation["lambda_grid"]:
        shares = allocation_shares(weights, costs, float(lambda_value))["Cost-adjusted"]
        for index, share in enumerate(shares):
            sensitivity_rows.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "lambda": float(lambda_value),
                    "expert": f"E{index + 1}",
                    "share": float(share),
                    "payment": float(residual * share),
                }
            )

    runtime_rows = [
        {
            "dataset": dataset_name,
            "seed": seed,
            "method": "Cost-adjusted",
            "runtime_ms": proposed_runtime,
            "n_experts": len(costs),
        },
        {
            "dataset": dataset_name,
            "seed": seed,
            "method": "Exact Shapley",
            "runtime_ms": shapley_runtime,
            "n_experts": len(costs),
        },
    ]
    return behavior_rows, comparison_rows, sensitivity_rows, runtime_rows


def wide_summary(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    grouped = frame.groupby(list(group_columns), sort=False)[list(metrics)].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{statistic}" for metric, statistic in grouped.columns]
    return grouped.reset_index()


def write_outputs(output_dir: Path, name: str, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"{name}_runs.csv", index=False)
    return frame
