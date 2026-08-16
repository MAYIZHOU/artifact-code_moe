from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

from .costs import expert_cost_matrix, service_cost
from .metrics import classification_metrics, regression_metrics
from .models.moe import MoE
from .objectives import per_sample_predictive_loss, utility_from_loss


@dataclass
class CollectedPredictions:
    prediction: torch.Tensor
    target: torch.Tensor
    weights: torch.Tensor
    expert_predictions: torch.Tensor
    active_mask: torch.Tensor
    expert_costs: torch.Tensor


@torch.no_grad()
def collect_predictions(
    model: MoE,
    loader: Iterable,
    device: torch.device,
    base_costs: Sequence[float],
    input_scale: float,
) -> CollectedPredictions:
    model.eval()
    predictions, targets, weights, experts, masks, costs = [], [], [], [], [], []
    for inputs, target in loader:
        inputs, target = inputs.to(device), target.to(device)
        output = model(inputs)
        predictions.append(output.prediction.cpu())
        targets.append(target.cpu())
        weights.append(output.weights.cpu())
        experts.append(output.expert_predictions.cpu())
        masks.append(output.active_mask.cpu())
        costs.append(expert_cost_matrix(inputs, base_costs, input_scale).cpu())
    return CollectedPredictions(
        prediction=torch.cat(predictions),
        target=torch.cat(targets),
        weights=torch.cat(weights),
        expert_predictions=torch.cat(experts),
        active_mask=torch.cat(masks),
        expert_costs=torch.cat(costs),
    )


def _result(
    task: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    active_mask: torch.Tensor,
    expert_costs: torch.Tensor,
    cost_mode: str,
    alpha: float,
    evaluation_beta: float,
    loss_reference: float,
) -> Dict[str, float]:
    target_numpy = target.numpy().reshape(-1)
    prediction_numpy = prediction.numpy()
    metrics = (
        classification_metrics(prediction_numpy, target_numpy.astype(int))
        if task == "classification"
        else regression_metrics(prediction_numpy, target_numpy)
    )
    predictive_loss = per_sample_predictive_loss(task, prediction, target)
    utility = float(utility_from_loss(predictive_loss, alpha, loss_reference).item())
    expected_cost = float(service_cost(weights, expert_costs, active_mask, cost_mode).mean().item())
    metrics.update(
        {
            "utility": utility,
            "cost": expected_cost,
            "welfare": utility - float(evaluation_beta) * expected_cost,
        }
    )
    return metrics


def evaluate_all_methods(
    standard_model: MoE,
    market_model: MoE,
    validation_loader: Iterable,
    test_loader: Iterable,
    task: str,
    device: torch.device,
    objective_config: Dict,
    cost_config: Dict,
    best_expert_split: str = "validation",
    include_cost_aware_selection: bool = True,
) -> Dict[str, Dict[str, float]]:
    base_costs = cost_config["values"]
    input_scale = float(cost_config.get("input_scale", 0.0))
    standard = collect_predictions(standard_model, test_loader, device, base_costs, input_scale)
    market = collect_predictions(market_model, test_loader, device, base_costs, input_scale)
    selection = standard
    if best_expert_split == "validation":
        selection = collect_predictions(standard_model, validation_loader, device, base_costs, input_scale)
    elif best_expert_split != "test":
        raise ValueError("best_expert_split must be 'validation' or 'test'.")

    expert_losses = []
    for index in range(selection.expert_predictions.shape[1]):
        expert_losses.append(
            float(
                per_sample_predictive_loss(
                    task,
                    selection.expert_predictions[:, index],
                    selection.target,
                ).mean()
            )
        )
    evaluation_beta = float(objective_config.get("evaluation_beta", objective_config["beta"]))
    cost_aware_values = []
    for index, expert_loss in enumerate(expert_losses):
        expert_utility = float(
            utility_from_loss(
                per_sample_predictive_loss(
                    task,
                    selection.expert_predictions[:, index],
                    selection.target,
                ),
                objective_config["alpha"],
                objective_config["loss_reference"],
            )
        )
        expert_cost = float(selection.expert_costs[:, index].mean())
        cost_aware_values.append(expert_utility - evaluation_beta * expert_cost)
    cost_aware_index = int(np.argmax(cost_aware_values))
    n_samples, n_experts = standard.weights.shape
    uniform_weights = torch.full_like(standard.weights, 1.0 / n_experts)
    uniform_mask = torch.ones_like(standard.active_mask)
    uniform_prediction = standard.expert_predictions.mean(dim=1)
    cost_aware_weights = torch.zeros_like(standard.weights)
    cost_aware_weights[:, cost_aware_index] = 1.0
    cost_aware_mask = cost_aware_weights.bool()
    cost_aware_prediction = standard.expert_predictions[:, cost_aware_index]

    common = {
        "task": task,
        "target": standard.target,
        "expert_costs": standard.expert_costs,
        "cost_mode": cost_config.get("evaluation_mode", cost_config.get("mode", "weighted")),
        "alpha": objective_config["alpha"],
        "evaluation_beta": objective_config.get("evaluation_beta", objective_config["beta"]),
        "loss_reference": objective_config["loss_reference"],
    }
    results = {
        "Uniform Average": _result(
            prediction=uniform_prediction,
            weights=uniform_weights,
            active_mask=uniform_mask,
            **common,
        ),
        "Standard MoE": _result(
            prediction=standard.prediction,
            weights=standard.weights,
            active_mask=standard.active_mask,
            **common,
        ),
        "MoE Market": _result(
            prediction=market.prediction,
            target=market.target,
            weights=market.weights,
            active_mask=market.active_mask,
            expert_costs=market.expert_costs,
            task=task,
            cost_mode=cost_config.get("evaluation_mode", cost_config.get("mode", "weighted")),
            alpha=objective_config["alpha"],
            evaluation_beta=objective_config.get("evaluation_beta", objective_config["beta"]),
            loss_reference=objective_config["loss_reference"],
        ),
    }
    if include_cost_aware_selection:
        results["Welfare-aware Static Selection"] = _result(
            prediction=cost_aware_prediction,
            weights=cost_aware_weights,
            active_mask=cost_aware_mask,
            **common,
        )
        results["Welfare-aware Static Selection"]["selected_expert"] = cost_aware_index
    return results
