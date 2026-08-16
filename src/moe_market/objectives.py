from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def per_sample_predictive_loss(task: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if task == "classification":
        probabilities = prediction.clamp_min(1e-12)
        return F.nll_loss(probabilities.log(), target.long(), reduction="none")
    if task == "regression":
        return (prediction.reshape(-1) - target.reshape(-1)).pow(2)
    raise ValueError(f"Unsupported task: {task}")


def individual_expert_loss(task: str, raw_prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if task == "classification":
        return F.cross_entropy(raw_prediction, target.long())
    return F.mse_loss(raw_prediction.reshape(-1), target.reshape(-1))


def utility_from_loss(loss: torch.Tensor, alpha: float, loss_reference: float) -> torch.Tensor:
    if loss_reference <= 0:
        raise ValueError("loss_reference must be positive.")
    return float(alpha) * torch.exp(-loss / float(loss_reference)).mean()


def market_objective(
    task: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    instance_cost: torch.Tensor,
    alpha: float,
    beta: float,
    loss_reference: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss = per_sample_predictive_loss(task, prediction, target)
    utility = utility_from_loss(loss, alpha, loss_reference)
    expected_cost = instance_cost.mean()
    objective = -utility + float(beta) * expected_cost
    return objective, utility, expected_cost

