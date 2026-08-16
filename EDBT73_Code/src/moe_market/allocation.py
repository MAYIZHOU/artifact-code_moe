from __future__ import annotations

import itertools
import math
from typing import Dict, FrozenSet, Sequence

import numpy as np
import torch

from .objectives import per_sample_predictive_loss, utility_from_loss


def allocation_shares(weights: np.ndarray, costs: Sequence[float], discount: float) -> Dict[str, np.ndarray]:
    average_participation = weights.mean(axis=0)
    costs_array = np.asarray(costs, dtype=float)
    score = average_participation * np.exp(-float(discount) * costs_array)
    cost_adjusted = score / score.sum() if score.sum() > 0 else np.zeros_like(score)
    participation = average_participation / average_participation.sum()
    uniform = np.full(weights.shape[1], 1.0 / weights.shape[1])
    return {
        "Uniform": uniform,
        "Participation-only": participation,
        "Cost-adjusted": cost_adjusted,
    }


def coalition_values(
    task: str,
    expert_predictions: torch.Tensor,
    weights: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    loss_reference: float,
) -> Dict[FrozenSet[int], float]:
    n_experts = expert_predictions.shape[1]
    values: Dict[FrozenSet[int], float] = {frozenset(): 0.0}
    for size in range(1, n_experts + 1):
        for coalition_tuple in itertools.combinations(range(n_experts), size):
            coalition = frozenset(coalition_tuple)
            indices = list(coalition_tuple)
            coalition_weights = weights[:, indices]
            coalition_weights = coalition_weights / coalition_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            prediction = (coalition_weights.unsqueeze(-1) * expert_predictions[:, indices]).sum(dim=1)
            loss = per_sample_predictive_loss(task, prediction, target)
            values[coalition] = float(utility_from_loss(loss, alpha, loss_reference).item())
    return values


def exact_shapley(values: Dict[FrozenSet[int], float], n_experts: int) -> np.ndarray:
    result = np.zeros(n_experts, dtype=float)
    denominator = math.factorial(n_experts)
    all_experts = set(range(n_experts))
    for expert in range(n_experts):
        others = all_experts - {expert}
        for size in range(n_experts):
            coefficient = math.factorial(size) * math.factorial(n_experts - size - 1) / denominator
            for coalition_tuple in itertools.combinations(others, size):
                coalition = frozenset(coalition_tuple)
                result[expert] += coefficient * (
                    values[coalition | {expert}] - values[coalition]
                )
    return result


def payment_shares_from_shapley(shapley_values: np.ndarray) -> np.ndarray:
    nonnegative = np.clip(np.asarray(shapley_values, dtype=float), 0.0, None)
    if nonnegative.sum() <= 1e-12:
        return np.full(len(nonnegative), 1.0 / len(nonnegative))
    return nonnegative / nonnegative.sum()

