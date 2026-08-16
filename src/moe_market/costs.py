from __future__ import annotations

import time
from typing import Dict, List, Sequence

import torch
import torch.nn as nn


def expert_cost_matrix(
    inputs: torch.Tensor,
    base_costs: Sequence[float],
    input_scale: float = 0.0,
) -> torch.Tensor:
    costs = torch.as_tensor(base_costs, dtype=inputs.dtype, device=inputs.device).reshape(1, -1)
    costs = costs.expand(inputs.shape[0], -1)
    if input_scale:
        complexity = inputs.reshape(inputs.shape[0], -1).abs().mean(dim=1, keepdim=True)
        costs = costs * (1.0 + float(input_scale) * complexity)
    return costs


def service_cost(
    weights: torch.Tensor,
    costs: torch.Tensor,
    active_mask: torch.Tensor,
    mode: str = "weighted",
) -> torch.Tensor:
    if mode == "weighted":
        return (weights * costs).sum(dim=1)
    if mode == "discrete":
        return (active_mask.to(costs.dtype) * costs).sum(dim=1)
    raise ValueError("cost.mode must be 'weighted' or 'discrete'.")


@torch.no_grad()
def measure_latency_profile(
    experts: Sequence[nn.Module],
    sample: torch.Tensor,
    warmup: int = 10,
    repeats: int = 50,
    normalization: str = "mean",
) -> Dict[str, List[float]]:
    """Measure per-instance expert latency and derive normalized unit cost rates."""
    if sample.shape[0] <= 0:
        raise ValueError("A non-empty measurement batch is required.")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive.")

    measurements: List[float] = []
    for expert in experts:
        expert.eval()
        for _ in range(warmup):
            expert(sample)
        if sample.is_cuda:
            torch.cuda.synchronize(sample.device)

        samples_ms: List[float] = []
        for _ in range(repeats):
            if sample.is_cuda:
                torch.cuda.synchronize(sample.device)
            start = time.perf_counter()
            expert(sample)
            if sample.is_cuda:
                torch.cuda.synchronize(sample.device)
            samples_ms.append((time.perf_counter() - start) * 1000.0 / sample.shape[0])
        measurements.append(float(torch.tensor(samples_ms, dtype=torch.float64).median().item()))

    if normalization == "mean":
        normalizer = sum(measurements) / len(measurements)
    elif normalization == "minimum":
        normalizer = min(measurements)
    else:
        raise ValueError("normalization must be 'mean' or 'minimum'.")
    normalizer = max(normalizer, 1e-12)
    normalized = [value / normalizer for value in measurements]
    return {
        "latency_ms_per_instance": measurements,
        "normalized_unit_cost": normalized,
        "normalization_value_ms": [normalizer],
    }


@torch.no_grad()
def measure_latency_costs(
    experts: Sequence[nn.Module],
    sample: torch.Tensor,
    warmup: int = 10,
    repeats: int = 50,
) -> List[float]:
    """Backward-compatible wrapper returning normalized latency costs only."""
    return measure_latency_profile(experts, sample, warmup, repeats)["normalized_unit_cost"]
