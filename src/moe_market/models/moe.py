from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class MoEOutput:
    prediction: torch.Tensor
    weights: torch.Tensor
    expert_predictions: torch.Tensor
    active_mask: torch.Tensor


class MoE(nn.Module):
    def __init__(
        self,
        experts: Sequence[nn.Module],
        gate: nn.Module,
        task: str,
        routing: str = "dense",
        top_k: int = 1,
    ):
        super().__init__()
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        if routing not in {"dense", "top_k"}:
            raise ValueError("routing must be 'dense' or 'top_k'.")
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.task = task
        self.routing = routing
        self.top_k = int(top_k)

    def routing_weights(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.gate(inputs), dim=-1)
        if self.routing == "top_k":
            k = min(max(1, self.top_k), weights.shape[1])
            indices = torch.topk(weights, k=k, dim=-1).indices
            mask = torch.zeros_like(weights, dtype=torch.bool).scatter_(1, indices, True)
            weights = weights.masked_fill(~mask, 0.0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            mask = weights > 0
        return weights, mask

    def forward(self, inputs: torch.Tensor) -> MoEOutput:
        weights, active_mask = self.routing_weights(inputs)
        raw_outputs = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        if self.task == "classification":
            expert_predictions = torch.softmax(raw_outputs, dim=-1)
        else:
            expert_predictions = raw_outputs
        prediction = (weights.unsqueeze(-1) * expert_predictions).sum(dim=1)
        return MoEOutput(prediction, weights, expert_predictions, active_mask)

    def freeze_experts(self) -> None:
        for expert in self.experts:
            expert.eval()
            for parameter in expert.parameters():
                parameter.requires_grad = False
