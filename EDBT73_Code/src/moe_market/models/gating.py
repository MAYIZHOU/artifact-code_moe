from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn


class TabularGate(nn.Module):
    def __init__(self, input_dim: int, n_experts: int, hidden: int = 64, depth: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            previous = hidden
        layers.append(nn.Linear(previous, n_experts))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ImageGate(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], n_experts: int):
        super().__init__()
        channels, _, _ = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.output = nn.Linear(32 * 4 * 4, n_experts)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(inputs))


def build_gate(
    input_shape: Union[int, Tuple[int, int, int]],
    n_experts: int,
    hidden: int = 64,
    depth: int = 1,
) -> nn.Module:
    if isinstance(input_shape, int):
        return TabularGate(input_shape, n_experts, hidden=hidden, depth=depth)
    return ImageGate(input_shape, n_experts)
