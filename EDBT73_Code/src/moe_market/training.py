from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from .costs import expert_cost_matrix, service_cost
from .models.moe import MoE
from .objectives import individual_expert_loss, market_objective, per_sample_predictive_loss


def _clone_state(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def train_experts_independently(
    model: MoE,
    provider_loaders: Sequence[Iterable],
    validation_loader: Iterable,
    task: str,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> MoE:
    if len(provider_loaders) != len(model.experts):
        raise ValueError("One provider loader is required for each expert.")
    model.to(device)
    for expert_index, (expert, loader) in enumerate(zip(model.experts, provider_loaders)):
        trainable = [parameter for parameter in expert.parameters() if parameter.requires_grad]
        if not trainable:
            expert.eval()
            continue
        optimizer = torch.optim.Adam(trainable, lr=learning_rate, weight_decay=weight_decay)
        best_loss, best_state = float("inf"), None
        for _ in range(epochs):
            expert.train()
            for inputs, target in loader:
                inputs, target = inputs.to(device), target.to(device)
                loss = individual_expert_loss(task, expert(inputs), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            expert.eval()
            validation_losses: List[float] = []
            with torch.no_grad():
                for inputs, target in validation_loader:
                    inputs, target = inputs.to(device), target.to(device)
                    validation_losses.append(float(individual_expert_loss(task, expert(inputs), target).item()))
            validation_loss = float(np.mean(validation_losses))
            if validation_loss < best_loss:
                best_loss, best_state = validation_loss, _clone_state(expert)
        if best_state is None:
            raise RuntimeError(f"Expert {expert_index} did not complete a training epoch.")
        expert.load_state_dict(best_state)
    model.freeze_experts()
    return model


def _validation_objective(
    model: MoE,
    loader: Iterable,
    task: str,
    objective_name: str,
    device: torch.device,
    objective_config: Dict,
    cost_config: Dict,
) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for inputs, target in loader:
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            if objective_name == "predictive":
                value = per_sample_predictive_loss(task, output.prediction, target).mean()
            else:
                costs = expert_cost_matrix(inputs, cost_config["values"], cost_config.get("input_scale", 0.0))
                instance_cost = service_cost(
                    output.weights,
                    costs,
                    output.active_mask,
                    cost_config.get("training_mode", cost_config.get("mode", "weighted")),
                )
                value, _, _ = market_objective(
                    task,
                    output.prediction,
                    target,
                    instance_cost,
                    objective_config["alpha"],
                    objective_config["beta"],
                    objective_config["loss_reference"],
                )
            values.append(float(value.item()))
    return float(np.mean(values))


def train_moe(
    model: MoE,
    train_loader: Iterable,
    validation_loader: Iterable,
    task: str,
    objective_name: str,
    device: torch.device,
    training_config: Dict,
    objective_config: Dict,
    cost_config: Dict,
) -> MoE:
    if objective_name not in {"predictive", "market"}:
        raise ValueError("objective_name must be 'predictive' or 'market'.")
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable,
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    best_value, best_state = float("inf"), None
    patience = int(training_config.get("patience", training_config.get("epochs", 10)))
    stale_epochs = 0
    for _ in range(int(training_config.get("epochs", 10))):
        model.train()
        if all(not parameter.requires_grad for expert in model.experts for parameter in expert.parameters()):
            for expert in model.experts:
                expert.eval()
        for inputs, target in train_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            if objective_name == "predictive":
                loss = per_sample_predictive_loss(task, output.prediction, target).mean()
            else:
                costs = expert_cost_matrix(inputs, cost_config["values"], cost_config.get("input_scale", 0.0))
                instance_cost = service_cost(
                    output.weights,
                    costs,
                    output.active_mask,
                    cost_config.get("training_mode", cost_config.get("mode", "weighted")),
                )
                loss, _, _ = market_objective(
                    task,
                    output.prediction,
                    target,
                    instance_cost,
                    objective_config["alpha"],
                    objective_config["beta"],
                    objective_config["loss_reference"],
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        value = _validation_objective(
            model,
            validation_loader,
            task,
            objective_name,
            device,
            objective_config,
            cost_config,
        )
        if value < best_value:
            best_value, best_state, stale_epochs = value, _clone_state(model), 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not complete an epoch.")
    model.load_state_dict(best_state)
    return model


def copy_experts(source: MoE, destination: MoE, freeze: bool = False) -> None:
    for source_expert, destination_expert in zip(source.experts, destination.experts):
        destination_expert.load_state_dict(copy.deepcopy(source_expert.state_dict()))
        if hasattr(source_expert, "estimator") and hasattr(destination_expert, "estimator"):
            destination_expert.estimator = copy.deepcopy(source_expert.estimator)
    if freeze:
        destination.freeze_experts()
