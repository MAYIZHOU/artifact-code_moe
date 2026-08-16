from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from moe_market.config import load_config, project_path, with_overrides
from moe_market.costs import measure_latency_costs
from moe_market.data.cleaning import make_provider_loaders
from moe_market.models.experts import InputShape, build_expert_pool, fit_native_experts
from moe_market.models.gating import build_gate
from moe_market.models.moe import MoE
from moe_market.reproducibility import set_seed
from moe_market.training import copy_experts, train_experts_independently, train_moe


def add_common_arguments(parser: argparse.ArgumentParser, default_config: str) -> None:
    parser.add_argument("--config", default=str(PROJECT_ROOT / default_config))
    parser.add_argument("--dataset", help="Run one registered or locally configured dataset.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override random seeds.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])


def prepare_config(arguments: argparse.Namespace) -> Dict[str, Any]:
    return with_overrides(load_config(arguments.config), arguments.dataset, arguments.seeds)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(name)


def dataset_override(config: Dict[str, Any], dataset_name: str) -> Optional[Dict[str, Any]]:
    overrides = config.get("dataset_overrides", {})
    return overrides.get(dataset_name)


def build_model(config: Dict[str, Any], input_shape: InputShape, output_dim: int) -> MoE:
    model_config = config["model"]
    experts = build_expert_pool(model_config["expert_pool"], input_shape, output_dim, model_config)
    gate = build_gate(
        input_shape,
        len(experts),
        hidden=int(model_config.get("gate_hidden", 64)),
        depth=int(model_config.get("gate_depth", 1)),
    )
    return MoE(
        experts,
        gate,
        task=config["task"],
        routing=model_config.get("routing", "dense"),
        top_k=int(model_config.get("top_k", 1)),
    )


def _reset_loader(loader: Iterable, seed: int) -> None:
    generator = getattr(loader, "generator", None)
    if generator is not None:
        generator.manual_seed(seed)


def _resolve_cost_values(
    config: Dict[str, Any],
    model: MoE,
    train_loader: Iterable,
    device: torch.device,
) -> None:
    cost_config = config["cost"]
    source = cost_config.get("source", "fixed")
    if source == "fixed":
        values = cost_config.get("values")
        if values is None or len(values) != len(model.experts):
            raise ValueError("cost.values must contain one fixed cost for each expert.")
        return
    if source != "latency":
        raise ValueError("cost.source must be 'fixed' or 'latency'.")
    if cost_config.get("measure_once", False) and cost_config.get("values") is not None:
        return
    sample, _ = next(iter(train_loader))
    measurement_batch = int(cost_config.get("measurement_batch_size", 64))
    sample = sample[:measurement_batch].to(device)
    model.to(device)
    cost_config["values"] = measure_latency_costs(
        model.experts,
        sample,
        warmup=int(cost_config.get("warmup", 10)),
        repeats=int(cost_config.get("repeats", 50)),
    )


def train_comparison_models(
    config: Dict[str, Any],
    input_shape: InputShape,
    output_dim: int,
    train_dataset: torch.utils.data.Dataset,
    train_loader: Iterable,
    validation_loader: Iterable,
    seed: int,
    device: torch.device,
) -> Tuple[MoE, MoE]:
    """Train Standard MoE and MoE Market from matched initial conditions."""
    mode = config["model"].get("training_mode", "joint")

    if mode == "independent_frozen":
        set_seed(seed)
        provider_model = build_model(config, input_shape, output_dim)
        _reset_loader(train_loader, seed)
        fit_native_experts(provider_model, train_loader, seed)
        provider_mode = config["model"].get("provider_split", "full")
        provider_loaders = make_provider_loaders(
            train_dataset,
            len(provider_model.experts),
            provider_mode,
            int(config["data"].get("batch_size", 64)),
            seed,
            int(config["data"].get("num_workers", 0)),
        )
        train_experts_independently(
            provider_model,
            provider_loaders,
            validation_loader,
            config["task"],
            device,
            int(
                config["training"].get(
                    "expert_fit_epochs",
                    config["training"].get("expert_pretrain_epochs", 10),
                )
            ),
            float(config["training"].get("learning_rate", 1e-3)),
            float(config["training"].get("weight_decay", 0.0)),
        )
        _resolve_cost_values(config, provider_model, train_loader, device)
        set_seed(seed)
        standard = build_model(config, input_shape, output_dim)
        set_seed(seed)
        market = build_model(config, input_shape, output_dim)
        copy_experts(provider_model, standard, freeze=True)
        copy_experts(provider_model, market, freeze=True)
    elif mode == "joint":
        set_seed(seed)
        standard = build_model(config, input_shape, output_dim)
        _reset_loader(train_loader, seed)
        fit_native_experts(standard, train_loader, seed)
        set_seed(seed)
        market = build_model(config, input_shape, output_dim)
        _reset_loader(train_loader, seed)
        fit_native_experts(market, train_loader, seed)
        _resolve_cost_values(config, standard, train_loader, device)
    else:
        raise ValueError("model.training_mode must be 'joint' or 'independent_frozen'.")

    _reset_loader(train_loader, seed)
    train_moe(
        standard,
        train_loader,
        validation_loader,
        config["task"],
        "predictive",
        device,
        config["training"],
        config["objective"],
        config["cost"],
    )
    _reset_loader(train_loader, seed)
    train_moe(
        market,
        train_loader,
        validation_loader,
        config["task"],
        "market",
        device,
        config["training"],
        config["objective"],
        config["cost"],
    )
    return standard, market


def rows_from_results(
    dataset_name: str,
    seed: int,
    results: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, metrics in results.items():
        row: Dict[str, Any] = {"dataset": dataset_name, "seed": seed, "method": method}
        row.update(metrics)
        rows.append(row)
    return rows


def output_directory(config: Dict[str, Any]) -> Path:
    return project_path(config, config["output_dir"])
