from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    for key in ["experiment", "task", "datasets", "seeds", "data", "model", "training", "objective", "cost", "output_dir"]:
        if key not in config:
            raise KeyError(f"Missing required configuration key: {key}")
    if config["task"] not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'.")
    if not isinstance(config["datasets"], list) or not config["datasets"]:
        raise ValueError("datasets must be a non-empty list.")
    if not isinstance(config["seeds"], list) or not config["seeds"]:
        raise ValueError("seeds must be a non-empty list.")
    if config["model"].get("routing", "dense") not in {"dense", "top_k"}:
        raise ValueError("model.routing must be 'dense' or 'top_k'.")
    if config["model"].get("training_mode", "joint") not in {"joint", "independent_frozen"}:
        raise ValueError("model.training_mode must be 'joint' or 'independent_frozen'.")
    if float(config["objective"].get("loss_reference", 0.0)) <= 0:
        raise ValueError("objective.loss_reference must be positive.")
    for key in ["mode", "training_mode", "evaluation_mode"]:
        if key in config["cost"] and config["cost"][key] not in {"weighted", "discrete"}:
            raise ValueError(f"cost.{key} must be 'weighted' or 'discrete'.")


def with_overrides(
    config: Dict[str, Any],
    dataset: Optional[str] = None,
    seeds: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    updated = copy.deepcopy(config)
    if dataset:
        updated["datasets"] = [dataset]
    if seeds:
        updated["seeds"] = [int(seed) for seed in seeds]
    return updated


def project_path(config: Dict[str, Any], value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def require(config: Dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing required configuration key: {dotted_key}")
        value = value[key]
    return value
