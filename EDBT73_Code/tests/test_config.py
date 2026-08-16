from pathlib import Path

import pytest

from moe_market.config import load_config, validate_config


def test_all_repository_configs_are_valid():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs").glob("*.yaml"):
        assert load_config(path)["datasets"]


def test_invalid_task_is_rejected():
    config = {
        "experiment": "bad",
        "task": "clustering",
        "datasets": ["x"],
        "seeds": [1],
        "data": {},
        "model": {},
        "training": {},
        "objective": {"loss_reference": 1.0},
        "cost": {},
        "output_dir": "results/raw/bad",
    }
    with pytest.raises(ValueError):
        validate_config(config)
