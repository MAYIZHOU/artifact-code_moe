from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


TABULAR_DATASETS: Dict[str, Dict[str, Any]] = {
    "breast_cancer": {"source": "uci", "uci_id": 17, "task": "classification"},
    "page_blocks": {"source": "uci", "uci_id": 78, "task": "classification"},
    "yeast": {"source": "uci", "uci_id": 110, "task": "classification"},
    "adult": {"source": "uci", "uci_id": 2, "task": "classification"},
    "ionosphere": {"source": "uci", "uci_id": 52, "task": "classification"},
    "abalone": {"source": "uci", "uci_id": 1, "task": "regression"},
    "concrete": {"source": "uci", "uci_id": 165, "task": "regression"},
    "forest_fires": {"source": "uci", "uci_id": 162, "task": "regression"},
    "bike_sharing": {
        "source": "uci",
        "uci_id": 275,
        "task": "regression",
        "drop_columns": ["instant", "dteday", "casual", "registered"],
    },
    "diabetes": {"source": "sklearn", "loader": "load_diabetes", "task": "regression"},
    "wine_quality": {"source": "uci", "uci_id": 186, "task": "regression"},
}


IMAGE_DATASETS: Dict[str, Dict[str, Any]] = {
    "mnist": {"torchvision": "MNIST", "channels": 1, "num_classes": 10},
    "fashion_mnist": {"torchvision": "FashionMNIST", "channels": 1, "num_classes": 10},
    "kmnist": {"torchvision": "KMNIST", "channels": 1, "num_classes": 10},
    "svhn": {"torchvision": "SVHN", "channels": 3, "num_classes": 10},
}


def get_dataset_spec(name: str, task: Optional[str] = None) -> Dict[str, Any]:
    if name in TABULAR_DATASETS:
        spec = deepcopy(TABULAR_DATASETS[name])
    elif name in IMAGE_DATASETS:
        spec = deepcopy(IMAGE_DATASETS[name])
        spec["source"] = "torchvision"
        spec["task"] = "classification"
    else:
        raise KeyError(
            f"Unknown dataset '{name}'. Add it to data/registry.py or use a local dataset config."
        )
    spec["name"] = name
    if task and spec.get("task") != task:
        raise ValueError(f"Dataset '{name}' is registered for {spec.get('task')}, not {task}.")
    return spec
