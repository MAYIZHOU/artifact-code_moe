from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import sklearn
import torch


def _package_version(distribution: str) -> Optional[str]:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def environment_metadata() -> Dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "xgboost": _package_version("xgboost"),
        "catboost": _package_version("catboost"),
        "matplotlib": _package_version("matplotlib"),
        "pyyaml": _package_version("PyYAML"),
        "ucimlrepo": _package_version("ucimlrepo"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def save_run(
    output_dir: Path,
    dataset: str,
    seed: int,
    rows: Iterable[Dict[str, Any]],
    config: Dict,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "seed": seed,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "environment": environment_metadata(),
        "run_metadata": run_metadata or {},
        "results": list(rows),
    }
    path = output_dir / f"{dataset}_seed{seed}.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=True, allow_nan=True)
    return path


def write_combined_csv(output_dir: Path) -> Path:
    records: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("*_seed*.json")):
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        for row in payload["results"]:
            records.append({"dataset": payload["dataset"], "seed": payload["seed"], **row})
    frame = pd.DataFrame(records)
    destination = output_dir / "all_runs.csv"
    frame.to_csv(destination, index=False)
    return destination
