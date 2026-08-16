from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from scipy.io import arff

from .registry import get_dataset_spec


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


@dataclass
class RawTabularData:
    name: str
    task: str
    features: pd.DataFrame
    target: pd.Series


def _install_certifi_https_opener() -> None:
    """Use certifi instead of potentially malformed Windows certificate-store entries."""
    import ssl
    import urllib.request

    try:
        import certifi
    except ImportError as exc:
        raise RuntimeError("Install certifi to download registered UCI datasets.") from exc
    context = ssl.create_default_context(cafile=certifi.where())
    urllib.request.install_opener(
        urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    )


def _single_target(targets: Any, target_column: Optional[str] = None) -> pd.Series:
    if isinstance(targets, pd.Series):
        return targets
    frame = pd.DataFrame(targets)
    if target_column:
        if target_column not in frame.columns:
            raise KeyError(f"Target column '{target_column}' was not found.")
        return frame[target_column]
    if frame.shape[1] != 1:
        raise ValueError(
            "The dataset contains multiple target columns. Set dataset.target_column in the config."
        )
    return frame.iloc[:, 0]


def _load_local(spec: Dict[str, Any], project_root: Path) -> RawTabularData:
    path = Path(spec["path"]).expanduser()
    if not path.is_absolute():
        path = project_root / path
    file_format = spec.get("format", path.suffix.lstrip(".")).lower()
    if file_format == "csv":
        frame = pd.read_csv(path, **spec.get("read_options", {}))
    elif file_format in {"z", "unix_z"}:
        try:
            import unlzw3
        except ImportError as exc:
            raise RuntimeError(
                "Install unlzw3==0.2.2 to read local Unix .Z dataset files."
            ) from exc
        payload = unlzw3.unlzw(path.read_bytes())
        buffer = io.StringIO(payload) if isinstance(payload, str) else io.BytesIO(payload)
        frame = pd.read_csv(buffer, **spec.get("read_options", {}))
    elif file_format in {"xls", "xlsx", "excel"}:
        frame = pd.read_excel(path, **spec.get("read_options", {}))
    elif file_format == "parquet":
        frame = pd.read_parquet(path, **spec.get("read_options", {}))
    elif file_format == "arff":
        values, _ = arff.loadarff(path)
        frame = pd.DataFrame(values)
    else:
        raise ValueError(f"Unsupported local dataset format: {file_format}")
    target_column = spec.get("target_column")
    if not target_column or target_column not in frame.columns:
        raise KeyError("A valid dataset.target_column is required for local datasets.")
    features = frame.drop(columns=[target_column])
    drop_columns = list(spec.get("drop_columns", []))
    if drop_columns:
        missing = [column for column in drop_columns if column not in features.columns]
        if missing:
            raise KeyError(f"Configured drop columns were not found: {missing}")
        features = features.drop(columns=drop_columns)
    return RawTabularData(
        name=spec["name"],
        task=spec["task"],
        features=features,
        target=frame[target_column],
    )


def _load_adult_files(spec: Dict[str, Any], project_root: Path) -> RawTabularData:
    directory = Path(spec["directory"]).expanduser()
    if not directory.is_absolute():
        directory = project_root / directory
    train_path = directory / spec.get("train_file", "adult.data")
    test_path = directory / spec.get("test_file", "adult.test")
    missing = [str(path) for path in (train_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Adult dataset files were not found: {missing}")

    read_options = {
        "names": ADULT_COLUMNS,
        "skipinitialspace": True,
        "na_values": ["?"],
    }
    train = pd.read_csv(train_path, **read_options)
    test = pd.read_csv(test_path, skiprows=1, **read_options)
    frame = pd.concat([train, test], ignore_index=True)
    frame["income"] = frame["income"].astype(str).str.strip().str.rstrip(".")
    return RawTabularData(
        name=spec.get("name", "adult"),
        task=spec.get("task", "classification"),
        features=frame.drop(columns=["income"]),
        target=frame["income"],
    )


def load_raw_tabular_data(
    dataset_name: str,
    task: str,
    project_root: Union[str, Path],
    dataset_override: Optional[Dict[str, Any]] = None,
) -> RawTabularData:
    """Load raw values only. All cleaning is intentionally delegated to cleaning.py."""
    if dataset_override and dataset_override.get("source") == "adult_files":
        spec = dict(dataset_override)
        spec.setdefault("name", dataset_name)
        spec.setdefault("task", task)
        return _load_adult_files(spec, Path(project_root))
    if dataset_override and dataset_override.get("source") == "local":
        spec = dict(dataset_override)
        spec.setdefault("name", dataset_name)
        spec.setdefault("task", task)
        return _load_local(spec, Path(project_root))

    spec = get_dataset_spec(dataset_name, task)
    if dataset_override:
        spec.update(dataset_override)
    if spec.get("source") != "uci":
        if spec.get("source") == "sklearn":
            from sklearn import datasets

            loader = getattr(datasets, spec["loader"])
            dataset = loader(as_frame=True)
            return RawTabularData(
                dataset_name,
                task,
                pd.DataFrame(dataset.data).reset_index(drop=True),
                pd.Series(dataset.target).reset_index(drop=True),
            )
        raise ValueError(
            f"Tabular dataset '{dataset_name}' must use source 'uci', 'sklearn', or 'local'."
        )

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError("Install ucimlrepo to download registered UCI datasets.") from exc

    _install_certifi_https_opener()
    dataset = fetch_ucirepo(id=int(spec["uci_id"]))
    features = pd.DataFrame(dataset.data.features).reset_index(drop=True)
    drop_columns = [column for column in spec.get("drop_columns", []) if column in features.columns]
    if drop_columns:
        features = features.drop(columns=drop_columns)
    target = _single_target(dataset.data.targets, spec.get("target_column")).reset_index(drop=True)
    return RawTabularData(dataset_name, task, features, target)
