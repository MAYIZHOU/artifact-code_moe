from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from ..reproducibility import set_seed, worker_init_fn
from .loaders import RawTabularData
from .registry import IMAGE_DATASETS


class ArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, target: np.ndarray, task: str):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        target_dtype = torch.long if task == "classification" else torch.float32
        target_tensor = torch.as_tensor(target, dtype=target_dtype)
        self.target = target_tensor if task == "classification" else target_tensor.reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int):
        return self.features[index], self.target[index]


@dataclass
class CleanedTabularData:
    name: str
    task: str
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    input_dim: int
    num_outputs: int
    preprocessor: ColumnTransformer
    label_encoder: Optional[LabelEncoder]
    target_scaler: Optional[StandardScaler]


@dataclass
class ImageData:
    name: str
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    input_shape: Tuple[int, int, int]
    num_outputs: int


def _one_hot_encoder(categories: Any = "auto") -> OneHotEncoder:
    try:
        return OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse=False,
        )


def _decode_objects(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(
                lambda value: value.decode("utf-8")
                if isinstance(value, (bytes, bytearray))
                else value
            )
    return result


def _build_preprocessor(
    features: pd.DataFrame,
    categorical_categories: Optional[Dict[str, List[Any]]] = None,
) -> ColumnTransformer:
    categorical = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical = [column for column in features.columns if column not in categorical]
    transformers = []
    if numerical:
        transformers.append(
            (
                "numerical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            )
        )
    if categorical:
        encoder_categories: Any = "auto"
        if categorical_categories:
            encoder_categories = [
                categorical_categories.get(
                    column,
                    sorted(features[column].dropna().unique().tolist()),
                )
                for column in categorical
            ]
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _one_hot_encoder(encoder_categories)),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("The dataset does not contain usable feature columns.")
    return ColumnTransformer(transformers, remainder="drop")


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, workers: int) -> DataLoader:
    generator = set_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator,
        worker_init_fn=worker_init_fn(),
        pin_memory=torch.cuda.is_available(),
    )


def clean_tabular_data(raw: RawTabularData, config: Dict[str, Any], seed: int) -> CleanedTabularData:
    """Split first, then fit cleaning operations on training data only."""
    features = _decode_objects(raw.features).replace([np.inf, -np.inf], np.nan)
    categorical_columns = list(config.get("categorical_columns", []))
    missing_categorical = [column for column in categorical_columns if column not in features.columns]
    if missing_categorical:
        raise KeyError(f"Configured categorical columns were not found: {missing_categorical}")
    for column in categorical_columns:
        features[column] = features[column].astype("category")
    target = raw.target.copy()
    valid_target = ~pd.isna(target)
    features = features.loc[valid_target].reset_index(drop=True)
    target = target.loc[valid_target].reset_index(drop=True)

    label_encoder = None
    target_scaler = None
    if raw.task == "classification":
        label_encoder = LabelEncoder()
        target_values = label_encoder.fit_transform(target.astype(str))
        stratify = target_values
        num_outputs = int(len(label_encoder.classes_))
    else:
        target_values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=np.float32)
        valid_numeric = ~np.isnan(target_values)
        features = features.loc[valid_numeric].reset_index(drop=True)
        target_values = target_values[valid_numeric]
        stratify = None
        num_outputs = 1

    test_ratio = float(config.get("test_ratio", 0.1))
    validation_ratio = float(config.get("validation_ratio", 0.1))
    if test_ratio <= 0 or validation_ratio <= 0 or test_ratio + validation_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be positive and sum to less than one.")

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        target_values,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify,
    )
    relative_validation = validation_ratio / (1.0 - test_ratio)
    stratify_train_val = y_train_val if raw.task == "classification" else None
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_validation,
        random_state=seed,
        stratify=stratify_train_val,
    )

    preprocessor = _build_preprocessor(
        x_train,
        config.get("categorical_categories"),
    )
    x_train_clean = np.asarray(preprocessor.fit_transform(x_train), dtype=np.float32)
    x_validation_clean = np.asarray(preprocessor.transform(x_validation), dtype=np.float32)
    x_test_clean = np.asarray(preprocessor.transform(x_test), dtype=np.float32)

    if raw.task == "regression" and bool(config.get("scale_target", True)):
        target_scaler = StandardScaler()
        y_train = target_scaler.fit_transform(np.asarray(y_train).reshape(-1, 1)).reshape(-1)
        y_validation = target_scaler.transform(np.asarray(y_validation).reshape(-1, 1)).reshape(-1)
        y_test = target_scaler.transform(np.asarray(y_test).reshape(-1, 1)).reshape(-1)

    train_dataset = ArrayDataset(x_train_clean, np.asarray(y_train), raw.task)
    validation_dataset = ArrayDataset(x_validation_clean, np.asarray(y_validation), raw.task)
    test_dataset = ArrayDataset(x_test_clean, np.asarray(y_test), raw.task)
    batch_size = int(config.get("batch_size", 64))
    workers = int(config.get("num_workers", 0))
    return CleanedTabularData(
        name=raw.name,
        task=raw.task,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=_make_loader(train_dataset, batch_size, True, seed, workers),
        validation_loader=_make_loader(validation_dataset, batch_size, False, seed, workers),
        test_loader=_make_loader(test_dataset, batch_size, False, seed, workers),
        input_dim=x_train_clean.shape[1],
        num_outputs=num_outputs,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        target_scaler=target_scaler,
    )


def make_provider_loaders(
    train_dataset: Dataset,
    n_experts: int,
    mode: str,
    batch_size: int,
    seed: int,
    workers: int = 0,
) -> List[DataLoader]:
    if mode == "full":
        return [_make_loader(train_dataset, batch_size, True, seed + i, workers) for i in range(n_experts)]
    if mode != "disjoint":
        raise ValueError("provider_split.mode must be 'full' or 'disjoint'.")
    base, remainder = divmod(len(train_dataset), n_experts)
    lengths = [base + (1 if i < remainder else 0) for i in range(n_experts)]
    subsets = random_split(train_dataset, lengths, generator=set_seed(seed))
    return [_make_loader(subset, batch_size, True, seed + i, workers) for i, subset in enumerate(subsets)]


def _image_transforms(dataset_name: str, image_size: int, augment: bool):
    from torchvision import transforms

    channels = IMAGE_DATASETS[dataset_name]["channels"]
    base = [transforms.Resize((image_size, image_size))]
    if channels == 1:
        base.append(transforms.Grayscale(num_output_channels=3))
    train_steps = list(base)
    if augment and dataset_name == "svhn":
        train_steps.extend([transforms.RandomHorizontalFlip(), transforms.RandomCrop(image_size, padding=4)])
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return transforms.Compose(train_steps + [transforms.ToTensor(), normalize]), transforms.Compose(
        base + [transforms.ToTensor(), normalize]
    )


def _install_certifi_https_opener() -> None:
    """Avoid malformed Windows certificate-store entries during dataset downloads."""
    import ssl
    import urllib.request

    try:
        import certifi
    except ImportError as exc:
        raise RuntimeError("Install certifi to download torchvision datasets over HTTPS.") from exc
    context = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    urllib.request.install_opener(opener)


def prepare_image_data(
    dataset_name: str,
    config: Dict[str, Any],
    project_root: Union[str, Path],
    seed: int,
) -> ImageData:
    from torchvision import datasets

    if bool(config.get("use_certifi", True)):
        _install_certifi_https_opener()
    if dataset_name not in IMAGE_DATASETS:
        raise KeyError(f"Unknown image dataset: {dataset_name}")
    spec = IMAGE_DATASETS[dataset_name]
    dataset_class = getattr(datasets, spec["torchvision"])
    image_size = int(config.get("image_size", 32))
    train_transform, test_transform = _image_transforms(dataset_name, image_size, bool(config.get("augment", False)))
    root = Path(project_root) / config.get("data_dir", "data")
    common = {"root": root, "download": True}
    if dataset_name == "svhn":
        full_train = dataset_class(split="train", transform=train_transform, **common)
        full_validation = dataset_class(split="train", transform=test_transform, **common)
        test_dataset = dataset_class(split="test", transform=test_transform, **common)
    else:
        full_train = dataset_class(train=True, transform=train_transform, **common)
        full_validation = dataset_class(train=True, transform=test_transform, **common)
        test_dataset = dataset_class(train=False, transform=test_transform, **common)

    validation_ratio = float(config.get("validation_ratio", 0.1))
    validation_size = max(1, int(len(full_train) * validation_ratio))
    indices = torch.randperm(len(full_train), generator=set_seed(seed)).tolist()
    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    max_train_samples = int(config.get("max_train_samples", 0))
    max_validation_samples = int(config.get("max_validation_samples", 0))
    if max_train_samples > 0:
        train_indices = train_indices[:max_train_samples]
    if max_validation_samples > 0:
        validation_indices = validation_indices[:max_validation_samples]
    train_dataset = Subset(full_train, train_indices)
    validation_dataset = Subset(full_validation, validation_indices)
    max_test_samples = int(config.get("max_test_samples", 0))
    if max_test_samples > 0:
        test_dataset = Subset(test_dataset, range(min(max_test_samples, len(test_dataset))))
    batch_size = int(config.get("batch_size", 128))
    workers = int(config.get("num_workers", 0))
    return ImageData(
        name=dataset_name,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=_make_loader(train_dataset, batch_size, True, seed, workers),
        validation_loader=_make_loader(validation_dataset, batch_size, False, seed, workers),
        test_loader=_make_loader(test_dataset, batch_size, False, seed, workers),
        input_shape=(3, image_size, image_size),
        num_outputs=int(spec["num_classes"]),
    )
