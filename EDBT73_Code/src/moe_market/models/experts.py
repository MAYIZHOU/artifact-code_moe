from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


class TabularMLPExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TabularLSTMExpert(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = inputs.unsqueeze(-1)
        output, _ = self.lstm(sequence)
        return self.head(output[:, -1, :])


class TabularTransformerExpert(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
    ):
        super().__init__()
        self.projection = nn.Linear(1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(inputs.unsqueeze(-1))
        encoded = self.encoder(tokens).mean(dim=1)
        return self.head(encoded)


class NativeTreeClassifierExpert(nn.Module):
    """Expose a fitted tree classifier as a fixed-logit MoE expert."""

    def __init__(self, output_dim: int, estimator_kind: str, options: Optional[Mapping] = None):
        super().__init__()
        if estimator_kind not in {"xgboost", "catboost"}:
            raise ValueError("estimator_kind must be 'xgboost' or 'catboost'.")
        self.output_dim = int(output_dim)
        self.estimator_kind = estimator_kind
        self.options = dict(options or {})
        self.estimator = None

    def fit(self, features: np.ndarray, target: np.ndarray, seed: int) -> None:
        features = features.reshape(features.shape[0], -1)
        common = {
            "iterations": int(self.options.get("iterations", 200)),
            "depth": int(self.options.get("depth", 6)),
            "learning_rate": float(self.options.get("learning_rate", 0.05)),
        }
        if self.estimator_kind == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as error:
                raise ImportError(
                    "The hybrid expert pool requires xgboost. Install the project requirements."
                ) from error
            xgboost_options = {
                "n_estimators": common["iterations"],
                "max_depth": common["depth"],
                "learning_rate": common["learning_rate"],
                "subsample": float(self.options.get("subsample", 0.8)),
                "colsample_bytree": float(self.options.get("colsample_bytree", 0.8)),
                "objective": "binary:logistic" if self.output_dim == 2 else "multi:softprob",
                "eval_metric": "logloss",
                "random_state": int(seed),
                "n_jobs": int(self.options.get("threads", 1)),
                "tree_method": "hist",
            }
            if self.output_dim != 2:
                xgboost_options["num_class"] = self.output_dim
            self.estimator = XGBClassifier(
                **xgboost_options,
            )
        else:
            try:
                from catboost import CatBoostClassifier
            except ImportError as error:
                raise ImportError(
                    "The hybrid expert pool requires catboost. Install the project requirements."
                ) from error
            self.estimator = CatBoostClassifier(
                iterations=common["iterations"],
                depth=common["depth"],
                learning_rate=common["learning_rate"],
                loss_function="Logloss" if self.output_dim == 2 else "MultiClass",
                random_seed=int(seed),
                thread_count=int(self.options.get("threads", 1)),
                verbose=False,
                allow_writing_files=False,
            )
        self.estimator.fit(features, target)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.estimator is None:
            raise RuntimeError(f"{self.estimator_kind} expert must be fitted before use.")
        features = inputs.detach().cpu().numpy().reshape(inputs.shape[0], -1)
        probabilities = np.asarray(
            self.estimator.predict_proba(features),
            dtype=np.float32,
        )
        if probabilities.ndim == 1:
            probabilities = np.column_stack([1.0 - probabilities, probabilities])
        logits = np.log(np.clip(probabilities, 1e-12, 1.0))
        return torch.as_tensor(logits, dtype=inputs.dtype, device=inputs.device)


class NativeTreeRegressorExpert(nn.Module):
    """Expose a fitted tree regressor as a fixed-output MoE expert."""

    def __init__(self, estimator_kind: str, options: Optional[Mapping] = None):
        super().__init__()
        if estimator_kind not in {"xgboost", "catboost"}:
            raise ValueError("estimator_kind must be 'xgboost' or 'catboost'.")
        self.estimator_kind = estimator_kind
        self.options = dict(options or {})
        self.estimator = None

    def fit(self, features: np.ndarray, target: np.ndarray, seed: int) -> None:
        common = {
            "iterations": int(self.options.get("iterations", 200)),
            "depth": int(self.options.get("depth", 6)),
            "learning_rate": float(self.options.get("learning_rate", 0.05)),
        }
        if self.estimator_kind == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as error:
                raise ImportError(
                    "The hybrid expert pool requires xgboost. Install the project requirements."
                ) from error
            self.estimator = XGBRegressor(
                n_estimators=common["iterations"],
                max_depth=common["depth"],
                learning_rate=common["learning_rate"],
                subsample=float(self.options.get("subsample", 0.8)),
                colsample_bytree=float(self.options.get("colsample_bytree", 0.8)),
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=int(seed),
                n_jobs=int(self.options.get("threads", 1)),
                tree_method="hist",
            )
        else:
            try:
                from catboost import CatBoostRegressor
            except ImportError as error:
                raise ImportError(
                    "The hybrid expert pool requires catboost. Install the project requirements."
                ) from error
            self.estimator = CatBoostRegressor(
                iterations=common["iterations"],
                depth=common["depth"],
                learning_rate=common["learning_rate"],
                loss_function="RMSE",
                random_seed=int(seed),
                thread_count=int(self.options.get("threads", 1)),
                verbose=False,
                allow_writing_files=False,
            )
        self.estimator.fit(features, target)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.estimator is None:
            raise RuntimeError(f"{self.estimator_kind} expert must be fitted before use.")
        prediction = np.asarray(
            self.estimator.predict(inputs.detach().cpu().numpy()),
            dtype=np.float32,
        ).reshape(-1, 1)
        return torch.as_tensor(prediction, dtype=inputs.dtype, device=inputs.device)


def fit_native_experts(model: nn.Module, loader: Iterable, seed: int) -> None:
    native_experts = [
        expert
        for expert in getattr(model, "experts", [])
        if isinstance(expert, (NativeTreeClassifierExpert, NativeTreeRegressorExpert))
    ]
    if not native_experts:
        return
    feature_batches, target_batches = [], []
    for features, target in loader:
        feature_batches.append(features.detach().cpu().numpy())
        target_batches.append(target.detach().cpu().numpy().reshape(-1))
    features = np.concatenate(feature_batches, axis=0)
    target = np.concatenate(target_batches, axis=0)
    for expert in native_experts:
        expert.fit(features, target, seed)


class ImageMLPExpert(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], hidden_dims: Sequence[int], output_dim: int):
        super().__init__()
        input_dim = reduce(mul, input_shape, 1)
        self.network = nn.Sequential(
            nn.Flatten(),
            TabularMLPExpert(input_dim, hidden_dims, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ImageLinearExpert(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(reduce(mul, input_shape, 1), output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ImageCNNExpert(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], output_dim: int):
        super().__init__()
        channels, height, width = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(inputs))


class ImagePatchTransformerExpert(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        output_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        layers: int = 2,
        patch_size: int = 4,
    ):
        super().__init__()
        channels, height, width = input_shape
        if height % patch_size or width % patch_size:
            raise ValueError("Image dimensions must be divisible by patch_size.")
        self.patch_size = patch_size
        patch_dim = channels * patch_size * patch_size
        patch_count = (height // patch_size) * (width // patch_size)
        self.projection = nn.Linear(patch_dim, hidden_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position = nn.Parameter(torch.zeros(1, patch_count + 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(hidden_dim, output_dim)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = inputs.shape
        patch = self.patch_size
        patches = inputs.unfold(2, patch, patch).unfold(3, patch, patch)
        patches = patches.contiguous().view(batch, channels, -1, patch, patch)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(batch, -1, channels * patch * patch)
        tokens = self.projection(patches)
        class_token = self.class_token.expand(batch, -1, -1)
        encoded = torch.cat([class_token, tokens], dim=1)
        encoded = self.encoder(encoded + self.position[:, : encoded.shape[1], :])
        return self.head(encoded[:, 0, :])


InputShape = Union[int, Tuple[int, int, int]]


def build_expert_pool(
    pool_name: str,
    input_shape: InputShape,
    output_dim: int,
    options: Optional[Mapping] = None,
) -> List[nn.Module]:
    if pool_name == "tabular_classification":
        if not isinstance(input_shape, int):
            raise TypeError("Tabular expert pools require an integer input dimension.")
        return [
            TabularLSTMExpert(64, output_dim),
            TabularMLPExpert(input_shape, [128, 64], output_dim),
            TabularMLPExpert(input_shape, [64, 64], output_dim),
            TabularTransformerExpert(output_dim, hidden_dim=64, heads=4, layers=2),
        ]
    if pool_name == "tabular_classification_hybrid":
        if not isinstance(input_shape, int):
            raise TypeError("Tabular expert pools require an integer input dimension.")
        tree_options = dict((options or {}).get("tree", {}))
        return [
            TabularMLPExpert(input_shape, [128, 64], output_dim),
            TabularTransformerExpert(output_dim, hidden_dim=64, heads=4, layers=2),
            NativeTreeClassifierExpert(output_dim, "xgboost", tree_options),
            NativeTreeClassifierExpert(output_dim, "catboost", tree_options),
        ]
    if pool_name == "tabular_regression":
        if not isinstance(input_shape, int):
            raise TypeError("Tabular expert pools require an integer input dimension.")
        return [
            TabularMLPExpert(input_shape, [64], output_dim),
            TabularMLPExpert(input_shape, [128, 64], output_dim),
            TabularMLPExpert(input_shape, [64, 64], output_dim),
            TabularTransformerExpert(output_dim, hidden_dim=64, heads=4, layers=2),
        ]
    if pool_name == "tabular_regression_hybrid":
        if not isinstance(input_shape, int):
            raise TypeError("Tabular expert pools require an integer input dimension.")
        if output_dim != 1:
            raise ValueError("The tabular regression hybrid pool requires output_dim=1.")
        tree_options = dict((options or {}).get("tree", {}))
        return [
            TabularMLPExpert(input_shape, [128, 64], output_dim),
            TabularTransformerExpert(output_dim, hidden_dim=64, heads=4, layers=2),
            NativeTreeRegressorExpert("xgboost", tree_options),
            NativeTreeRegressorExpert("catboost", tree_options),
        ]
    if pool_name in {"image_paper", "image_modern"}:
        if isinstance(input_shape, int):
            raise TypeError("Image expert pools require a (channels, height, width) tuple.")
        if pool_name == "image_paper":
            return [
                ImageMLPExpert(input_shape, [64], output_dim),
                ImageMLPExpert(input_shape, [128, 64], output_dim),
                ImageMLPExpert(input_shape, [64, 64], output_dim),
                ImagePatchTransformerExpert(input_shape, output_dim, 64, 4, 2),
            ]
        return [
            ImageCNNExpert(input_shape, output_dim),
            ImageMLPExpert(input_shape, [512, 512], output_dim),
            ImageLinearExpert(input_shape, output_dim),
            ImagePatchTransformerExpert(input_shape, output_dim, 128, 4, 3),
        ]
    if pool_name == "image_classification_hybrid":
        if isinstance(input_shape, int):
            raise TypeError("Image expert pools require a (channels, height, width) tuple.")
        tree_options = dict((options or {}).get("tree", {}))
        return [
            ImageMLPExpert(input_shape, [512, 512], output_dim),
            ImagePatchTransformerExpert(input_shape, output_dim, 128, 4, 3),
            NativeTreeClassifierExpert(output_dim, "xgboost", tree_options),
            NativeTreeClassifierExpert(output_dim, "catboost", tree_options),
        ]
    raise KeyError(f"Unknown expert pool: {pool_name}")
