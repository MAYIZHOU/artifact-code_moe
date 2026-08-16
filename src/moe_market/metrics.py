from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score


def classification_metrics(probabilities: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(target, prediction)),
        "f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "ce": float(-np.log(np.clip(probabilities[np.arange(len(target)), target], 1e-12, 1.0)).mean()),
    }
    try:
        if probabilities.shape[1] == 2:
            result["auc"] = float(roc_auc_score(target, probabilities[:, 1]))
        else:
            result["auc"] = float(roc_auc_score(target, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        result["auc"] = float("nan")
    return result


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    mse = float(mean_squared_error(target, prediction))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(target, prediction)),
    }

