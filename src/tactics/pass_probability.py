"""Small deterministic pass-completion probability model."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.tactics.pass_candidates import FEATURE_NAMES


@dataclass
class PassProbabilityModel:
    feature_names: tuple[str, ...] = FEATURE_NAMES
    coefficients: tuple[float, ...] = ()
    intercept: float = 0.0
    means: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()
    model_type: str = "heuristic_fallback"
    n_positive: int = 0
    n_negative: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _matrix(features: pd.DataFrame | Sequence[Mapping[str, float]], names: Sequence[str]) -> np.ndarray:
    frame = features if isinstance(features, pd.DataFrame) else pd.DataFrame(features)
    return frame.loc[:, list(names)].to_numpy(float)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def heuristic_probability(features: pd.DataFrame | Sequence[Mapping[str, float]]) -> np.ndarray:
    """A fixed, monotonic baseline; it is not trained on event outcomes."""
    x = _matrix(features, FEATURE_NAMES)
    distance, forward, angle, source_pressure, target_pressure, space, support, obstruction, width, length = x.T
    score = (1.0 - np.clip(distance / 0.70, 0, 1)) + 0.35 * np.clip(forward, -0.3, 0.3)
    score += 0.30 * np.clip(target_pressure / 0.30, 0, 1) + 0.20 * np.clip(space / 0.30, 0, 1)
    score += 0.08 * np.clip(support / 4, 0, 1) - 0.20 * np.clip(obstruction / 3, 0, 1)
    score -= 0.05 * np.abs(angle) / np.pi
    return _sigmoid(score - 0.65)


def fit_pass_model(training_data: pd.DataFrame, labels: Sequence[int], config: Mapping[str, Any] | None = None) -> PassProbabilityModel:
    """Fit regularized logistic regression or honestly return the baseline."""
    cfg = dict(config or {})
    y = np.asarray(labels, dtype=int)
    positives, negatives = int(y.sum()), int((y == 0).sum())
    minimum = int(cfg.get("minimum_examples_per_class", 10))
    if positives < minimum or negatives < minimum:
        return PassProbabilityModel(n_positive=positives, n_negative=negatives)
    x = _matrix(training_data, FEATURE_NAMES)
    means, scales = x.mean(axis=0), x.std(axis=0)
    scales[scales < 1e-9] = 1.0
    z = (x - means) / scales
    weights = np.zeros(z.shape[1])
    intercept = 0.0
    learning_rate = float(cfg.get("learning_rate", 0.08))
    regularization = float(cfg.get("regularization", 0.05))
    for _ in range(int(cfg.get("iterations", 500))):
        p = _sigmoid(z @ weights + intercept)
        error = p - y
        weights -= learning_rate * ((z.T @ error) / len(y) + regularization * weights)
        intercept -= learning_rate * float(error.mean())
    return PassProbabilityModel(coefficients=tuple(weights), intercept=intercept,
                               means=tuple(means), scales=tuple(scales), model_type="logistic_regression",
                               n_positive=positives, n_negative=negatives)


def predict_pass_probability(model: PassProbabilityModel, features: pd.DataFrame | Sequence[Mapping[str, float]]) -> np.ndarray:
    if model.model_type == "heuristic_fallback":
        return heuristic_probability(features)
    x = _matrix(features, model.feature_names)
    return _sigmoid(((x - np.asarray(model.means)) / np.asarray(model.scales)) @ np.asarray(model.coefficients) + model.intercept)