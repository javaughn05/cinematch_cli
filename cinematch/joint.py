from __future__ import annotations

import numpy as np


def joint_score(predictions: list[float] | np.ndarray, lam: float = 1.0) -> float:
    preds = np.asarray(predictions, dtype=float)
    if preds.size == 0:
        raise ValueError("joint_score requires at least one prediction")
    return float(preds.mean() - lam * preds.std())


def score_columns(prediction_matrix: np.ndarray, lam: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, std, and joint score for rows of user predictions."""
    means = prediction_matrix.mean(axis=1)
    stds = prediction_matrix.std(axis=1)
    return means, stds, means - lam * stds
