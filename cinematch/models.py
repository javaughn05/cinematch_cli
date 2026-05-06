from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from cinematch.features import FeatureEncoder, fit_encoder_from_dir

ALPHAS = np.logspace(-3, 3, 13)


@dataclass
class TrainedUserModel:
    username: str
    estimator: object
    y_mean: float
    model_name: str
    alpha: float
    mae: float
    n_train: int
    n_test: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.estimator.predict(X) + self.y_mean, 0.5, 5.0)


def _cv_for_size(n_rows: int) -> int:
    return max(2, min(5, n_rows))


def _fit_candidate(estimator: object, X: np.ndarray, y_centered: np.ndarray, cv: int) -> GridSearchCV:
    search = GridSearchCV(
        estimator,
        {"alpha": ALPHAS},
        cv=KFold(n_splits=cv, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
    )
    search.fit(X, y_centered)
    return search


def train_user_model(
    username: str,
    df: pd.DataFrame,
    encoder: FeatureEncoder,
    model_kind: str = "best",
    test_size: float = 0.2,
) -> TrainedUserModel:
    X, y = encoder.transform_user(df)
    if len(y) < 10:
        raise ValueError(f"{username} has only {len(y)} rated enriched films")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )
    y_mean = float(y_train.mean())
    y_train_centered = y_train - y_mean
    cv = _cv_for_size(len(y_train))

    candidates: list[tuple[str, GridSearchCV]] = []
    if model_kind in {"ridge", "best"}:
        candidates.append(("ridge", _fit_candidate(Ridge(), X_train, y_train_centered, cv)))
    if model_kind in {"lasso", "best"}:
        candidates.append(
            (
                "lasso",
                _fit_candidate(Lasso(max_iter=10000, random_state=42), X_train, y_train_centered, cv),
            )
        )
    if not candidates:
        raise ValueError("model_kind must be one of: ridge, lasso, best")

    best_name = ""
    best_search: GridSearchCV | None = None
    best_cv_score = -np.inf
    for name, search in candidates:
        if float(search.best_score_) > best_cv_score:
            best_name = name
            best_search = search
            best_cv_score = float(search.best_score_)
    assert best_search is not None

    preds = np.clip(best_search.best_estimator_.predict(X_test) + y_mean, 0.5, 5.0)
    mae = float(mean_absolute_error(y_test, preds))
    return TrainedUserModel(
        username=username,
        estimator=best_search.best_estimator_,
        y_mean=y_mean,
        model_name=best_name,
        alpha=float(best_search.best_params_["alpha"]),
        mae=mae,
        n_train=len(y_train),
        n_test=len(y_test),
    )


def train_all_models(
    enriched_dir: Path = Path("data/enriched"),
    model_dir: Path = Path("models"),
    model_kind: str = "best",
) -> tuple[FeatureEncoder, dict[str, TrainedUserModel]]:
    encoder, frames = fit_encoder_from_dir(enriched_dir)
    models = {
        username: train_user_model(username, df, encoder, model_kind=model_kind)
        for username, df in frames.items()
    }
    save_artifacts(encoder, models, model_dir)
    return encoder, models


def save_artifacts(
    encoder: FeatureEncoder,
    models: dict[str, TrainedUserModel],
    model_dir: Path = Path("models"),
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoder, model_dir / "feature_encoder.joblib")
    for username, model in models.items():
        joblib.dump(model, model_dir / f"{username}.joblib")


def load_artifacts(model_dir: Path = Path("models")) -> tuple[FeatureEncoder, dict[str, TrainedUserModel]]:
    encoder = joblib.load(model_dir / "feature_encoder.joblib")
    models = {
        path.stem: joblib.load(path)
        for path in sorted(model_dir.glob("*.joblib"))
        if path.name != "feature_encoder.joblib"
    }
    if not models:
        raise ValueError(f"No user models found in {model_dir}")
    return encoder, models
